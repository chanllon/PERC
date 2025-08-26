

from mamba_ssm import Mamba2
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score
from transformers import LongformerModel
from transformers import AutoModel
from torch.nn import CrossEntropyLoss
from longformer.sliding_chunks import pad_to_window_size
import torchmetrics
# from longformer.longformer import Longformer, LongformerConfig
import torch, gzip, argparse, glob, os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace
from transformers import AdamW
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

import logging
logger = logging.getLogger(__name__)

TEXT_FIELD_NAME = "text"
LABEL_FIELD_NAME = 'label'

class ClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, seqlen, label_define, num_samples=None, mask_padding_with_zero=True):
        self.data = []
        
        with (gzip.open(file_path, 'rt') if file_path.endswith('.gz') else open(file_path,encoding='utf-8')) as fin:
            for i, line in enumerate(tqdm(fin, desc=f'loading input file {file_path.split("/")[-1]}', unit_scale=1)):
                # items = line.strip().split('\tSEP\t')
                # items = line.strip().split('\t')
                _, temp = line.strip().split('\t', 1)
                text, label = temp.rsplit('\t', 1)
                # if len(items) != 10: continue
                self.data.append({
                    # "text": items[0]+items[1],
                    "text":  text,
                    "label": label
                })
                if num_samples and len(self.data) > num_samples:
                    break
                
        # with open(label_define, 'r', encoding='utf-8') as f:
        #     self.label_define = f.read()
        self.label_define = self.read_all_label_files_in_directory(label_define)
            
            
        self.seqlen = seqlen
        self._tokenizer = tokenizer
        all_labels = list(set([e[LABEL_FIELD_NAME] for e in self.data]))
        self.label_to_idx = {e: i for i, e in enumerate(sorted(all_labels))}
        # print("label_to_index: ", self.label_to_idx)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.mask_padding_with_zero = mask_padding_with_zero

    def read_all_label_files_in_directory(self, directory_path):
        label_define_list = []

        # 遍历目录下的所有文件
        for filename in os.listdir(directory_path):
            # 检查文件是否是txt文件
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                # 读取txt文件内容并添加到列表中
                with open(file_path, 'r', encoding='utf-8') as f:
                    label_define_list.append(f.read())

        return label_define_list
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._convert_to_tensors(self.data[idx])

    def _convert_to_tensors(self, instance):
        def tok(s):
            return self._tokenizer.tokenize(s)
        # tokens = [self._tokenizer.cls_token] + tok(instance[TEXT_FIELD_NAME])

        tokens = [self._tokenizer.cls_token] + tok(instance[TEXT_FIELD_NAME][:self.seqlen-2])

        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:self.seqlen-1] +[self._tokenizer.sep_token_id]
        
        input_len = len(token_ids)
        attention_mask = [1 if self.mask_padding_with_zero else 0] * input_len
        
        padding_length = self.seqlen - input_len
        token_ids = token_ids + ([self._tokenizer.pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)

        assert len(token_ids) == self.seqlen, "Error with input length {} vs {}".format(
            len(token_ids), self.seqlen
        )
        assert len(attention_mask) == self.seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), self.seqlen
        )

        label = self.label_to_idx[instance[LABEL_FIELD_NAME]]
        label_define_tensors = []
        for each_label in self.label_define:
            label_define_tokens = tok(each_label)
            
            label_define_ids = self._tokenizer.convert_tokens_to_ids(label_define_tokens)
            label_padding_length = self.seqlen - len(label_define_ids)
            label_define_ids = label_define_ids + ([self._tokenizer.pad_token_id] * label_padding_length)
            label_define_tensor = torch.tensor(label_define_ids)
            label_define_tensors.append(label_define_tensor)

        return (torch.tensor(token_ids), torch.tensor(attention_mask), label_define_tensors, torch.tensor(label))


        
class TextChannelAttention(nn.Module):
    def __init__(self, in_planes, hidden_size, ratio=16):
        super(TextChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes // 16, hidden_size, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class TextSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TextSpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class LongformerClassifier(pl.LightningModule):
    
    def __init__(self, init_args):
        super().__init__()
        if isinstance(init_args, dict):
            init_args = Namespace(**init_args)
        self.init_args = init_args

        logger.info(f'loading model from config: {self.init_args.config_path}, model: {self.init_args.model_dir}')
        config_path = init_args.config_path or init_args.model_dir
        # config = LongformerConfig.from_pretrained(config_path)
        # config.attention_mode = init_args.attention_mode
        logger.info(f'attention mode set to {init_args.attention_mode}')
        
        self.mamba2_1 = Mamba2(d_model=768, d_state=64, d_conv=4, expand=16)
        self.mamba2 = Mamba2(d_model=768, d_state=64, d_conv=4, expand=16)
        self.model = AutoModel.from_pretrained(self.init_args.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.init_args.tokenizer)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.model.config.attention_mode = init_args.attention_mode
        

        # self.auroc = AUROC(num_classes=init_args.num_labels, task='multiclass')
        self.auroc = MulticlassAUROC(num_classes=init_args.num_labels, average="macro", thresholds=None)
        self.f1 = MulticlassF1Score(num_classes=init_args.num_labels)
        # self.mae = MeanAbsoluteError()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        
        # self.classifier = nn.Linear(self.model.config.hidden_size,self.init_args.num_labels)
        self.text_channel_attention = TextChannelAttention(self.model.config.hidden_size, self.model.config.hidden_size)
        self.text_spatial_attention = TextSpatialAttention()
        
        self.classifier = nn.Linear(init_args.num_labels, init_args.num_labels)

        # Create a linear layer for the Mamba model's output
        self.mamba_linear = nn.Linear(self.model.config.max_position_embeddings, init_args.num_labels)
        self.former_linear = nn.Linear(self.model.config.hidden_size, init_args.num_labels)
        
        self.s_c_attention_output_linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

        reduced_dim = self.tokenizer.model_max_length // init_args.num_labels
        self.seq_linear = nn.Linear(reduced_dim * init_args.num_labels, self.tokenizer.model_max_length)
        self.label_linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

        # self.accumulated_linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.each_label_linears = nn.ModuleList([nn.Linear(self.tokenizer.model_max_length, reduced_dim) for _ in range(init_args.num_labels)])
        
    def forward(self, input_ids, attention_mask, label_define, labels=None):
                
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.model.config.attention_window[0], self.tokenizer.pad_token_id)
        
        attention_mask[:, 0] = 2  # global attention for the first token
        
     
        # Convert tokenized text to tensor and pass it through the embedding layer and Mamba model
        accumulated_mamba_output = None
        previous_mamba_output = None
        reduced_outputs = []
        for each_label, each_label_linear in zip(label_define, self.each_label_linears):
            embedded_text = self.model.embeddings(each_label)
            mamba_output = self.mamba2(embedded_text)
            
            # if previous_mamba_output is not None:
            #     # mamba_output = (mamba_output + previous_mamba_output) / 2
            #     mamba_output = mamba_output + previous_mamba_output
                
            # if accumulated_mamba_output is None:
            #     accumulated_mamba_output = each_label_linear(mamba_output)
            # else:
            #     accumulated_mamba_output += each_label_linear(mamba_output)
            
            mamba_output_transposed = mamba_output.permute(0, 2, 1)
            # previous_mamba_output = mamba_output
            reduced_output = each_label_linear(mamba_output_transposed)
            reduced_outputs.append(reduced_output)

        accumulated_mamba_output = torch.cat(reduced_outputs, dim=-1)
        accumulated_mamba_output = self.seq_linear(accumulated_mamba_output)
        accumulated_mamba_output = accumulated_mamba_output.permute(0, 2, 1)
        # embedded_text = self.model.embeddings(label_define)
        # mamba_output = self.mamba2(embedded_text)
              
        
        
        #longformer output
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        sequence_output = self.mamba2_1(sequence_output)
        
        # CBAM handle, 处理output
        s_c_attention_output = sequence_output.permute(0, 2, 1)
        s_c_attention_output = s_c_attention_output.float()
        s_c_attention_output = self.text_channel_attention(s_c_attention_output) * s_c_attention_output
        s_c_attention_output = self.text_spatial_attention(s_c_attention_output) * s_c_attention_output
        s_c_attention_output = s_c_attention_output.permute(0, 2, 1)
        

        # mamba_output = mamba_output * s_c_attention_output
        s_c_attention_output = self.s_c_attention_output_linear(s_c_attention_output)
        mamba_output = self.label_linear(accumulated_mamba_output)
        distances = torch.sqrt(torch.sum((mamba_output - s_c_attention_output) ** 2, dim=2))
        # mamba_output = s_c_attention_output
        mamba_out = self.mamba_linear(distances)
        
        pooled_out = self.former_linear(pooled_output)
        
        # result = torch.cat([mamba_out, pooled_out],dim=1)
        result = (mamba_out + pooled_out)/2
        # result = pooled_out
        
        logits = self.classifier(result)
        
    

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.init_args.num_labels), labels.view(-1))
            # criterion = FocalLoss()
            # print('label形状',labels.size())
            # loss = criterion(logits.view(-1, self.init_args.num_labels), labels.unsqueeze(1).expand(-1,15))
        return logits, loss
    
    def _get_loader(self, split, shuffle=False):
        if split == 'train':
            fname = self.init_args.train_file
        elif split == 'dev':
            fname = self.init_args.dev_file
        elif split == 'test':
            fname = self.init_args.test_file
        else:
            assert False
        is_train = split == 'train'

        dataset = ClassificationDataset(
            fname, tokenizer=self.tokenizer, seqlen=self.init_args.seqlen, num_samples=self.init_args.num_samples, 
            label_define = self.init_args.label_define
        )

        loader = DataLoader(dataset, batch_size=self.init_args.batch_size, num_workers=self.init_args.num_workers, 
                            shuffle=(shuffle and is_train), pin_memory = self.init_args.pin_memory)
        return loader
        
    def setup(self, stage):
        self.train_loader = self._get_loader("train")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 或者 "epoch"，取决于你的调度器是按步骤还是按周期更新
                "monitor": "val_loss"
            },
        }

    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, label_define, labels= batch
        logits, loss = self(input_ids, attention_mask, label_define, labels)
        # self.log('train_loss', loss)
        self.log('tr_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # if self.trainer.lr_schedulers:
        #     lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        #     current_lr = lr_scheduler.get_last_lr()[-1]
        #     self.log('current_lr', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.trainer.optimizers:
        # 获取第一个优化器的当前学习率
            cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('cur_lr', cur_lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        # do something with all training_step outputs, for example:
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        self.log("training_loss_mean", epoch_mean, prog_bar=False)

        # print(self.trainer.optimizers[0].state_dict())
        
        # free up the memory
        self.training_step_outputs.clear()

    
    def validation_step(self, batch, batch_idx):
        inputs, attention_mask, label_define, target = batch
        logits, loss = self(inputs, attention_mask, label_define, target)
        pred = logits.softmax(dim=-1)
        self.validation_step_outputs.append((pred, target))

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return pred
    
    def on_validation_epoch_end(self):
        all_preds, all_targets = zip(*self.validation_step_outputs)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        acc = torchmetrics.functional.accuracy(all_preds, all_targets, task='multiclass', num_classes=self.init_args.num_labels)
        auroc = self.auroc(all_preds, all_targets)
        f1 = self.f1(all_preds, all_targets)
        
        self.log('auc', auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.validation_step_outputs.clear()  # free memory
        
    def test_step(self, batch, batch_idx):
        inputs, attention_mask, label_define, target = batch
        logits, loss = self(inputs, attention_mask, label_define, target)
        pred = logits.softmax(dim=-1)
        self.test_step_outputs.append((pred, target))

        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return pred
    
    def on_test_epoch_end(self):
        all_preds, all_targets = zip(*self.test_step_outputs)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        acc = torchmetrics.functional.accuracy(all_preds, all_targets, task='multiclass', num_classes=self.init_args.num_labels)
        auroc = self.auroc(all_preds, all_targets)
        f1 = self.f1(all_preds, all_targets)
        
        self.log('test_auc', auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        
        self.test_step_outputs.clear()  # free memory

def parse_args():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--config_path', type=str, default= "./longformer-chinese-base-4096", help='Path to the model configuration file.')
    parser.add_argument('--model_dir', type=str, default= "./longformer-chinese-base-4096", help='Directory where the model is stored.')
    parser.add_argument('--tokenizer', type=str, default= "./longformer-chinese-base-4096", help='Tokenizer to be used.')
    parser.add_argument('--num_labels', type=int, default=-1, help='Number of labels for classification.')
    parser.add_argument('--attention_mode', default='sliding_chunks')
    parser.add_argument('--label_define', type=str, default="/home/zhjy/cz/program/DA-LongRefine/labe_define/2ability_element")

    # Data configuration
    parser.add_argument('--train_file', type=str, default='/home/zhjy/cz/program/DA-LongRefine/marksources_more518/kx_augument_to_1600/518kx_train_augmented.txt', help='Path to the training data file.')
    parser.add_argument('--dev_file', type=str, default='/home/zhjy/cz/program/DA-LongRefine/marksources_more518/518kx_origin/dev_with_80.txt', help='Path to the development data file.')
    parser.add_argument('--test_file', type=str, default='/home/zhjy/cz/program/DA-LongRefine/marksources_more518/518kx_origin/dev.txt', help='Path to the test data file.')
    parser.add_argument('--input_dir', type=str, default=None, help='optionally provide a directory of the data and train/test/dev files will be automatically detected')
    parser.add_argument('--seqlen', type=int, default=4096, help='Maximum sequence length.')
    
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to load from the data files.')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    # 可以在不同的epoch上面选择不同的梯度积累：https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
    parser.add_argument('--grad_accum', type=int, default=1)
    # https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html
    parser.add_argument('--accelerator', type=str, default= "auto", help='Input gpu or cpu will run on specified device')
    parser.add_argument('--devices', default="auto", help = "gpu nmuber")
    parser.add_argument('--precision', type=str, default='bf16-mixed', help='16, 32, or bf16-mixed')
    parser.add_argument('--seed', type=int, default=1996)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Use pinned memory for data loading')
    parser.add_argument('--lr_scheduler', type=str, default='linear', choices=arg_to_scheduler_choices, metavar=arg_to_scheduler_metavar, help='Learning rate scheduler.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay for optimizer.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--model_save_path', type=str, default="./v2_saved_log_label_model/model/", help='path to the model (if not setting checkpoint)')
    parser.add_argument('--log_save_path', type=str, default="./v2_saved_log_label_model/log/", help='path to the model (if not setting checkpoint)')
    
    # Test configuration
    parser.add_argument('--load_ckpt_path', default=None, type=str, help='path to the model (eg: path/to/checkpoint.ckpt)')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--do_predict', default=False, action='store_true')
    

    args = parser.parse_args()
    
    if args.input_dir is not None:
        files = glob.glob(args.input_dir + '/*')
        for f in files:
            fname = f.split('/')[-1]
            if 'train' in fname:
                args.train_file = f
            elif 'dev' in fname or 'val' in fname:
                args.dev_file = f
            elif 'test' in fname:
                args.test_file = f
                
    return args


def get_train_params(args):
    train_params = {}
    train_params["accelerator"] = args.accelerator
    train_params["devices"] = args.devices
    train_params["precision"] = args.precision
    train_params["default_root_dir"] = args.model_save_path
    train_params["max_epochs"] = args.max_epochs
    train_params["accumulate_grad_batches"] = args.grad_accum
    
    # train_params['track_grad_norm'] = -1 这玩意在2.2.3没了，可以换成def on_before_optimizer_step，
    #                       如果发现范数上升，可能梯度爆炸了，可以在Trainer里面加入gradient_clip_val=0.5把梯度全局范数裁剪到小于0.5
    # train_params['limit_val_batches'] = args.limit_val_batches
    # train_params['val_check_interval'] = args.val_check_interval
    return train_params

def infer_num_labels(args):
    if isinstance(args.tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    # Dataset will be constructred inside model, here we just want to read labels (seq len doesn't matter here)
    ds = ClassificationDataset(args.train_file, tokenizer=tokenizer, seqlen=args.seqlen, label_define=args.label_define)
    num_labels = len(ds.label_to_idx)
    return num_labels

def main():
    args = parse_args()
    print(f"Tokenizer: {args.tokenizer}")
    seed_everything(args.seed, workers=True)
    train_params = get_train_params(args)
    args.num_labels = infer_num_labels(args)
    
    # 定义checkpoint回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # 监控验证集上的loss
        mode='max',  # 我们希望最小化loss
        save_top_k=1,  # 只保存最好的一个模型
        verbose=True,  # 打印保存模型的信息
        save_last=True,  # 除了最好的模型，也保存最后一个模型
        dirpath=args.model_save_path,  # 模型保存路径
        filename='best-checkpoint{epoch:02d}-{val_acc:.2f}',  # 最佳模型的保存名称
    )

    train_params["callbacks"] = [checkpoint_callback]
    
    logger = TensorBoardLogger(save_dir=args.log_save_path, name="my_model")
    train_params["logger"] = logger

    # 初始化模型
    
    model = LongformerClassifier(args)

    # 初始化训练器
    trainer = pl.Trainer(**train_params)

    # 如果不是仅测试，则进行训练和验证
    if not args.test_only:
        trainer.fit(model)
    else:
        # 加载测试模型
        test_model_path = args.load_ckpt_path
        if test_model_path is None:
             raise ValueError("Please provide --load_ckpt_path paramater (example: path/to/checkpoint.ckpt)")
        model = LongformerClassifier.load_from_checkpoint(test_model_path, init_args=args)

    # 测试
    print("Test best model:")
    trainer.test(model)

    if not args.test_only:
        # 加载最后一个模型
        last_checkpoint_path = os.path.join(args.model_save_path, "last.ckpt")
        if os.path.exists(last_checkpoint_path):
            last_model = LongformerClassifier.load_from_checkpoint(last_checkpoint_path, init_args=args)
            print("Test last model:")
            # 测试最后一个模型
            trainer.test(last_model)


    # 如果需要进行预测
    # if args.do_predict:
    #     # 这里添加你的预测代码
    #     pass
    

if __name__ == "__main__":
    main()

