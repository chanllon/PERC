import jieba,nltk,random,time,re,os
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from nltk.corpus import wordnet as wn
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil

# pip install googletrans==4.0.0-rc1
# nltk.download('omw-1.4')



generated_cache = {}
current_position = {}

global tokenizer, model, device
tokenizer = None
model = None
device = torch.device("cuda:1")

translator = Translator(service_urls=[ 
      'translate.google.com' 
    ]) #接入谷歌的翻译服务，设定service_urls为translate.google.cn即可翻译中文
translator.raise_Exception = True  #避免ip被限制


def is_chinese(char):
    """检查字符是否为中文。"""
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def get_teaching_methods(label):
    # teaching_methods = {
    #     0: "单点式教学",
    #     1: "启发式教学",
    #     2: "辩论式教学",
    #     3: "情景式教学",
    #     4: "案例式教学",
    #     5: "研究型教学",
    #     6: "基于问题的教学",
    #     7: "项目式教学",
    #     8: "换位式教学",
    #     9: "比较式教学",
    #     10: "讲座式教学",
    #     11: "头脑风暴式教学",
    #     12: "协作式教学",
    #     13: "支架式教学",
    #     14: "探究式学习"
    # }
    # return teaching_methods.get(label, "未知教学模式")
    
    # course_type = {
    #     0: "新授课",
    #     1: "复习课",
    #     2: "讲评课",
    #     3: "练习课",
    #     4: "测试课",
    #     5: "自学课",
    #     6: "讲授式课",
    #     7: "抛锚式课",
    #     8: "合作式课",
    #     9: "实践活动课"
    # }
    # return course_type.get(label, "未知课型")

    ability_elements = {
        0 : "A1",
        1 : "A2",
        2 : "A3",
        3 : "B1",
        4 : "B2",
        5 : "B3",
        6 : "C1",
        7 : "C2",
        8 : "C3"
    }
    return ability_elements.get(label, "未知能力要素")


def synonym_replace_nlpaug(text, label=None, probability=0.5, word_interval=2):
    words = list(jieba.cut(text))
    new_text = [''] * len(words)  # 创建一个空列表来存储新文本
    words_to_translate = []
    indices_to_translate = []  # 保存需要翻译的词汇的索引位置

    for i, word in enumerate(words):
        try:
            flag = all(is_chinese(char) for char in word)
            if i % word_interval == 0 and random.random() < probability:
                if flag:
                    synsets = wn.synsets(word, lang='cmn')
                else:
                    synsets = wn.synsets(word)
                if synsets and synsets[0].lemmas():
                    lemma = random.choice(synsets[0].lemmas())
                    word = lemma.name()
                    if flag and word is not None: 
                        words_to_translate.append(word)
                        indices_to_translate.append(i)
                    else:
                        new_text[i] = word
                else:
                    new_text[i] = word
            else:
                new_text[i] = word
        except Exception as e:
            print(f"处理单词 '{word}' 时发生错误: {e}")
            new_text[i] = word

    time.sleep(random.uniform(2,7))
    # 翻译收集到的词汇
    if words_to_translate:
        try:
            words_to_translate_str = ','.join(words_to_translate)
            translated_results = translator.translate(words_to_translate_str, dest='zh-CN')
            translated_words = translated_results.text.split('，') if getattr(translated_results, 'text', None) is not None else []
            # print(translated_words)
            for idx, result in zip(indices_to_translate, translated_words):
                # 检查翻译结果是否为None，如果是，则使用原始单词
                new_text[idx] = result if result is not None else words_to_translate[indices_to_translate.index(idx)]
        except Exception as e:
            print(f"批量翻译时发生错误: {e}")
            # 如果翻译失败，使用原始的英文单词
            for idx in indices_to_translate:
                new_text[idx] = words_to_translate[indices_to_translate.index(idx)]

    return ''.join(new_text)


def delete_text(text, label=None):
    words = list(jieba.cut(text))
    num_words = len(words)
    num_to_delete = random.randint(num_words // 10, max(1, num_words // 2))  # 计算要删除的词语数量
    indices_to_delete = sorted(random.sample(range(num_words), num_to_delete), reverse=True)  # 从后往前排序索引

    for index in indices_to_delete:
        del words[index]  # 删除指定索引的词语

    return ''.join(words)


def shuffle_random_sentences(text, label=None, min_shuffle=1):
    # 使用多种标点符号分割句子
    sentences = [s for s in re.split('(\。|\.|\!|\?|\！|\？|\，|\,|\:|\(|\)|\（|\)|\")', text) if s]
    if len(sentences) == 1:
        words = list(jieba.cut(text))
        random.shuffle(words)
        return ' '.join(words)
    num_sentences_to_shuffle = min(random.randint(min_shuffle, len(sentences)), len(sentences))
    # 随机选择句子进行打乱
    indices_to_shuffle = random.sample(range(len(sentences)), num_sentences_to_shuffle)
    for i in indices_to_shuffle:
        # 排除所有用作分割的标点符号
        if sentences[i] not in ['。', '.', '!', '?', '！', '？', '，', ',', ':', '(', ')', '（', '）', '\"']:
            words = list(jieba.cut(sentences[i]))
            random.shuffle(words)
            sentences[i] = ''.join(words)
    # 重新组合句子
    return ''.join(sentences)

# 每次都生成新的
# def insert_text(text, label, tokenizer=None, model=None):
#     # 生成新的内容
#     if len(text) > 4000:
#         prompt_text = text[:4000]
#     else: prompt_text = text
#     user_message = f"请你参考下面我发给你的这个教学资源的内容，帮我写一个教学模式是{get_teaching_methods(label)}的教学资源：{prompt_text}"
#     messages = [{"role": "user", "content": user_message}]
#     input_ids = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)
#     outputs = model.generate(
#         input_ids,
#         max_new_tokens=300,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     generated_response = outputs[0][input_ids.shape[-1]:]
#     generated_text = tokenizer.decode(generated_response, skip_special_tokens=True)

#     # 清理生成的文本
#     generated_text_cleaned = generated_text.replace('\n', ' ')

#     # 将生成的文本插入到原始文本的随机位置
#     words = list(jieba.cut(text))
#     insert_position = random.randint(0, 3500)    
#     augmented_text = ''.join(words[:insert_position] + [generated_text_cleaned] + words[insert_position:])

#     # global reset_count
#     # if reset_count > 5:
#     #     model.reset_states()
#     #     reset_count = 0
#     # else: reset_count += 1

#     return augmented_text
    
# 生成一堆，放到缓存里，每次用一点
def insert_text(text, label, tokenizer=None, model=None, max_new_tokens=7000, chunk_size=300):
    global generated_cache, current_position

    # 如果label更新了，或者当前label的缓存已用完，重新生成文本
    if label not in generated_cache or current_position[label] + chunk_size > len(generated_cache[label]):
        # 生成新的内容
        prompt_text = text[:4000] if len(text) > 4000 else text
        user_message = f"请你参考下面我发给你的这个教学资源的内容，帮我写一个能力要素是{get_teaching_methods(label)}的教学资源：{prompt_text}"
        messages = [{"role": "user", "content": user_message}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        generated_response = outputs[0][input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_response, skip_special_tokens=True)
        generated_text_cleaned = generated_text.replace('\n', ' ')

        # 存储生成的文本和当前位置
        generated_cache[label] = generated_text_cleaned
        current_position[label] = 0

    # 计算剩余可用的文本长度
    remaining_text_length = len(generated_cache[label]) - current_position[label]
    # 如果剩余文本不足chunk_size，则使用剩余的所有文本
    actual_chunk_size = min(chunk_size, remaining_text_length)
    # 从缓存中获取文本
    chunk = generated_cache[label][current_position[label]:current_position[label] + actual_chunk_size]
    current_position[label] += actual_chunk_size

    # 将获取的文本插入到原始文本的随机位置
    words = list(jieba.cut(text))
    insert_position = random.randint(0, len(words))
    augmented_text = ''.join(words[:insert_position] + [chunk] + words[insert_position:])

    return augmented_text


def data_augmentation(text, label):
    global tokenizer, model, device

    try: 
        augmentation_functions = [synonym_replace_nlpaug, delete_text, shuffle_random_sentences]
        if tokenizer is not None and model is not None:
            augmentation_functions.append(lambda t, l: insert_text(t, l, tokenizer, model))
        augmentation_function = random.choice(augmentation_functions)
        # print(f"choose: {augmentation_function}")
        return augmentation_function(text,label)
    except Exception as e:
        print(f"An exception occurred: {e}")

        # Update the device
        device_id = int(str(device).split(":")[-1])  # Get the current device id
        device_id = (device_id + 1) % 4  # Update the device id
        device = torch.device(f"cuda:{device_id}")  # Create new device
        
        model_id = "/home/zhjy/cz/program/Llama/shenzhi-wang/Llama3-8B-Chinese-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto"
        ).to(device)
        return data_augmentation(text.replace("\t", " ").replace("\n", " "), label)


def read_file_to_augu(filename, max_augu_num):
    
    # copy origin datas from old file to new file if new file is not exist
    new_filename = os.path.splitext(filename)[0] + f'_math_augmented_{max_augu_num}.txt'
    if not os.path.exists(new_filename):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(new_filename, 'w', encoding='utf-8') as new_file:
            for line in lines:
                id, temp = line.strip().split('\t', 1)
                text, label = temp.rsplit('\t', 1)
                # text, label, _ = temp.rsplit('\t', 2)
                # text, _, label = temp.rsplit('\t', 2)
                new_file.write(f"{id}\t{text}\t{label}\n")

    # 读取文件并统计每个类别的数量
    label_counts = {}
    data_by_label = {}
    max_id = 0  # 初始化最大ID
    with open(new_filename, 'r', encoding='utf-8') as file:
        for line in file:
            id, temp = line.strip().split('\t', 1)
            text, label = temp.rsplit('\t', 1)
            id = int(id)
            max_id = max(max_id, id)  # 更新最大ID
            label_counts[label] = label_counts.get(label, -1) + 1
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append((id, text, label))

    # 数据增强
    
    # 无进度条
    # for label, texts in data_by_label.items():
    #     count = label_counts[label]
    #     while count < max_augu_num:
    #         for id, text, label in texts:
    #             if count >= max_augu_num:
    #                 break
    #             augmented_text = data_augmentation(text, int(label), tokenizer, model)
    #             max_id += 1  # 为增强后的数据生成新的ID
    #             augmented_data.append((max_id, augmented_text, label))
    #             count += 1

    # 有进度条
    for label, texts in tqdm(data_by_label.items(), desc='Augmenting data', unit='label'):
        count = label_counts[label]
        progress_bar = tqdm(total=max_augu_num, initial=count, desc=f'Processing label {label}', unit='text')
        augmented_data = []
        while count < max_augu_num:
            for id, text, label in texts:
                if count >= max_augu_num:
                    break
                augmented_text = data_augmentation(text=text, label=int(label))
                augmented_text = augmented_text.replace('\t', '')
                max_id += 1
                augmented_data.append((max_id, augmented_text, label))
                count += 1
                progress_bar.update(1)
        progress_bar.close()
        
        # Write the augmented data to the new file after each label's augmentation is completed
        with open(new_filename, 'a+', encoding='utf-8') as file:
        # Check if the last character of the file is a newline. If not, add one.
            file.seek(0, os.SEEK_END)
            if file.tell() > 0:
                file.seek(file.tell() - 1, os.SEEK_SET)
                if file.read(1) != '\n':
                    file.write('\n')

            # Now write the augmented data
            for id, augmented_text, label in augmented_data:
                file.write(f"{id}\t{augmented_text}\t{label}\n")
    
    with open(new_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    random.shuffle(lines)
    with open(new_filename, 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line)

    return new_filename 


def main():

    global tokenizer, model, device
    model_id = "/home/zhjy/cz/program/Llama/shenzhi-wang/Llama3-8B-Chinese-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, torch_dtype="auto", device_map="auto"
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto"
    ).to(device)
    filename = "/home/zhjy/cz/program/data_process/math_nlys/train.txt"
    read_file_to_augu(filename, 400)


main()