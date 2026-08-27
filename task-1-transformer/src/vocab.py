"""字符级词表：简单、无需分词器/预训练模型，适合 ChnSentiCorp 短文本情感分类。"""
from collections import Counter  # 用于统计字符出现频率，方便按频率截取高频字符建表

# 三个特殊符号的固定 id，模型的 embedding 层第 0/1/2 行分别对应它们
PAD_ID = 0  # 填充符：补齐短文本到统一长度用，不参与语义
UNK_ID = 1  # 未登录字：词表里没有的字符都映射到这个 id
CLS_ID = 2  # 分类符：放在序列开头，通常用它的输出向量做分类
NUM_SPECIAL = 3  # PAD / UNK / CLS，特殊符号总数，建表时要预留这三个位置


def build_vocab(texts, max_vocab_size: int = 6000) -> dict:
    """从训练文本里按字符频率建表，返回 char -> id 映射（含特殊符号）。"""
    counter = Counter()  # 用 Counter 统计每个字符出现的次数
    for text in texts:  # 遍历传入的每一条文本（如训练集的所有句子）
        counter.update(str(text))  # str(text) 防止非字符串类型报错；update 会把字符串拆成单字符逐个计数

    # 先把三个特殊符号按固定 id 放进词表
    char2id = {"[PAD]": PAD_ID, "[UNK]": UNK_ID, "[CLS]": CLS_ID}
    # 从计数结果里取出频率最高的若干字符，数量 = 总词表大小减去特殊符号占用的 3 个位置
    most_common = counter.most_common(max_vocab_size - NUM_SPECIAL)
    for ch, _ in most_common:  # most_common 返回 (字符, 频次) 元组列表，这里只需要字符本身
        char2id[ch] = len(char2id)  # 按加入顺序（即频率从高到低）分配递增的 id
    return char2id  # 返回完整的 字符 -> id 映射表


def encode(text: str, char2id: dict, max_len: int) -> list:
    """[CLS] + 字符 id，截断/补齐到 max_len，返回 python list[int]。"""
    ids = [CLS_ID]  # 序列第一个 token 固定是 [CLS]，占用一个长度位
    for ch in str(text)[: max_len - 1]:  # 只取前 max_len - 1 个字符，为 [CLS] 让出位置
        ids.append(char2id.get(ch, UNK_ID))  # 查表转 id；字符不在词表里就用 UNK_ID 兜底
    ids = ids[:max_len]  # 双重保险：确保长度不超过 max_len（正常情况下上面的切片已保证）
    ids += [PAD_ID] * (max_len - len(ids))  # 长度不足 max_len 时，用 PAD_ID 补齐到定长
    return ids  # 返回定长的 id 列表，供模型批量输入使用
