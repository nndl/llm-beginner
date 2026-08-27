"""手写字节级 BPE（Byte-level Byte Pair Encoding）分词器。

不依赖 tiktoken / sentencepiece。核心思路（参考 GPT-2 的字节级 BPE，简化版）：

1. 任何文本先按 UTF-8 编码成字节序列，字节的取值范围是 0~255，因此 256 个字节
   本身就构成了一个“天然完整”的基础词表——不会有超出词表的字符（OOV）。
   这是选择字节级而非字符级的原因：中文/英文/标点/emoji 混合文本也能无损处理。
2. 训练（train）：反复统计相邻 token 对的出现频率，把频率最高的一对合并成一个
   新 token，直到词表达到目标大小。合并规则本身（哪两个 id 合并成哪个新 id）
   就是 BPE 学到的“子词表”，记录为一个有序的 merges 列表。
3. 编码（encode）：给定新文本，先转成字节 id 序列，然后按 merges 学到的“优先级”
   （越早学到的合并，优先级越高）反复合并，直到无法再合并。
4. 解码（decode）：把 id 序列还原成对应的字节片段并拼接，再按 UTF-8 解码回字符串。
   因为合并/拆分是完全可逆的字节级操作，所以 decode(encode(text)) 恒等于 text，
   round-trip 天然成立（不依赖训练语料是否覆盖了这些字符）。
"""
from __future__ import annotations  # 允许在较低 Python 版本中延迟解析类型标注

import json  # 保存/加载 tokenizer 的 merge 规则
from pathlib import Path  # 使用跨平台路径对象读写 JSON 文件


def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计序列中所有相邻 (id, id) 对出现的次数。"""
    counts: dict[tuple[int, int], int] = {}  # pair -> 在当前 token 序列中的出现次数
    for a, b in zip(ids, ids[1:]):  # 把原序列与右移一位的序列配对，遍历所有相邻 token
        counts[(a, b)] = counts.get((a, b), 0) + 1  # 不存在时从 0 开始计数
    return counts  # 返回供 BPE 训练或编码选择 merge 的频率表


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把序列中所有出现的 pair 替换成 new_id，返回新序列。"""
    out = []  # 保存 merge 之后的新 token 序列
    i = 0  # 当前读取位置
    n = len(ids)  # 缓存序列长度，避免循环中重复计算
    while i < n:  # 从左到右扫描，保证不重叠地应用同一条 merge
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)  # 用新 token 替换命中的二元组
            i += 2  # 两个旧 token 已消费，跳过它们
        else:
            out.append(ids[i])  # 未命中时保留当前位置原 token
            i += 1  # 只向前移动一个 token
    return out  # 返回一次 merge 后的序列


class BPETokenizer:
    """字节级 BPE 分词器。

    属性：
        merges: dict[(id1, id2) -> new_id]，按学习顺序排列，new_id 越小代表越先学到、
            编码时优先级越高。
        vocab: dict[id -> bytes]，id 对应的原始字节片段，用于 decode。
    """

    def __init__(self, merges: dict[tuple[int, int], int] | None = None,
                 vocab: dict[int, bytes] | None = None):
        self.merges: dict[tuple[int, int], int] = merges or {}  # 保存 pair -> 新 token id 的规则
        if vocab is not None:
            self.vocab = vocab  # 调用方已提供完整词表时直接采用
        else:
            # 基础词表：256 个字节，id 0~255 与自身一一对应
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            self._rebuild_vocab_from_merges()

    def _rebuild_vocab_from_merges(self) -> None:
        """按 merges 的学习顺序，把新 id 的字节片段 = 两个子片段拼接。"""
        base_vocab = {idx: bytes([idx]) for idx in range(256)}  # 从 256 个单字节基础 token 重建
        # merges 按 new_id 升序即为学习顺序（new_id = 256 + i）
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            base_vocab[new_id] = base_vocab[a] + base_vocab[b]  # 新 token 的字节串由两个子 token 拼接而成
        self.vocab = base_vocab  # 用重建结果替换当前词表

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)  # 词表大小也就是 embedding/out head 所需类别数

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """在给定语料上训练 BPE 合并规则，目标词表大小为 vocab_size（含 256 基础字节）。"""
        assert vocab_size >= 256, "vocab_size 至少要 >= 256（基础字节数）"  # 不能删掉基础字节 token
        ids = list(text.encode("utf-8"))  # 初始 token 序列就是原始 UTF-8 字节值
        num_merges = vocab_size - 256  # 目标词表扣掉基础字节后，剩余可学习 merge 数

        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):  # 每轮恰好学习一条新的 BPE 合并规则
            counts = _get_pair_counts(ids)  # 在“已合并过”的当前序列上重新统计频率
            if not counts:
                break
            pair = max(counts, key=counts.get)  # 选择频率最高的相邻 pair
            new_id = 256 + i  # 新 token id 按训练轮次连续分配，编码时也代表优先级
            ids = _merge(ids, pair, new_id) # 将序列中的id替换为merge后的
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (i + 1) % 50 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"(freq={counts[pair]}), 当前序列长度={len(ids)}")

        self.merges = merges  # 持久化本次训练出的 merge 表
        self.vocab = vocab  # 持久化与 merge 表对应的字节词表

    def encode(self, text: str) -> list[int]:
        """把字符串编码成 token id 列表。"""
        ids = list(text.encode("utf-8"))  # 先把任意 Unicode 文本转换为基础字节 token
        if not self.merges:
            return ids
        while len(ids) >= 2:
            counts = _get_pair_counts(ids)  # 当前序列仍可能包含多个可应用的 pair
            # 选优先级最高（即最早学到、new_id 最小）的可合并 pair
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]  # 查询该 pair 在训练时分配的新 token id
            ids = _merge(ids, pair, new_id)  # 同一条规则可替换序列中所有不重叠出现位置
        return ids  # 没有更多训练过的 pair 可用时，当前序列即最终编码

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表还原成字符串。"""
        byte_chunks = b"".join(self.vocab[idx] for idx in ids)  # 逐 token 取回字节片段并拼接
        return byte_chunks.decode("utf-8", errors="replace")  # 拼回完整字节流后解码；非法采样片段替换显示

    def save(self, path: str) -> None:
        """保存为 JSON：merges 按学习顺序存成 [[a, b], ...] 列表（下标即学习顺序）。"""
        # merges 按 new_id (= 256 + i) 排序还原出学习顺序的 pair 列表
        ordered_pairs = [None] * len(self.merges)  # JSON 列表下标可隐式保存训练优先级
        for (a, b), new_id in self.merges.items():
            ordered_pairs[new_id - 256] = [a, b]  # id 256 对应第 0 条 merge，以此类推
        data = {
            "merges": ordered_pairs,
            "vocab_size": self.vocab_size,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")  # 写为可读 JSON

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # 读取保存的 merge 列表与元数据
        merges: dict[tuple[int, int], int] = {}
        for i, pair in enumerate(data["merges"]):
            merges[(pair[0], pair[1])] = 256 + i  # 用列表下标恢复训练时的新 token id
        tok = cls(merges=merges)  # 构造器会据此自动重建 id -> bytes 词表
        return tok  # 返回可直接 encode/decode 的 tokenizer
