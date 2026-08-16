"""PDF text normalization and character-based chunking."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List


def extract_pdf_text(pdf_path: str | Path) -> str:
    # 定义一个函数，从 PDF 文件中提取可读取的文本内容
    #
    # pdf_path：
    #   PDF 文件的路径，可以是字符串 str，也可以是 Path 对象
    #
    # -> str：
    #   表示该函数最终返回一个字符串，即 PDF 中提取出的全部文本
    #
    # 函数的作用：
    #   读取 PDF 的每一页，并将每页中的文本提取出来，
    #   最后把所有页面的文本拼接成一个完整的字符串
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))   # 创建 PDF 阅读器对象
    # 遍历 PDF 中的每一页，并提取文本
    #
    # page.extract_text()：
    #   尝试从当前页面中提取文本
    #
    # page.extract_text() or ""：
    #   如果当前页面没有提取到文本，extract_text() 可能返回 None，
    #   此时使用空字符串 "" 代替，避免后续拼接时报错
    #
    # "\n\n".join(...)：
    #   使用两个换行符将不同页面的文本连接起来
    #   这样可以在不同 PDF 页面之间保留明显的分隔
    #
    # 最终得到一个完整的字符串：
    #
    # 第1页文本
    #
    # 第2页文本
    #
    # 第3页文本
    #
    # ...
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # 定义文本切分函数
    #
    # text：
    #   需要切分的原始文本
    #
    # chunk_size：
    #   每个文本块的最大字符数
    #
    # overlap：
    #   相邻两个文本块之间重叠的字符数
    #
    # 返回值：
    #   List[str]，即由多个文本块组成的字符串列表
    #
    # 例如：
    #   原始文本：ABCDEFGHIJK
    #   chunk_size=5
    #   overlap=2
    #
    #   可能得到：
    #   ["ABCDE", "DEFGH", "GHIJK"]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    # 统一换行符
    #
    # Windows 中可能使用 "\r\n"
    # Mac/Linux 中可能使用 "\n"
    #
    # 这里将 "\r\n" 或 "\r" 统一转换成 "\n"
    text = re.sub(r"\r\n?", "\n", text)
    # 将连续的空格和 Tab 压缩成一个普通空格
    #
    # 例如：
    # "hello     world" → "hello world"
    # "hello\t\tworld"  → "hello world"
    #
    # 这样可以减少 PDF 提取文本后产生的大量无意义空格。
    text = re.sub(r"[ \t]+", " ", text)
    # 将连续 3 个及以上的换行符压缩成两个换行符
    #
    # 例如：
    # "\n\n\n\n" → "\n\n"
    #
    # 保留两个换行符通常可以表示一个段落之间的分隔。
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # 如果清洗之后文本为空，则直接返回空列表
    if not text:
        return []

    chunks: List[str] = []    # 创建一个列表，用于保存最终切分得到的所有文本块
    start = 0   # 第一次从文本的第 0 个字符开始
    text_length = len(text)  # 获取文本总字符数
    while start < text_length:   # 只要当前起始位置还没有到达文本末尾，就继续切分
        # 计算当前文本块的理论结束位置
        #
        # start + chunk_size：
        #   从当前 start 开始最多取 chunk_size 个字符
        #
        # min(...)：
        #   防止 end 超过文本总长度
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # 如果当前 end 还没有到达文本末尾，
            # 说明当前文本后面还有内容，
            # 此时可以尝试寻找一个更自然的切分位置。
            window = text[start:end]   # 获取当前候选文本窗口
            # 尝试寻找自然的断句位置
            #
            # rfind() 从右向左查找最后一次出现的位置。
            #
            # 优先寻找：
            #   "\n\n"：段落结束
            #   "。"：中文句号
            #   "！"：中文感叹号
            #   "？"：中文问号
            #
            # max() 会选择这些位置中最靠后的一个，
            # 从而尽可能让当前 chunk 接近 chunk_size。
            break_at = max(window.rfind("\n\n"), window.rfind("。"), window.rfind("！"), window.rfind("？"))
            # 判断找到的断点是否足够靠后
            #
            # break_at >= chunk_size // 2：
            #   只有当断点位于当前窗口的后半部分时，
            #   才使用这个自然断点。
            #
            # 这样可以避免出现：
            #   chunk_size=500
            #   但刚找到一个句号在第 20 个字符处，
            #   就提前结束 chunk，
            #   导致文本块过短。
            if break_at >= chunk_size // 2:
                # 将 end 调整到自然断点之后
                #
                # +1 是为了把句号/感叹号/问号本身包含在当前 chunk 中。
                end = start + break_at + 1

        # 获取当前文本块
        #
        # strip()：
        #   去除文本块首尾多余的空格和换行符
        chunk = text[start:end].strip()
        # 如果当前文本块不是空字符串，
        # 就加入最终结果列表
        if chunk:
            chunks.append(chunk)
        # 如果已经到达文本末尾，
        # 说明所有文本都已经处理完成，
        # 直接退出循环。
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks  # 返回所有切分得到的文本块