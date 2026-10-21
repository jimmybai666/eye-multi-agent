"""PDF 文本提取器 - 从临床指南 PDF 中提取结构化文本

支持：
- 逐页提取文本（基于 block 级别，过滤图片/噪音）
- 合并表格碎片，保留完整段落
- 按章节标题智能切分
- 大章节自动递归切分
- chunk 前附加章节标题上下文
- 过滤过短/无意义 chunk
"""

import re
import hashlib
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CHUNK_CHARS = 80
MIN_BLOCK_CHARS = 5

# 过滤掉这些章节标题（目录、页眉等噪音）
SKIP_SECTION_TITLES = {
    "TABLE OF CONTENTS",
    "CONTENTS",
    "LIST OF TABLES",
    "LIST OF FIGURES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "ABBREVIATIONS",
}


class Section:
    """章节信息"""
    def __init__(self, title: str, content: str, page_start: int):
        self.title = title
        self.content = content
        self.page_start = page_start


def _clean_block_text(text: str) -> str:
    """清理单个 block 的文本"""
    text = text.strip()
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
    return text


def _merge_short_blocks(blocks: List[str], min_length: int = 80) -> List[str]:
    """
    合并连续的短文本块。

    PDF 表格/列表经常被 PyMuPDF 切成很多短 block，
    把连续的短 block 合并成一段，保留完整语义。
    """
    if not blocks:
        return []

    merged = []
    buffer = ""

    for block in blocks:
        if not block.strip():
            continue

        if len(buffer) == 0:
            buffer = block
        elif len(buffer) < min_length or len(block) < min_length:
            buffer = buffer + "\n" + block
        else:
            merged.append(buffer)
            buffer = block

    if buffer:
        merged.append(buffer)

    return merged


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 提取全文文本（基于 block 级别提取，过滤图片块）

    Args:
        pdf_path: PDF 文件路径

    Returns:
        全文文本（带页码标记）
    """
    import pymupdf as fitz

    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")

        page_texts = []
        for block in blocks:
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            # block_type: 0=text, 1=image
            if block[4] is None:
                continue
            block_type = block[6] if len(block) > 6 else 0
            if block_type == 1:
                continue

            text = _clean_block_text(block[4])
            if len(text) < MIN_BLOCK_CHARS:
                continue
            page_texts.append(text)

        # 合并该页的短 block
        page_texts = _merge_short_blocks(page_texts, min_length=80)

        if page_texts:
            page_content = "\n\n".join(page_texts)
            pages.append(f"[PAGE {page_num + 1}]\n{page_content}")

    page_count = len(doc)
    doc.close()

    full_text = "\n\n".join(pages)
    logger.info(f"[PDF] 提取完成: {Path(pdf_path).name}, {page_count} 页")
    return full_text


def split_into_sections(text: str) -> List[Section]:
    """
    按章节标题切分文本

    识别模式（按优先级）：
    1. 编号标题: "1. Introduction", "2.1 Diagnosis"
    2. 全大写标题: "DIAGNOSIS", "TREATMENT"
    3. 首字母大写独立行（>3字符, <60字符）

    Args:
        text: 全文文本

    Returns:
        章节列表
    """
    patterns = [
        re.compile(r'\n(\d+\.?\d*\.?\s+[A-Z][A-Za-z\s,/\-\(\)]{3,60})\n'),
        re.compile(r'\n([A-Z][A-Z\s/\-]{3,50})\n'),
    ]

    best_matches = []
    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if len(matches) > len(best_matches):
            best_matches = matches

    if len(best_matches) < 2:
        return []

    def get_page_number(pos: int) -> int:
        page_markers = list(re.finditer(r'\[PAGE (\d+)\]', text[:pos]))
        if page_markers:
            return int(page_markers[-1].group(1))
        return 1

    sections = []
    for i, match in enumerate(best_matches):
        start = match.start()
        end = best_matches[i + 1].start() if i + 1 < len(best_matches) else len(text)
        content = text[start:end].strip()

        content = re.sub(r'\[PAGE \d+\]', '', content).strip()

        title = match.group(1).strip()
        if title.upper() in SKIP_SECTION_TITLES:
            continue

        if content and len(content) >= MIN_CHUNK_CHARS:
            sections.append(Section(
                title=title,
                content=content,
                page_start=get_page_number(start)
            ))

    logger.info(f"[PDF] 识别到 {len(sections)} 个章节")
    return sections


def _prefix_with_section_title(chunk_text: str, section_title: str) -> str:
    """给 chunk 加上章节标题前缀，帮助 embedding 理解上下文"""
    if section_title and section_title != "unknown":
        return f"[Section: {section_title}]\n{chunk_text}"
    return chunk_text


def pdf_to_documents(
    pdf_path: str,
    guideline_name: str,
    max_chunk_size: int = 1500,
    chunk_overlap: int = 300,
    cache_dir: Optional[str] = None,
) -> List[Document]:
    """
    从 PDF 提取文档块（完整流程）

    Args:
        pdf_path: PDF 文件路径
        guideline_name: 指南名称（用于 metadata）
        max_chunk_size: 最大块大小
        chunk_overlap: 块重叠大小
        cache_dir: 缓存目录（可选，缓存提取的文本）

    Returns:
        Document 列表
    """
    pdf_path = str(pdf_path)

    # 检查缓存 — 注意：改了提取逻辑后需要清缓存或 force_rebuild
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        # 用 "v2_" 前缀区分新旧缓存
        file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
        cache_file = cache_path / f"v2_{file_hash}.txt"

        if cache_file.exists():
            logger.info(f"[PDF] 从缓存加载: {Path(pdf_path).name}")
            text = cache_file.read_text(encoding="utf-8")
        else:
            text = extract_text_from_pdf(pdf_path)
            cache_file.write_text(text, encoding="utf-8")
            logger.info(f"[PDF] 已缓存: {cache_file.name}")
    else:
        text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        logger.warning(f"[PDF] 文件为空: {pdf_path}")
        return []

    sections = split_into_sections(text)

    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    if sections:
        for section in sections:
            if len(section.content) > max_chunk_size:
                chunks = splitter.split_text(section.content)
            else:
                chunks = [section.content]

            for chunk in chunks:
                # 过滤过短 chunk
                if len(chunk.strip()) < MIN_CHUNK_CHARS:
                    continue

                page_content = _prefix_with_section_title(chunk, section.title)

                documents.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": Path(pdf_path).name,
                        "guideline": guideline_name,
                        "section": section.title,
                        "page": section.page_start,
                        "type": "guideline",
                    }
                ))
    else:
        logger.info(f"[PDF] 未识别到章节，使用递归切分")
        clean_text = re.sub(r'\[PAGE \d+\]', '', text)
        chunks = splitter.split_text(clean_text)

        for chunk in chunks:
            if len(chunk.strip()) < MIN_CHUNK_CHARS:
                continue

            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": Path(pdf_path).name,
                    "guideline": guideline_name,
                    "section": "unknown",
                    "page": 0,
                    "type": "guideline",
                }
            ))

    filtered_count = len(documents)
    logger.info(
        f"[PDF] {Path(pdf_path).name}: "
        f"{len(sections)} 个章节, {filtered_count} 个有效文档块"
    )
    return documents
