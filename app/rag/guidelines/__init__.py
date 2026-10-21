"""Guidelines RAG 模块

提供临床指南 PDF 的索引和检索功能：
- PDF 指南提取（ICO, AOA, ESCRS 等）
- FAISS 向量检索

使用示例：
    from app.rag.guidelines import GuidelinesRAG

    # 1. 确保 data/rag/guidelines/pdf/ 下有 PDF 文件
    # 2. 构建索引
    rag = GuidelinesRAG()
    rag.build_index()

    # 3. 检索
    results = rag.search_with_sources("glaucoma diagnosis criteria")
    for r in results["results"]:
        print(f"[{r['guideline']}] {r['section']}")
        print(r["preview"])
"""

from app.rag.guidelines.rag import GuidelinesRAG
from app.rag.guidelines.pdf_extractor import (
    extract_text_from_pdf,
    split_into_sections,
    pdf_to_documents,
)

__all__ = [
    "GuidelinesRAG",
    "extract_text_from_pdf",
    "split_into_sections",
    "pdf_to_documents",
]
