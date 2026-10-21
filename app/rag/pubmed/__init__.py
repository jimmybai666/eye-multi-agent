"""PubMed RAG 模块

提供 PubMed/MEDLINE 医学文献的搜索、索引和检索功能：
- NCBI E-utilities 搜索文献（支持文献类型、日期范围过滤）
- 文献摘要获取与结构化切分
- FAISS 向量检索

PubMed 摘要和元数据属于公共领域（NCBI open data policy）。

使用示例：
    from app.rag.pubmed import PubMedExplorer, PubMedRAG

    # 1. 搜索文献
    explorer = PubMedExplorer()
    articles = explorer.search_reviews("glaucoma diagnosis", max_results=20)

    # 2. 构建索引
    rag = PubMedRAG()
    rag.build_index(articles)

    # 3. 检索
    results = rag.search_with_sources("glaucoma diagnostic criteria")

快速模式（从搜索直接建索引）：
    rag = PubMedRAG()
    rag.build_index_from_reviews("glaucoma diagnosis")
    results = rag.search_with_sources("IOP measurement methods")
"""

from app.rag.pubmed.rag import PubMedRAG
from app.rag.pubmed.explorer import (
    PubMedExplorer,
    ArticleInfo,
    format_articles_for_agent,
    ATTRIBUTION,
)

__all__ = [
    "PubMedRAG",
    "PubMedExplorer",
    "ArticleInfo",
    "format_articles_for_agent",
    "ATTRIBUTION",
]
