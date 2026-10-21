"""MedlinePlus RAG 模块

提供 MedlinePlus 健康主题的探索、索引和检索功能：
- NLM Search API 搜索相关主题
- 健康主题页面爬取与缓存
- FAISS 向量检索

内容属于公共领域（US federal government），需标注来源。

使用示例：
    from app.rag.medlineplus import MedlinePlusExplorer, MedlinePlusRAG

    # 1. 探索主题
    explorer = MedlinePlusExplorer()
    topics = explorer.search_topics("glaucoma")

    # 2. 构建索引
    rag = MedlinePlusRAG()
    rag.build_index([t.name for t in topics])

    # 3. 检索
    results = rag.search_with_sources("glaucoma symptoms")

快速模式（从搜索直接建索引）：
    rag = MedlinePlusRAG()
    rag.build_index_from_search("eye diseases")
    results = rag.search_with_sources("cataract treatment")
"""

from app.rag.medlineplus.rag import MedlinePlusRAG
from app.rag.medlineplus.explorer import (
    MedlinePlusExplorer,
    TopicInfo,
    format_topics_for_agent,
)

__all__ = [
    "MedlinePlusRAG",
    "MedlinePlusExplorer",
    "TopicInfo",
    "format_topics_for_agent",
]
