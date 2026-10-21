"""RAG Module - 检索增强生成模块

支持多种知识源的 RAG 实现：

知识库 A - 临床指南：
- Guidelines：临床指南（ICO, AOA, ESCRS 等 PDF）

知识库 B - 眼科专业知识库：
- EyeWiki：眼科医学知识百科
- MedlinePlus：NIH/NLM 健康主题（公共领域）

知识库 C - 医学文献库：
- PubMed：MEDLINE 医学文献摘要（Review/Meta-Analysis 为主）

EyeWiki 工作流：
    from app.rag import EyeWikiExplorer, EyeWikiRAG

    explorer = EyeWikiExplorer()
    result = explorer.explore("Glaucoma")
    rag = EyeWikiRAG()
    rag.build_index_from_articles([a.name for a in result["articles"]])
    docs = rag.search("What are the symptoms?")

Guidelines 工作流：
    from app.rag import GuidelinesRAG

    # 确保 data/rag/guidelines/pdf/ 下有 PDF 文件
    rag = GuidelinesRAG()
    rag.build_index()
    results = rag.search_with_sources("glaucoma diagnosis criteria")

MedlinePlus 工作流：
    from app.rag import MedlinePlusExplorer, MedlinePlusRAG

    explorer = MedlinePlusExplorer()
    topics = explorer.search_topics("glaucoma")
    rag = MedlinePlusRAG()
    rag.build_index([t.name for t in topics])
    results = rag.search_with_sources("glaucoma symptoms")

PubMed 工作流：
    from app.rag import PubMedExplorer, PubMedRAG

    explorer = PubMedExplorer()
    articles = explorer.search_reviews("glaucoma diagnosis")
    rag = PubMedRAG()
    rag.build_index(articles)
    results = rag.search_with_sources("glaucoma diagnostic criteria")

Evidence Retriever（多级检索编排）：
    from app.rag import EvidenceRetriever

    retriever = EvidenceRetriever()
    package = await retriever.retrieve(
        query="glaucoma diagnosis",
        suspected_diagnosis="primary open-angle glaucoma",
    )
    print(package.evidence)              # 证据列表
    print(package.missing_information)    # 缺失信息
"""

# EyeWiki
from app.rag.eyewiki import (
    EyeWikiRAG,
    EyeWikiExplorer,
    ArticleInfo,
    format_articles_for_agent
)

# Guidelines
from app.rag.guidelines import GuidelinesRAG

# MedlinePlus
from app.rag.medlineplus import (
    MedlinePlusRAG,
    MedlinePlusExplorer,
    TopicInfo,
    format_topics_for_agent,
)

# PubMed
from app.rag.pubmed import (
    PubMedRAG,
    PubMedExplorer,
)
from app.rag.pubmed.explorer import (
    ArticleInfo as PubMedArticleInfo,
    format_articles_for_agent as format_pubmed_articles_for_agent,
)

# Evidence (编排层)
from app.rag.evidence import (
    EvidenceRetriever,
    EvidenceItem,
    EvidencePackage,
    format_evidence_for_clinical,
    format_evidence_for_critic,
)

__all__ = [
    # EyeWiki
    "EyeWikiRAG",
    "EyeWikiExplorer",
    "ArticleInfo",
    "format_articles_for_agent",
    # Guidelines
    "GuidelinesRAG",
    # MedlinePlus
    "MedlinePlusRAG",
    "MedlinePlusExplorer",
    "TopicInfo",
    "format_topics_for_agent",
    # PubMed
    "PubMedRAG",
    "PubMedExplorer",
    "PubMedArticleInfo",
    "format_pubmed_articles_for_agent",
    # Evidence
    "EvidenceRetriever",
    "EvidenceItem",
    "EvidencePackage",
    "format_evidence_for_clinical",
    "format_evidence_for_critic",
]
