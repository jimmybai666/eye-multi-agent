"""Evidence 模块 — 多级证据检索编排层

将 4 个 RAG 知识库（Guidelines、EyeWiki、MedlinePlus、PubMed）的检索结果
统一为结构化证据包，供 Clinical Agent 和 Critic Agent 使用。

使用示例：
    from app.rag.evidence import EvidenceRetriever

    retriever = EvidenceRetriever()
    package = await retriever.retrieve(
        query="glaucoma diagnosis",
        suspected_diagnosis="primary open-angle glaucoma",
    )

    # 结构化输出
    print(package.evidence)
    print(package.missing_information)
"""

from app.rag.evidence.schema import (
    EvidenceItem,
    EvidencePackage,
    format_evidence_for_clinical,
    format_evidence_for_critic,
)
from app.rag.evidence.retriever import EvidenceRetriever

__all__ = [
    "EvidenceRetriever",
    "EvidenceItem",
    "EvidencePackage",
    "format_evidence_for_clinical",
    "format_evidence_for_critic",
]
