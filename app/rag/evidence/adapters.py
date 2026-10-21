"""Evidence Adapters — 将 4 个 RAG 的异构输出统一转为 EvidenceItem

每个适配器从 Document.metadata 提取来源信息，映射到统一的 evidence_level。

Metadata 结构参考：
- Guidelines: source(文件名), guideline(指南名), section(章节), page(页码), type="guideline"
- EyeWiki: source(URL), disease(疾病名)
- MedlinePlus: source(URL), topic(主题名), section(章节), type="medlineplus", attribution
- PubMed: pmid, title, journal, year, doi, source(URL), type="pubmed", section(label), attribution
"""

from typing import List

from langchain_core.documents import Document

from app.rag.evidence.schema import EvidenceItem


def guidelines_to_evidence(docs: List[Document]) -> List[EvidenceItem]:
    """Guidelines RAG 结果 → EvidenceItem（evidence_level=guideline）"""
    items = []
    for doc in docs:
        meta = doc.metadata
        guideline_name = meta.get("guideline", meta.get("source", "Clinical Guideline"))
        page = meta.get("page", "")

        source_name = guideline_name
        if page:
            source_name += f", p.{page}"

        items.append(EvidenceItem(
            claim=doc.page_content[:500],
            source=meta.get("source", ""),
            source_name=source_name,
            evidence_level="guideline",
        ))
    return items


def eyewiki_to_evidence(docs: List[Document]) -> List[EvidenceItem]:
    """EyeWiki RAG 结果 → EvidenceItem（evidence_level=encyclopedia）"""
    items = []
    for doc in docs:
        meta = doc.metadata
        disease = meta.get("disease", "")

        items.append(EvidenceItem(
            claim=doc.page_content[:500],
            source=meta.get("source", ""),
            source_name=f"EyeWiki: {disease}" if disease else "EyeWiki",
            evidence_level="encyclopedia",
        ))
    return items


def medlineplus_to_evidence(docs: List[Document]) -> List[EvidenceItem]:
    """MedlinePlus RAG 结果 → EvidenceItem（evidence_level=encyclopedia）"""
    items = []
    for doc in docs:
        meta = doc.metadata
        topic = meta.get("topic", "")

        items.append(EvidenceItem(
            claim=doc.page_content[:500],
            source=meta.get("source", ""),
            source_name=f"MedlinePlus: {topic}" if topic else "MedlinePlus",
            evidence_level="encyclopedia",
        ))
    return items


def pubmed_to_evidence(docs: List[Document]) -> List[EvidenceItem]:
    """PubMed RAG 结果 → EvidenceItem（evidence_level=literature）"""
    items = []
    for doc in docs:
        meta = doc.metadata
        pmid = meta.get("pmid", "")
        journal = meta.get("journal", "")

        source_name = f"PMID:{pmid}" if pmid else "PubMed"
        if journal:
            source_name += f" ({journal})"

        items.append(EvidenceItem(
            claim=doc.page_content[:500],
            source=meta.get("source", ""),
            source_name=source_name,
            evidence_level="literature",
        ))
    return items
