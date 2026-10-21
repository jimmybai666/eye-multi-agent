"""Evidence Schema — 结构化证据包数据模型

所有 RAG 检索结果统一转为此格式，供 Clinical Agent 和 Critic Agent 使用。

证据等级（3 级）：
    guideline    — GuidelinesRAG — 临床指南，最高优先级
    encyclopedia — EyeWiki + MedlinePlus — 专业知识参考
    literature   — PubMed — 文献证据，补充用
"""

from typing import Literal
from pydantic import BaseModel, Field


EVIDENCE_LEVELS = {
    "guideline": 1,
    "encyclopedia": 2,
    "literature": 3,
}


class EvidenceItem(BaseModel):
    """单条证据"""

    claim: str = Field(description="证据声明")
    source: str = Field(description="出处 URL 或文档引用")
    source_name: str = Field(description="来源简称（如 ICO Guidelines / EyeWiki）")
    evidence_level: Literal[
        "guideline",
        "encyclopedia",
        "literature",
    ] = Field(description="证据等级：guideline(临床指南) > encyclopedia(专业知识) > literature(文献)")

    @property
    def level_rank(self) -> int:
        return EVIDENCE_LEVELS.get(self.evidence_level, 99)


class EvidencePackage(BaseModel):
    """结构化证据包 — Evidence Retriever 的输出"""

    query: str = Field(description="检索主题")
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    retrieval_status: Literal[
        "completed",
        "partial",
        "insufficient_evidence",
        "error",
    ] = Field(default="completed")
    retrieval_log: list[str] = Field(default_factory=list)

    @property
    def total_evidence_count(self) -> int:
        return len(self.evidence)

    @property
    def has_guideline_evidence(self) -> bool:
        return any(e.evidence_level == "guideline" for e in self.evidence)

    def get_sorted_evidence(self) -> list[EvidenceItem]:
        """按证据等级排序返回所有证据（guideline 在前）"""
        return sorted(self.evidence, key=lambda e: e.level_rank)


def format_evidence_for_clinical(package: EvidencePackage) -> str:
    """将证据包格式化为 Clinical Agent 可读的文本"""
    lines = []
    lines.append(f"=== Evidence Package: {package.query} ===")
    lines.append(f"Status: {package.retrieval_status}")
    lines.append("")

    if package.evidence:
        lines.append("## Evidence")
        for i, e in enumerate(package.get_sorted_evidence(), 1):
            lines.append(f"{i}. [{e.evidence_level.upper()}] {e.claim}")
            lines.append(f"   Source: {e.source_name}")
        lines.append("")

    if package.missing_information:
        lines.append("## Missing Information")
        for item in package.missing_information:
            lines.append(f"- {item}")
        lines.append("")

    lines.append(f"Total: {package.total_evidence_count} evidence items from {len(package.sources)} sources")
    return "\n".join(lines)


def format_evidence_for_critic(package: EvidencePackage) -> str:
    """将证据包格式化为 Critic Agent 可读的文本"""
    lines = []
    lines.append("=== Evidence Review ===")

    if package.has_guideline_evidence:
        lines.append("Guideline-level evidence is available.")
    else:
        lines.append("WARNING: No guideline-level evidence found. Diagnosis relies on lower-level sources.")

    lines.append("")

    if package.evidence:
        lines.append(f"Evidence: {len(package.evidence)} items")
        for e in package.get_sorted_evidence()[:5]:
            lines.append(f"  - [{e.evidence_level}] {e.claim} ({e.source_name})")

    if package.missing_information:
        lines.append(f"Missing info: {len(package.missing_information)} items")
        for item in package.missing_information:
            lines.append(f"  - {item}")

    lines.append(f"\nRetrieval: {package.retrieval_status} | {len(package.retrieval_log)} steps")
    return "\n".join(lines)
