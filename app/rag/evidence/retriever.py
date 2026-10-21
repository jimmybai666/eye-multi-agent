"""Evidence Retriever — 多级证据检索编排器

按证据层级依次检索 3层 4 个知识库，组装结构化证据包：
1. 临床指南（Guidelines RAG） → evidence_level = guideline
2. 专业知识（EyeWiki + MedlinePlus RAG） → evidence_level = encyclopedia
3. 医学文献（PubMed RAG） → evidence_level = literature

检索步骤是确定性代码逻辑，不依赖 LLM 做编排决策。
"""

import asyncio

from app.rag.evidence.schema import EvidenceItem, EvidencePackage
from app.rag.evidence.adapters import (
    guidelines_to_evidence,
    eyewiki_to_evidence,
    medlineplus_to_evidence,
    pubmed_to_evidence,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceRetriever:
    """多级证据检索编排器"""

    def __init__(self, embedding_model: str = "qwen3.7-embedding"):
        self.embedding_model = embedding_model
        self._guidelines_rag = None
        self._eyewiki_rag = None
        self._medlineplus_rag = None
        self._pubmed_rag = None

    @property
    def guidelines_rag(self):
        if self._guidelines_rag is None:
            from app.rag.guidelines.rag import GuidelinesRAG
            self._guidelines_rag = GuidelinesRAG(embedding_model=self.embedding_model)
        return self._guidelines_rag

    @property
    def eyewiki_rag(self):
        if self._eyewiki_rag is None:
            from app.rag.eyewiki.rag import EyeWikiRAG
            self._eyewiki_rag = EyeWikiRAG(embedding_model=self.embedding_model)
        return self._eyewiki_rag

    @property
    def medlineplus_rag(self):
        if self._medlineplus_rag is None:
            from app.rag.medlineplus.rag import MedlinePlusRAG
            self._medlineplus_rag = MedlinePlusRAG(embedding_model=self.embedding_model)
        return self._medlineplus_rag

    @property
    def pubmed_rag(self):
        if self._pubmed_rag is None:
            from app.rag.pubmed.rag import PubMedRAG
            self._pubmed_rag = PubMedRAG(embedding_model=self.embedding_model)
        return self._pubmed_rag

    async def retrieve(
        self,
        query: str,
        clinical_context: str = "",
        suspected_diagnosis: str = "",
        need_literature: bool = False,
        k_per_source: int = 3,
    ) -> EvidencePackage:
        """
        多级检索主入口

        Args:
            query: 检索主题（如 "diabetic retinopathy diagnosis"）
            clinical_context: 临床上下文（如 vision findings）
            suspected_diagnosis: 疑似诊断（用于拼接搜索词和证据分类）
            need_literature: 是否强制查文献
            k_per_source: 每个来源返回的结果数

        Returns:
            EvidencePackage
        """
        search_query = self._build_search_query(query, clinical_context, suspected_diagnosis)
        log = []
        all_evidence: list[EvidenceItem] = []
        sources: list[dict] = []


        # ── Step 1: 临床指南 ─────────────────────────────────
        log.append(f"Step 1: Searching guidelines for '{search_query}'")
        guideline_evidence = await asyncio.to_thread(self._search_guidelines, search_query, k_per_source)
        if guideline_evidence:
            all_evidence.extend(guideline_evidence)
            sources.append({"type": "guidelines", "count": len(guideline_evidence)})
            log.append(f"  → Found {len(guideline_evidence)} guideline items")
        else:
            log.append("  → No guideline index available or no results")


        # ── Step 2: 专业知识（EyeWiki + MedlinePlus） ──────────
        log.append(f"Step 2: Searching encyclopedias for '{search_query}'")

        eyewiki_evidence, medlineplus_evidence = await asyncio.gather(
            asyncio.to_thread(self._search_eyewiki, search_query, k_per_source),
            asyncio.to_thread(self._search_medlineplus, search_query, k_per_source),
        )

        if eyewiki_evidence:
            all_evidence.extend(eyewiki_evidence)
            sources.append({"type": "eyewiki", "count": len(eyewiki_evidence)})
            log.append(f"  → EyeWiki: {len(eyewiki_evidence)} items")
        else:
            log.append("  → EyeWiki: no index or no results")

        if medlineplus_evidence:
            all_evidence.extend(medlineplus_evidence)
            sources.append({"type": "medlineplus", "count": len(medlineplus_evidence)})
            log.append(f"  → MedlinePlus: {len(medlineplus_evidence)} items")
        else:
            log.append("  → MedlinePlus: no index or no results")


        # ── Step 3: 判断是否需要文献 有指南结果 → 跳过，无则查 ──────────────────────────
        should_search_pubmed = self._should_search_pubmed(
            guideline_evidence=guideline_evidence,
            need_literature=need_literature,
        )

        if should_search_pubmed:
            log.append(f"Step 3: Searching PubMed for '{search_query}'")
            pubmed_evidence = await asyncio.to_thread(self._search_pubmed, search_query, k_per_source)
            if pubmed_evidence:
                all_evidence.extend(pubmed_evidence)
                sources.append({"type": "pubmed", "count": len(pubmed_evidence)})
                log.append(f"  → PubMed: {len(pubmed_evidence)} items")
            else:
                log.append("  → PubMed: no index or no results")
        else:
            log.append("Step 3: PubMed search skipped (guideline evidence sufficient)")



        # ── Step 4: 组装证据包 ────────────────────────────────
        log.append(f"Step 4: Assembling evidence package ({len(all_evidence)} total items)")

        all_evidence.sort(key=lambda e: e.level_rank)
        missing = self._identify_missing(all_evidence)

        status = "completed" if all_evidence else "insufficient_evidence"

        package = EvidencePackage(
            query=query,
            evidence=all_evidence,
            missing_information=missing,
            sources=sources,
            retrieval_status=status,
            retrieval_log=log,
        )

        logger.info(
            f"[Evidence] Retrieved {package.total_evidence_count} items "
            f"from {len(sources)} sources"
        )
        return package

    def _build_search_query(
        self, query: str, clinical_context: str, suspected_diagnosis: str
    ) -> str:
        parts = []
        if suspected_diagnosis:
            parts.append(suspected_diagnosis)
        parts.append(query)
        if clinical_context:
            keywords = clinical_context[:200]
            parts.append(keywords)
        return " ".join(parts)

    def _search_guidelines(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            docs = self.guidelines_rag.search(query, k=k)
            return guidelines_to_evidence(docs)
        except (ValueError, FileNotFoundError):
            logger.debug("[Evidence] Guidelines index not available")
            return []
        except Exception as e:
            logger.warning(f"[Evidence] Guidelines search failed: {e}")
            return []

    def _search_eyewiki(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            docs = self.eyewiki_rag.search(query, k=k)
            return eyewiki_to_evidence(docs)
        except (ValueError, FileNotFoundError):
            logger.debug("[Evidence] EyeWiki index not available")
            return []
        except Exception as e:
            logger.warning(f"[Evidence] EyeWiki search failed: {e}")
            return []

    def _search_medlineplus(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            docs = self.medlineplus_rag.search(query, k=k)
            return medlineplus_to_evidence(docs)
        except (ValueError, FileNotFoundError):
            logger.debug("[Evidence] MedlinePlus index not available")
            return []
        except Exception as e:
            logger.warning(f"[Evidence] MedlinePlus search failed: {e}")
            return []

    def _search_pubmed(self, query: str, k: int) -> list[EvidenceItem]:
        try:
            docs = self.pubmed_rag.search(query, k=k)
            return pubmed_to_evidence(docs)
        except (ValueError, FileNotFoundError):
            logger.debug("[Evidence] PubMed index not available")
            return []
        except Exception as e:
            logger.warning(f"[Evidence] PubMed search failed: {e}")
            return []

    def _should_search_pubmed(
        self,
        guideline_evidence: list[EvidenceItem],
        need_literature: bool,
    ) -> bool:
        if need_literature:
            return True
        if not guideline_evidence:
            return True
        return False

    def _identify_missing(
        self,
        evidence: list[EvidenceItem],
    ) -> list[str]:
        """识别缺失的关键信息"""
        missing = []

        has_guideline = any(e.evidence_level == "guideline" for e in evidence)

        if not has_guideline:
            missing.append(
                "No clinical guideline evidence found. "
                "Consider checking if guideline index has been built."
            )
        if not evidence:
            missing.append(
                "No evidence retrieved from any knowledge base. "
                "Ensure at least one RAG index has been built."
            )

        return missing
