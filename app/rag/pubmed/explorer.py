"""PubMed Explorer - 通过 NCBI E-utilities 搜索医学文献

PubMed RAG 的前置阶段，负责：
1. 调用 esearch API 搜索文献（支持 MeSH、文献类型、日期范围过滤）
2. 调用 efetch API 获取文献详细信息（标题、摘要、MeSH、关键词等）
3. 提供结构化数据给 Agent 进行精选

NCBI E-utilities 使用条款：
- 无 API key: 3 requests/second
- PubMed 摘要和元数据属于公共领域
"""

import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import requests

from app.utils.logger import get_logger

logger = get_logger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

ATTRIBUTION = "Data from PubMed®, a service of the National Library of Medicine, National Institutes of Health."


@dataclass
class ArticleInfo:
    """PubMed 文献信息"""
    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    pub_types: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    abstract_sections: Dict[str, str] = field(default_factory=dict)

    @property
    def is_structured_abstract(self) -> bool:
        return len(self.abstract_sections) > 1

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class PubMedExplorer:
    """PubMed 探索器 - 通过 NCBI E-utilities 搜索和获取文献"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: NCBI API key（可选，提高速率限制到 10 req/s）
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EyeMultiAgent/1.0 (Research)"
        })
        self._last_request_time = 0.0
        self._min_interval = 0.35 if api_key else 1.0

    def search(
        self,
        query: str,
        max_results: int = 20,
        pub_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[str]:
        """
        搜索 PubMed，返回 PMID 列表

        Args:
            query: 搜索词（支持 PubMed 高级语法）
            max_results: 最大返回数
            pub_types: 文献类型过滤（如 ["Review", "Meta-Analysis", "Systematic Review"]）
            min_year: 最早年份
            max_year: 最晚年份

        Returns:
            PMID 列表
        """
        term = self._build_query(query, pub_types, min_year, max_year)
        logger.info(f"[PubMed] esearch: {term}")

        self._rate_limit()
        try:
            r = self.session.get(
                f"{EUTILS_BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmax": max_results,
                    "retmode": "json",
                    "sort": "relevance",
                    **({"api_key": self.api_key} if self.api_key else {}),
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            result = data.get("esearchresult", {})
            ids = result.get("idlist", [])
            total = result.get("count", "0")
            logger.info(f"[PubMed] 找到 {total} 篇，返回 {len(ids)} 个 PMID")
            return ids
        except Exception as e:
            logger.error(f"[PubMed] esearch 失败: {e}")
            return []

    def fetch_articles(self, pmids: List[str]) -> List[ArticleInfo]:
        """
        根据 PMID 列表获取文献详细信息

        Args:
            pmids: PMID 列表

        Returns:
            ArticleInfo 列表
        """
        if not pmids:
            return []

        logger.info(f"[PubMed] efetch: {len(pmids)} 篇文献")

        articles = []
        batch_size = 50
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_articles = self._fetch_batch(batch)
            articles.extend(batch_articles)
            if i + batch_size < len(pmids):
                time.sleep(0.5)

        logger.info(f"[PubMed] 成功获取 {len(articles)}/{len(pmids)} 篇")
        return articles

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 20,
        pub_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[ArticleInfo]:
        """
        搜索并获取文献详情（组合 esearch + efetch）

        Args:
            query: 搜索词
            max_results: 最大返回数
            pub_types: 文献类型过滤
            min_year: 最早年份
            max_year: 最晚年份

        Returns:
            ArticleInfo 列表
        """
        pmids = self.search(query, max_results, pub_types, min_year, max_year)
        if not pmids:
            return []
        return self.fetch_articles(pmids)

    def search_reviews(
        self,
        query: str,
        max_results: int = 20,
        min_year: int = 2021,
    ) -> List[ArticleInfo]:
        """
        搜索 Review/Meta-Analysis/Systematic Review 文献（快捷方法）

        Args:
            query: 搜索词（如 "glaucoma diagnosis"）
            max_results: 最大返回数
            min_year: 最早年份

        Returns:
            ArticleInfo 列表
        """
        return self.search_and_fetch(
            query=query,
            max_results=max_results,
            pub_types=["Review", "Meta-Analysis", "Systematic Review"],
            min_year=min_year,
        )

    def _build_query(
        self,
        query: str,
        pub_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> str:
        parts = [query]

        if pub_types:
            type_filter = " OR ".join(f"{pt}[pt]" for pt in pub_types)
            parts.append(f"({type_filter})")

        if min_year or max_year:
            y_min = min_year or 1900
            y_max = max_year or 2099
            parts.append(f"{y_min}:{y_max}[dp]")

        return " AND ".join(parts)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _fetch_batch(self, pmids: List[str]) -> List[ArticleInfo]:
        self._rate_limit()
        try:
            r = self.session.get(
                f"{EUTILS_BASE}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "rettype": "abstract",
                    **({"api_key": self.api_key} if self.api_key else {}),
                },
                timeout=30,
            )
            r.raise_for_status()
            return self._parse_articles_xml(r.text)
        except Exception as e:
            logger.error(f"[PubMed] efetch 失败: {e}")
            return []

    def _parse_articles_xml(self, xml_text: str) -> List[ArticleInfo]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"[PubMed] XML 解析失败: {e}")
            return []

        articles = []
        for article_el in root.findall(".//PubmedArticle"):
            info = self._parse_single_article(article_el)
            if info:
                articles.append(info)
        return articles

    def _parse_single_article(self, article_el: ET.Element) -> Optional[ArticleInfo]:
        pmid = article_el.findtext(".//PMID", "")
        if not pmid:
            return None

        title = article_el.findtext(".//ArticleTitle", "")
        journal = article_el.findtext(".//Journal/Title", "")
        year = article_el.findtext(".//PubDate/Year", "")
        if not year:
            medline_date = article_el.findtext(".//PubDate/MedlineDate", "")
            if medline_date:
                year = medline_date[:4]

        abstract_el = article_el.find(".//Abstract")
        abstract = ""
        abstract_sections = {}
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext()).strip()
                if label:
                    abstract_sections[label] = text
                if text:
                    if abstract:
                        abstract += "\n\n"
                    if label:
                        abstract += f"{label}: {text}"
                    else:
                        abstract += text

        pub_types = [
            pt.text for pt in article_el.findall(".//PublicationType")
            if pt.text
        ]

        mesh_terms = [
            desc.text for desc in article_el.findall(".//MeshHeading/DescriptorName")
            if desc.text
        ]

        keywords = [
            kw.text for kw in article_el.findall(".//Keyword")
            if kw.text
        ]

        doi_el = article_el.find(".//ArticleId[@IdType='doi']")
        doi = doi_el.text if doi_el is not None else ""

        return ArticleInfo(
            pmid=pmid,
            title=title,
            abstract=abstract,
            journal=journal,
            year=year,
            pub_types=pub_types,
            mesh_terms=mesh_terms,
            keywords=keywords,
            doi=doi,
            abstract_sections=abstract_sections,
        )


def format_articles_for_agent(articles: List[ArticleInfo], max_preview: int = 200) -> str:
    """
    将文献列表格式化为适合 Agent 阅读的文本

    Args:
        articles: 文献列表
        max_preview: 摘要预览最大字符数

    Returns:
        格式化文本
    """
    if not articles:
        return "未找到相关文献。"

    lines = [f"PubMed 检索结果（{len(articles)} 篇）：", ""]

    for i, art in enumerate(articles, 1):
        lines.append(f"{i}. [{art.year}] {art.title}")
        lines.append(f"   PMID: {art.pmid} | Journal: {art.journal}")
        if art.pub_types:
            lines.append(f"   Type: {', '.join(art.pub_types[:3])}")
        if art.mesh_terms:
            lines.append(f"   MeSH: {', '.join(art.mesh_terms[:5])}")
        if art.abstract:
            preview = art.abstract[:max_preview].replace("\n", " ")
            lines.append(f"   摘要: {preview}...")
        lines.append("")

    return "\n".join(lines)
