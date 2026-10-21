"""PubMed RAG - 医学文献检索增强生成

功能：
1. 通过 NCBI E-utilities 搜索 PubMed 文献
2. 获取文献摘要和元数据（标题、MeSH、关键词等）
3. 按摘要结构智能切分（结构化摘要按 label，非结构化用递归切分）
4. 构建 FAISS 向量索引
5. 语义检索

数据来源：
- PubMed / MEDLINE（NCBI/NLM）

版权说明：
PubMed 摘要和元数据属于公共领域（NCBI open data policy），可自由使用。
需标注来源：Data from PubMed, a service of the National Library of Medicine, NIH.
"""

import json
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models import create_embeddings
from app.rag.pubmed.explorer import PubMedExplorer, ArticleInfo, ATTRIBUTION
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PubMedRAG:
    """PubMed 医学文献 RAG"""

    def __init__(
        self,
        data_dir: str = "data/rag/pubmed",
        embedding_model: str = "qwen3.7-embedding",
        api_key: Optional[str] = None,
    ):
        """
        Args:
            data_dir: 数据目录（缓存和索引）
            embedding_model: Embedding 模型
            api_key: NCBI API key（可选）
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.index_dir = self.data_dir / "index"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[PubMed] 使用 Embedding 模型: {embedding_model}")
        self.embeddings = create_embeddings(model_key=embedding_model)

        self.explorer = PubMedExplorer(api_key=api_key)
        self.vector_store = None

    def build_index(
        self,
        articles: List[ArticleInfo],
        force_rebuild: bool = False,
    ):
        """
        从文献列表构建索引

        Args:
            articles: ArticleInfo 列表（从 explorer 获取）
            force_rebuild: 是否强制重建

        Example:
            explorer = PubMedExplorer()
            articles = explorer.search_reviews("glaucoma diagnosis")
            rag = PubMedRAG()
            rag.build_index(articles)
        """
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force_rebuild:
            logger.info("[PubMed] 索引已存在，跳过构建")
            self._load_index()
            return

        logger.info(f"[PubMed] 开始构建索引: {len(articles)} 篇文献")

        all_documents = []
        for article in articles:
            docs = self._article_to_documents(article)
            if docs:
                all_documents.extend(docs)

        if not all_documents:
            raise ValueError("[PubMed] 没有成功处理任何文献")

        logger.info(f"[PubMed] 共 {len(all_documents)} 个文档块，构建索引...")

        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
        self.vector_store.save_local(str(self.index_dir))

        meta = {
            "article_count": len(articles),
            "document_count": len(all_documents),
            "pmids": [a.pmid for a in articles],
            "attribution": ATTRIBUTION,
        }
        meta_file = self.index_dir / "meta.json"
        meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._save_articles_cache(articles)

        logger.info(f"[PubMed] 索引已保存: {index_file}")

    def build_index_from_search(
        self,
        query: str,
        max_results: int = 20,
        pub_types: Optional[List[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        force_rebuild: bool = False,
    ):
        """
        从搜索构建索引（先搜索再建索引）

        Args:
            query: 搜索词
            max_results: 最大文献数
            pub_types: 文献类型过滤
            min_year: 最早年份
            max_year: 最晚年份
            force_rebuild: 是否强制重建

        Example:
            rag = PubMedRAG()
            rag.build_index_from_search(
                "glaucoma diagnosis treatment",
                pub_types=["Review", "Meta-Analysis"],
                min_year=2021
            )
        """
        articles = self.explorer.search_and_fetch(
            query=query,
            max_results=max_results,
            pub_types=pub_types,
            min_year=min_year,
            max_year=max_year,
        )

        if not articles:
            raise ValueError(f"[PubMed] 搜索 '{query}' 未找到文献")

        logger.info(
            f"[PubMed] 从搜索构建索引: {len(articles)} 篇文献"
        )
        self.build_index(articles, force_rebuild=force_rebuild)

    def build_index_from_reviews(
        self,
        query: str,
        max_results: int = 20,
        min_year: int = 2021,
        force_rebuild: bool = False,
    ):
        """
        从 Review/Meta-Analysis 搜索构建索引（快捷方法）

        Args:
            query: 搜索词（如 "glaucoma diagnosis"）
            max_results: 最大文献数
            min_year: 最早年份
            force_rebuild: 是否强制重建
        """
        articles = self.explorer.search_reviews(
            query=query,
            max_results=max_results,
            min_year=min_year,
        )

        if not articles:
            raise ValueError(f"[PubMed] Review 搜索 '{query}' 未找到文献")

        self.build_index(articles, force_rebuild=force_rebuild)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询（如 "glaucoma diagnosis methods"）
            k: 返回结果数量

        Returns:
            Document 列表
        """
        if self.vector_store is None:
            self._load_index()

        logger.info(f"[PubMed] 检索: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        logger.info(f"[PubMed] 找到 {len(docs)} 个相关文档")
        return docs

    def search_with_sources(self, query: str, k: int = 3) -> dict:
        """
        检索并返回详细来源信息

        Returns:
            {
                "query": str,
                "total": int,
                "results": [
                    {
                        "content": str,
                        "pmid": str,
                        "title": str,
                        "journal": str,
                        "year": str,
                        "section": str,
                        "source": str (PubMed URL),
                        "attribution": str,
                        "preview": str,
                    },
                    ...
                ]
            }
        """
        docs = self.search(query, k)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "pmid": doc.metadata.get("pmid", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "journal": doc.metadata.get("journal", "N/A"),
                "year": doc.metadata.get("year", "N/A"),
                "section": doc.metadata.get("section", "full_abstract"),
                "doi": doc.metadata.get("doi", ""),
                "source": doc.metadata.get("source", "N/A"),
                "attribution": doc.metadata.get("attribution", ATTRIBUTION),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })

        return {
            "query": query,
            "total": len(results),
            "results": results,
        }

    def list_sources(self) -> dict:
        meta_file = self.index_dir / "meta.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))
        return {"article_count": 0, "document_count": 0, "pmids": []}

    # ── 内部方法 ──────────────────────────────────────────────

    def _load_index(self):
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            raise ValueError("[PubMed] 索引不存在，请先调用 build_index()")

        logger.info("[PubMed] 加载索引...")
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("[PubMed] 索引加载成功")

    def _article_to_documents(self, article: ArticleInfo) -> List[Document]:
        """
        将单篇文献转为 Document 列表

        策略：
        - 结构化摘要（有 label）：每个 section 单独成一个 Document
        - 非结构化摘要：整体 ≤ 1000 字符直接用，否则递归切分
        - 标题 + MeSH + 关键词合成一个 metadata Document（辅助语义匹配）
        """
        if not article.abstract:
            logger.debug(f"[PubMed] PMID:{article.pmid} 无摘要，跳过")
            return []

        base_metadata = {
            "pmid": article.pmid,
            "title": article.title,
            "journal": article.journal,
            "year": article.year,
            "doi": article.doi,
            "source": article.url,
            "type": "pubmed",
            "attribution": ATTRIBUTION,
        }

        documents = []

        if article.is_structured_abstract:
            for label, text in article.abstract_sections.items():
                if not text or len(text) < 20:
                    continue
                meta = {**base_metadata, "section": label}
                if len(text) > 1000:
                    chunks = self._split_text(text)
                    for chunk in chunks:
                        documents.append(Document(page_content=chunk, metadata=meta.copy()))
                else:
                    documents.append(Document(page_content=f"{label}: {text}", metadata=meta))
        else:
            meta = {**base_metadata, "section": "abstract"}
            if len(article.abstract) > 1000:
                chunks = self._split_text(article.abstract)
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata=meta.copy()))
            else:
                documents.append(Document(page_content=article.abstract, metadata=meta))

        meta_parts = [article.title]
        if article.mesh_terms:
            meta_parts.append("MeSH: " + ", ".join(article.mesh_terms))
        if article.keywords:
            meta_parts.append("Keywords: " + ", ".join(article.keywords))
        if article.pub_types:
            meta_parts.append("Type: " + ", ".join(article.pub_types))

        meta_text = "\n".join(meta_parts)
        documents.append(Document(
            page_content=meta_text,
            metadata={**base_metadata, "section": "metadata"},
        ))

        return documents

    def _split_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    def _save_articles_cache(self, articles: List[ArticleInfo]):
        cache_file = self.cache_dir / "articles.json"
        data = []
        for a in articles:
            data.append({
                "pmid": a.pmid,
                "title": a.title,
                "journal": a.journal,
                "year": a.year,
                "doi": a.doi,
                "pub_types": a.pub_types,
                "mesh_terms": a.mesh_terms,
                "keywords": a.keywords,
                "abstract_sections": a.abstract_sections,
                "has_abstract": bool(a.abstract),
            })
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[PubMed] 文献缓存已保存: {cache_file}")
