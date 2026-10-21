"""MedlinePlus RAG - 健康主题检索增强生成

功能：
1. 爬取 MedlinePlus 健康主题页面
2. 提取 topsum_section 公共领域摘要内容
3. 按章节智能切分（What is / Causes / Symptoms / Diagnosis / Treatment / Prevention）
4. 构建 FAISS 向量索引
5. 语义检索

数据来源：
- MedlinePlus Health Topics（NIH/NLM 公共领域内容）

版权说明：
MedlinePlus 健康主题摘要属于公共领域，可自由使用。
需标注来源：Courtesy of MedlinePlus from the National Library of Medicine
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models import create_embeddings
from app.rag.medlineplus.explorer import MedlinePlusExplorer, ATTRIBUTION
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MedlinePlusRAG:
    """MedlinePlus 健康主题 RAG"""

    BASE_URL = "https://medlineplus.gov"

    def __init__(
        self,
        data_dir: str = "data/rag/medlineplus",
        embedding_model: str = "qwen3.7-embedding",
    ):
        """
        初始化

        Args:
            data_dir: 数据目录（缓存和索引）
            embedding_model: Embedding 模型
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.index_dir = self.data_dir / "index"

        # 创建目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Embeddings
        logger.info(f"[MedlinePlus] 使用 Embedding 模型: {embedding_model}")
        self.embeddings = create_embeddings(model_key=embedding_model)

        # 向量数据库
        self.vector_store = None

    def build_index(
        self,
        topics: List[str],
        force_rebuild: bool = False,
    ):
        """
        从主题 slug 列表构建索引

        Args:
            topics: 主题 slug 列表（如 ["glaucoma", "cataract", "diabeticeyeproblems"]）
            force_rebuild: 是否强制重建

        Example:
            rag = MedlinePlusRAG()
            rag.build_index(["glaucoma", "cataract"])
        """
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force_rebuild:
            logger.info("[MedlinePlus] 索引已存在，跳过构建")
            self._load_index()
            return

        logger.info(f"[MedlinePlus] 开始构建索引: {len(topics)} 个主题")

        all_documents = []
        for topic in topics:
            docs = self._process_topic(topic)
            if docs:
                all_documents.extend(docs)
                logger.info(f"[MedlinePlus] ✓ {topic}: {len(docs)} 个块")
            else:
                logger.warning(f"[MedlinePlus] ✗ {topic}: 获取失败")

        if not all_documents:
            raise ValueError("[MedlinePlus] 没有成功获取任何文档")

        logger.info(f"[MedlinePlus] 共 {len(all_documents)} 个文档块，构建索引...")

        # 构建向量数据库
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)

        # 保存
        self.vector_store.save_local(str(self.index_dir))

        # 保存元信息
        meta = {
            "topics": topics,
            "total_documents": len(all_documents),
            "attribution": ATTRIBUTION,
        }
        meta_file = self.index_dir / "meta.json"
        meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(f"[MedlinePlus] 索引已保存: {index_file}")

    def build_index_from_search(
        self,
        keyword: str,
        max_topics: int = 10,
        force_rebuild: bool = False,
    ):
        """
        从搜索关键词构建索引（先搜索再建索引）

        Args:
            keyword: 搜索关键词（如 "eye diseases"）
            max_topics: 最多使用多少个主题
            force_rebuild: 是否强制重建

        Example:
            rag = MedlinePlusRAG()
            rag.build_index_from_search("glaucoma diagnosis treatment")
        """
        explorer = MedlinePlusExplorer()
        topics = explorer.search_topics(keyword, max_results=max_topics)

        if not topics:
            raise ValueError(f"[MedlinePlus] 搜索 '{keyword}' 未找到任何主题")

        slugs = [t.name for t in topics]
        logger.info(
            f"[MedlinePlus] 从搜索结果构建索引: {len(slugs)} 个主题 "
            f"({', '.join(slugs[:5])}{'...' if len(slugs) > 5 else ''})"
        )

        self.build_index(slugs, force_rebuild=force_rebuild)

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询（如 "glaucoma symptoms diagnosis"）
            k: 返回结果数量

        Returns:
            Document 列表
        """
        if self.vector_store is None:
            self._load_index()

        logger.info(f"[MedlinePlus] 检索: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        logger.info(f"[MedlinePlus] 找到 {len(docs)} 个相关文档")
        return docs

    def search_with_sources(self, query: str, k: int = 3) -> dict:
        """
        检索并返回详细来源信息

        Args:
            query: 查询
            k: 返回结果数量

        Returns:
            {
                "query": 查询内容,
                "total": 结果数,
                "results": [
                    {
                        "content": 文档内容,
                        "source": 来源 URL,
                        "topic": 主题名,
                        "section": 章节名,
                        "attribution": 来源声明,
                        "preview": 内容预览,
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
                "source": doc.metadata.get("source", "N/A"),
                "topic": doc.metadata.get("topic", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "attribution": doc.metadata.get("attribution", ATTRIBUTION),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })

        return {
            "query": query,
            "total": len(results),
            "results": results,
        }

    def list_sources(self) -> dict:
        """
        列出当前索引包含的来源

        Returns:
            {"topics": [...], "total_documents": int}
        """
        meta_file = self.index_dir / "meta.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))

        return {"topics": [], "total_documents": 0}

    # ── 内部方法 ──────────────────────────────────────────────

    def _load_index(self):
        """加载索引"""
        index_file = self.index_dir / "index.faiss"

        if not index_file.exists():
            raise ValueError(
                "[MedlinePlus] 索引不存在，请先调用 build_index()"
            )

        logger.info("[MedlinePlus] 加载索引...")
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("[MedlinePlus] 索引加载成功")

    def _process_topic(self, topic_slug: str) -> List[Document]:
        """
        处理单个主题：爬取 → 提取 → 切分

        Args:
            topic_slug: 主题 slug（如 "glaucoma"）

        Returns:
            Document 列表
        """
        url = f"{self.BASE_URL}/{topic_slug}.html"

        # 获取 HTML（带缓存）
        html = self._fetch_with_cache(url)
        if not html:
            logger.warning(f"[MedlinePlus] 无法获取: {url}")
            return []

        # 提取结构化内容
        topic_title, sections = self._extract_topic_content(html)
        if not topic_title:
            topic_title = topic_slug.replace("-", " ").title()

        if not sections:
            logger.warning(f"[MedlinePlus] 未提取到内容: {topic_slug}")
            return []

        # 切分并创建 Document
        documents = self._sections_to_documents(
            sections=sections,
            topic_title=topic_title,
            source_url=url,
        )

        logger.info(f"[MedlinePlus] {topic_slug}: {len(documents)} 个文档块")
        return documents

    def _fetch_with_cache(self, url: str) -> Optional[str]:
        """
        爬取网页（带 MD5 缓存）

        Args:
            url: 网页 URL

        Returns:
            HTML 文本（失败返回 None）
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.html"

        # 检查缓存
        if cache_file.exists():
            logger.info(f"[MedlinePlus] 从缓存加载: {url}")
            return cache_file.read_text(encoding="utf-8")

        # 爬取
        try:
            logger.info(f"[MedlinePlus] 爬取: {url}")
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"[MedlinePlus] 失败 {response.status_code}: {url}"
                )
                return None

            html = response.text

            # 保存缓存
            cache_file.write_text(html, encoding="utf-8")
            logger.info(f"[MedlinePlus] 已缓存: {cache_file.name}")
            return html

        except Exception as e:
            logger.error(f"[MedlinePlus] 爬取异常: {e}")
            return None

    def _extract_topic_content(
        self, html: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        从 HTML 提取主题内容（只提取 topsum_section 公共领域摘要）

        MedlinePlus 页面结构：
        - <article> 包裹全部内容
        - <section id="topsum_section"> 是 NLM 自有摘要（公共领域）
        - 内部按 <h2>/<h3> 分章节

        Args:
            html: 原始 HTML

        Returns:
            (topic_title, sections) 其中 sections = [{"title": ..., "content": ...}, ...]
        """
        soup = BeautifulSoup(html, "html.parser")

        # 提取页面标题
        topic_title = ""
        title_div = soup.find("div", class_="page-title")
        if title_div:
            h1 = title_div.find("h1")
            if h1:
                topic_title = h1.get_text(strip=True)

        # 提取 topsum_section（公共领域摘要）
        topsum = soup.find("section", {"id": "topsum_section"})
        if not topsum:
            # 回退：尝试找 article 标签
            topsum = soup.find("article")

        if not topsum:
            return topic_title, []

        # 移除脚本和样式
        for tag in topsum(["script", "style"]):
            tag.decompose()

        # 按标题标签切分章节
        # 注意：MedlinePlus 的 <h3> 嵌套在 <div class="syndicate"> 里，
        # 不是 topsum 的直接子元素，所以必须用 find_all 深度搜索
        headings = topsum.find_all(["h2", "h3"])

        if not headings:
            # 无标题结构，把整个 topsum 作为一个章节
            text = topsum.get_text(separator="\n", strip=True)
            if text:
                sections = [{"title": "Summary", "content": text}]
            else:
                sections = []
            return topic_title, sections

        # 构建 topsum 内所有元素的集合，用于边界检查
        # 防止 next_elements 跑出 topsum_section 收集到后续 section 的链接列表
        topsum_elements = set(topsum.descendants)

        sections = []
        for i, heading in enumerate(headings):
            title = heading.get_text(strip=True)
            next_heading = headings[i + 1] if i + 1 < len(headings) else None

            # 提取 heading 到下一个 heading 之间的全部文本
            # 利用 BeautifulSoup 的 next_elements 迭代器
            content_parts = []
            for elem in heading.next_elements:
                # 到达下一个标题就停止
                if next_heading and elem is next_heading:
                    break
                # 超出 topsum_section 边界就停止
                if elem not in topsum_elements:
                    break
                # 跳过标题自身的子元素
                if elem is heading or (hasattr(elem, 'name') and elem.name in ('h2', 'h3')):
                    continue
                # 收集文本节点
                if isinstance(elem, str):
                    text = elem.strip()
                    if text:
                        content_parts.append(text)

            content_text = " ".join(content_parts).strip()
            if content_text:
                sections.append({
                    "title": title,
                    "content": content_text,
                })

        # 过滤空内容和非医学章节（如 "Topic Image"、导航链接等）
        skip_titles = {
            "Topic Image", "MEDICAL ENCYCLOPEDIA",
            "Related Health Topics", "National Institutes of Health",
            "Disclaimers", "Start Here", "See, Play and Learn",
            "For You", "Resources", "Research",
        }
        sections = [
            s for s in sections
            if s["content"]
            and s["title"] not in skip_titles
            and len(s["content"]) > 20
        ]

        logger.info(
            f"[MedlinePlus] 提取 '{topic_title}': "
            f"{len(sections)} 个章节"
        )
        return topic_title, sections

    def _sections_to_documents(
        self,
        sections: List[Dict[str, str]],
        topic_title: str,
        source_url: str,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """
        将章节列表转为 Document 列表（大章节自动递归切分）

        Args:
            sections: [{"title": ..., "content": ...}, ...]
            topic_title: 主题标题
            source_url: 来源 URL
            max_chunk_size: 最大块大小
            chunk_overlap: 重叠大小

        Returns:
            Document 列表
        """
        documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        for section in sections:
            title = section["title"]
            content = section["content"]

            metadata = {
                "source": source_url,
                "topic": topic_title,
                "section": title,
                "type": "medlineplus",
                "attribution": ATTRIBUTION,
            }

            if len(content) > max_chunk_size:
                # 大章节递归切分
                chunks = splitter.split_text(content)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata=metadata.copy(),
                    ))
            else:
                documents.append(Document(
                    page_content=content,
                    metadata=metadata,
                ))

        return documents
