"""EyeWiki RAG - 眼科知识检索模块

功能：
1. 检索关键词（如疾病名、症状）
2. 构建和管理索引
3. 向量检索
4. 自动缓存（网页和索引）

"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models import create_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EyeWikiRAG:
    """EyeWiki 检索增强生成"""

    def __init__(
        self,
        data_dir: str = "data/rag/eyewiki",
        embedding_model: str = "qwen3.7-embedding",
    ):
        """
        初始化

        Args:
            data_dir: 数据目录（缓存和索引）
            embedding_model: Embedding 模型（"qwen3.7-embedding" 或 "text-embedding-3-small"）
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.index_dir = self.data_dir / "index"

        # 创建目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 创建 Embeddings
        logger.info(f"[EyeWiki] 使用 Embedding 模型: {embedding_model}")
        self.embeddings = create_embeddings(model_key=embedding_model)

        # 向量数据库
        self.vector_store = None

    def build_index(
        self,
        diseases: List[str],
        force_rebuild: bool = False,
        use_category: bool = True,
        max_articles_per_category: int = 10,
        auto_detect: bool = True
    ):
        """
        构建索引

        Args:
            diseases: 疾病列表（可以是分类名或具体页面名）
            force_rebuild: 是否强制重建（即使索引已存在）
            use_category: 是否使用分类页面爬取（auto_detect=False 时生效）
            max_articles_per_category: 每个分类最多爬取的文章数（避免过多）
            auto_detect: 是否自动检测输入类型（推荐=True）
        """
        # 检查索引是否存在
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force_rebuild:
            logger.info("[EyeWiki] 索引已存在，跳过构建")
            self._load_index()
            return

        logger.info(f"[EyeWiki] 开始构建索引: {len(diseases)} 个疾病")
        logger.info(f"[EyeWiki] 模式: {'分类爬取' if use_category else '单页爬取'}")

        # 收集所有文档
        all_documents = []
        for disease in diseases:
            if use_category:
                # 从分类页面获取所有相关文章
                articles = self._fetch_category_articles(disease, max_articles_per_category)
                logger.info(f"[EyeWiki] {disease}: 找到 {len(articles)} 篇文章")

                for article_name in articles:
                    url = f"https://eyewiki.aao.org/{article_name}"
                    docs = self._process_url(url, disease)
                    if docs:
                        all_documents.extend(docs)
            else:
                # 单页模式：直接爬取指定页面
                url = f"https://eyewiki.aao.org/{disease}"
                docs = self._process_url(url, disease)
                if docs:
                    all_documents.extend(docs)

        if not all_documents:
            raise ValueError("[EyeWiki] 没有成功获取任何文档")

        logger.info(f"[EyeWiki] 共 {len(all_documents)} 个文档块")

        # 构建向量数据库
        logger.info("[EyeWiki] 构建向量索引...")
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)

        # 保存
        self.vector_store.save_local(str(self.index_dir))
        logger.info(f"[EyeWiki] 索引已保存: {index_file}")

    def build_index_from_articles(
        self,
        articles: List[str],
        force_rebuild: bool = False,
        category_label: str = "Selected"
    ):
        """
        从精选的文章列表构建索引（Agent 精选后使用）

        Args:
            articles: 文章名称列表（如 ["Primary_Open-Angle_Glaucoma", ...]）
            force_rebuild: 是否强制重建
            category_label: 分类标签（用于 metadata）

        Example:
            # Agent 精选后
            selected_articles = [
                "Primary_Open-Angle_Glaucoma",
                "Glaucoma_Diagnosis"
            ]
            rag.build_index_from_articles(selected_articles)
        """
        # 检查索引是否存在
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force_rebuild:
            logger.info("[EyeWiki] 索引已存在，跳过构建")
            self._load_index()
            return

        logger.info(f"[EyeWiki] 从精选文章构建索引: {len(articles)} 篇")

        # 收集所有文档
        all_documents = []
        for article_name in articles:
            url = f"https://eyewiki.aao.org/{article_name}"
            docs = self._process_url(url, category_label)
            if docs:
                all_documents.extend(docs)
                logger.info(f"[EyeWiki] ✓ {article_name}: {len(docs)} 个块")
            else:
                logger.warning(f"[EyeWiki] ✗ {article_name}: 爬取失败")

        if not all_documents:
            raise ValueError("[EyeWiki] 没有成功获取任何文档")

        logger.info(f"[EyeWiki] 共 {len(all_documents)} 个文档块")

        # 构建向量数据库
        logger.info("[EyeWiki] 构建向量索引...")
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)

        # 保存
        self.vector_store.save_local(str(self.index_dir))
        logger.info(f"[EyeWiki] 索引已保存: {index_file}")





    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询关键词（如 "glaucoma symptoms"）
            k: 返回结果数量

        Returns:
            List[Document]: 检索到的文档列表
        """
        # 加载索引
        if self.vector_store is None:
            self._load_index()

        logger.info(f"[EyeWiki] 检索: {query}")

        # 向量检索
        docs = self.vector_store.similarity_search(query, k=k)

        logger.info(f"[EyeWiki] 找到 {len(docs)} 个相关文档")
        return docs

    def search_with_sources(self, query: str, k: int = 3) -> dict:
        """
        检索相关文档并返回详细来源信息（方便"跳转"）

        Args:
            query: 查询关键词
            k: 返回结果数量

        Returns:
            {
                "query": 查询内容,
                "results": [
                    {
                        "content": 文档内容,
                        "source": 来源URL,
                        "disease": 疾病分类,
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
                "disease": doc.metadata.get("disease", "N/A"),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })

        return {
            "query": query,
            "total": len(results),
            "results": results
        }

    def _load_index(self):
        """加载索引"""
        index_file = self.index_dir / "index.faiss"

        if not index_file.exists():
            raise ValueError(
                "[EyeWiki] 索引不存在，请先调用 build_index()"
            )

        logger.info("[EyeWiki] 加载索引...")
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("[EyeWiki] 索引加载成功")

    def _process_url(self, url: str, disease: str) -> List[Document]:
        """
        处理单个 URL：爬取 → 缓存 → 切分

        Args:
            url: 网页 URL
            disease: 疾病名

        Returns:
            Document 列表
        """
        # 获取文本（带缓存）
        text = self._fetch_with_cache(url)
        if not text:
            logger.warning(f"[EyeWiki] 无法获取: {url}")
            return []

        # 按章节切分文本
        # 医学文档通常有清晰的章节结构（如 Signs and Symptoms, Diagnosis, Treatment）
        # 按章节切分保持完整的语义单元
        chunks = self._split_by_sections(text)

        # 如果没有识别到章节结构，回退到递归分割
        if not chunks:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(text)

        # 创建 Document（过滤过短 chunk 和导航噪音）
        MIN_CHUNK_CHARS = 100
        NOISE_MARKERS = ["Jump to:", "navigation", "From EyeWiki", "View history", "All content on Eyewiki"]
        documents = []
        for chunk in chunks:
            text = chunk.strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            if any(marker in text for marker in NOISE_MARKERS):
                continue
            documents.append(Document(
                page_content=text,
                metadata={"source": url, "disease": disease}
            ))

        logger.info(f"[EyeWiki] {disease}: {len(documents)} 个文档块")
        return documents

    def _fetch_with_cache(self, url: str) -> Optional[str]:
        """
        爬取网页（带缓存）

        Args:
            url: 网页 URL

        Returns:
            网页纯文本（失败返回 None）
        """
        # 生成缓存文件名
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{url_hash}.txt"

        # 检查缓存
        if cache_file.exists():
            logger.info(f"[EyeWiki] 从缓存加载: {url}")
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        # 爬取
        try:
            logger.info(f"[EyeWiki] 爬取: {url}")
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                logger.error(f"[EyeWiki] 失败 {response.status_code}: {url}")
                return None

            # 解析
            soup = BeautifulSoup(response.text, "html.parser")

            # 移除脚本和样式
            for tag in soup(["script", "style"]):
                tag.decompose()

            # 提取主内容区域（排除导航、侧边栏、页脚）
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                content_div = soup.find("div", {"id": "bodyContent"})
            if not content_div:
                content_div = soup

            # 移除 EyeWiki 页面中的非内容元素
            for tag in content_div.find_all(["div"], class_=["navbox", "catlinks", "mw-references-wrap", "toc", "printfooter", "mw-indicators"]):
                tag.decompose()
            for tag in content_div.find_all(["table"], class_=["infobox", "navbox"]):
                tag.decompose()

            text = content_div.get_text(separator="\n", strip=True)

            # 保存缓存
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(text)

            logger.info(f"[EyeWiki] 已缓存: {cache_file.name}")
            return text

        except Exception as e:
            logger.error(f"[EyeWiki] 异常: {e}")
            return None

    def _split_by_sections(self, text: str, max_chunk_size: int = 1500) -> List[str]:
        """
        按章节结构智能切分文本（带大小兜底）

        策略：
        1. 识别章节标题（如 "Signs and Symptoms", "Treatment"）
        2. 按章节边界切分，保持语义完整
        3. 如果单个章节 > max_chunk_size，再次递归切分
        4. 如果识别不到章节，返回空（回退到常规切分）
        5. 每个 chunk 前加章节标题前缀

        Args:
            text: 原始文本
            max_chunk_size: 单个块的最大大小（字符数）

        Returns:
            切分后的文本块列表
        """
        import re

        section_pattern = re.compile(r'\n([A-Z][A-Za-z\s/\-]{3,50})\n')

        matches = list(section_pattern.finditer(text))

        if len(matches) < 2:
            return []

        # 跳过噪音章节标题
        skip_titles = {
            "REFERENCES", "CATEGORIES", "NAVIGATION", "EXTERNAL LINKS",
            "SEE ALSO", "FURTHER READING", "NOTES",
        }

        chunks = []
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            if title.upper() in skip_titles:
                continue

            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if len(section_text) > max_chunk_size:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                sub_chunks = splitter.split_text(section_text)
                for sc in sub_chunks:
                    chunks.append(f"[Section: {title}]\n{sc}")
            else:
                chunks.append(f"[Section: {title}]\n{section_text}")

        logger.info(f"[EyeWiki] 按章节切分: {len(chunks)} 个块")
        return chunks

    def _fetch_category_articles(self, category: str, max_articles: int = 10) -> List[str]:
        """
        从分类页面获取文章列表

        Args:
            category: 分类名称（如 "Glaucoma"）
            max_articles: 最多获取的文章数量

        Returns:
            文章名称列表（URL 路径部分）
        """
        # 构建分类页面 URL
        category_url = f"https://eyewiki.aao.org/Category:{category}"

        try:
            logger.info(f"[EyeWiki] 获取分类页面: {category_url}")
            response = requests.get(category_url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"[EyeWiki] 分类页面失败 {response.status_code}: {category_url}")
                return []

            soup = BeautifulSoup(response.text, "html.parser")

            # 查找分类页面中的文章链接
            # EyeWiki 的分类页面通常在 div.mw-category 或 div.mw-category-group 中
            articles = []

            # 方法1：从 mw-category 容器中提取
            category_div = soup.find("div", {"id": "mw-pages"})
            if category_div:
                links = category_div.find_all("a", href=True)
                for link in links:
                    href = link.get("href", "")

                    # 过滤：只要主文章链接，排除特殊页面
                    if href.startswith("/") and not any(x in href for x in [
                        "Category:", "Template:", "Special:", "Help:",
                        "File:", "MediaWiki:", "Talk:", "User:"
                    ]):
                        # 提取文章名（去掉开头的 /）
                        article_name = href.lstrip("/")
                        if article_name and article_name not in articles:
                            articles.append(article_name)

                            # 限制数量
                            if len(articles) >= max_articles:
                                break

            logger.info(f"[EyeWiki] 从分类 '{category}' 找到 {len(articles)} 篇文章（限制: {max_articles}）")
            return articles[:max_articles]

        except Exception as e:
            logger.error(f"[EyeWiki] 获取分类失败: {e}")
            return []
