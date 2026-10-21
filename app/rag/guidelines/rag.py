"""Guidelines RAG - 临床指南检索增强生成

功能：
1. 从 PDF 指南文件提取文本
2. 构建和管理 FAISS 向量索引
3. 向量检索临床指南内容

数据来源：
- ICO Guidelines（青光眼、糖尿病眼病）
- AOA Clinical Practice Guidelines
- ESCRS Cataract Surgery Guideline
"""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.models import create_embeddings
from app.rag.guidelines.pdf_extractor import pdf_to_documents
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GuidelinesRAG:
    """临床指南 RAG"""

    def __init__(
        self,
        data_dir: str = "data/rag/guidelines",
        embedding_model: str = "qwen3.7-embedding",
    ):
        """
        初始化

        Args:
            data_dir: 数据目录
            embedding_model: Embedding 模型
        """
        self.data_dir = Path(data_dir)
        self.pdf_dir = self.data_dir / "pdf"
        self.cache_dir = self.data_dir / "cache"
        self.index_dir = self.data_dir / "index"

        # 创建目录
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Embeddings
        logger.info(f"[Guidelines] 使用 Embedding 模型: {embedding_model}")
        self.embeddings = create_embeddings(model_key=embedding_model)

        # 向量数据库
        self.vector_store = None

    def build_index(self, force_rebuild: bool = False):
        """
        构建索引（扫描 pdf/ 下所有 PDF）

        Args:
            force_rebuild: 是否强制重建
        """
        index_file = self.index_dir / "index.faiss"
        if index_file.exists() and not force_rebuild:
            logger.info("[Guidelines] 索引已存在，跳过构建")
            self._load_index()
            return

        all_documents = []

        # 处理 PDF 文件
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"[Guidelines] 发现 {len(pdf_files)} 个 PDF 文件")

        if not pdf_files:
            raise ValueError(
                f"[Guidelines] {self.pdf_dir} 下没有 PDF 文件。"
                "请先下载临床指南 PDF 放到该目录。"
            )

        for pdf_file in pdf_files:
            guideline_name = self._pdf_name_to_label(pdf_file.stem)
            logger.info(f"[Guidelines] 处理 PDF: {pdf_file.name} → {guideline_name}")

            docs = pdf_to_documents(
                pdf_path=str(pdf_file),
                guideline_name=guideline_name,
                cache_dir=str(self.cache_dir),
            )
            if docs:
                all_documents.extend(docs)
                logger.info(f"[Guidelines] ✓ {pdf_file.name}: {len(docs)} 个块")
            else:
                logger.warning(f"[Guidelines] ✗ {pdf_file.name}: 提取失败")

        if not all_documents:
            raise ValueError("[Guidelines] 没有从 PDF 中提取到任何文档。")

        logger.info(f"[Guidelines] 共 {len(all_documents)} 个文档块，开始构建索引...")

        # 构建向量数据库
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)

        # 保存
        self.vector_store.save_local(str(self.index_dir))

        # 保存元信息
        meta = {
            "pdf_files": [f.name for f in pdf_files],
            "total_documents": len(all_documents),
        }
        meta_file = self.index_dir / "meta.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(f"[Guidelines] 索引已保存: {index_file}")

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        检索相关指南内容

        Args:
            query: 查询（如 "glaucoma diagnosis criteria"）
            k: 返回结果数量

        Returns:
            Document 列表
        """
        if self.vector_store is None:
            self._load_index()

        logger.info(f"[Guidelines] 检索: {query}")
        docs = self.vector_store.similarity_search(query, k=k)
        logger.info(f"[Guidelines] 找到 {len(docs)} 个相关文档")
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
                        "source": 来源文件,
                        "guideline": 指南名称,
                        "section": 章节名,
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
                "guideline": doc.metadata.get("guideline", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "preview": doc.page_content[:200].replace("\n", " ") + "...",
            })

        return {
            "query": query,
            "total": len(results),
            "results": results,
        }

    def list_sources(self) -> dict:
        """
        列出当前索引包含的指南来源

        Returns:
            {"pdf_files": [...], "total_documents": int}
        """
        meta_file = self.index_dir / "meta.json"
        if meta_file.exists():
            return json.loads(meta_file.read_text(encoding="utf-8"))

        return {
            "pdf_files": [f.name for f in self.pdf_dir.glob("*.pdf")],
            "total_documents": 0,
        }

    def _load_index(self):
        """加载索引"""
        index_file = self.index_dir / "index.faiss"

        if not index_file.exists():
            raise ValueError(
                "[Guidelines] 索引不存在，请先调用 build_index()。"
                f"请确保 {self.pdf_dir} 下有 PDF 文件。"
            )

        logger.info("[Guidelines] 加载索引...")
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("[Guidelines] 索引加载成功")

    @staticmethod
    def _pdf_name_to_label(stem: str) -> str:
        """
        将 PDF 文件名转为可读标签

        例：
        - AOA_Glaucoma_2024 → AOA Glaucoma 2024
        - ICO_Diabetic_Eye_Care → ICO Diabetic Eye Care
        """
        return stem.replace("_", " ").replace("-", " ")
