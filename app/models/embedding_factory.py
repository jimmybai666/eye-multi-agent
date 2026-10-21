"""Embedding 模型工厂 - 支持 OpenAI 和 Qwen"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from app.utils.logger import get_logger

# 加载 .env 文件
load_dotenv()

logger = get_logger(__name__)


# 支持的 Embedding 模型配置
EMBEDDING_MODELS = {
    # OpenAI Embeddings
    "text-embedding-3-small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
    },

    # Qwen Embeddings (兼容 OpenAI 接口)
    "qwen3.7-embedding": {
        "provider": "qwen",
        "model": "qwen/qwen3.7-text-embedding",
    },
}


def create_embeddings(
    model_key: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OpenAIEmbeddings:
    """
    创建 Embedding 模型实例

    Args:
        model_key: 模型键名（如 "text-embedding-3-small", "qwen3.7-embedding"）
        api_key: API Key（不传则从环境变量读取）
        dimensions: 向量维度（可选，覆盖默认值）

    Returns:
        OpenAIEmbeddings: Embedding 模型实例

    Raises:
        ValueError: 模型不存在或 API Key 未配置
    """
    # 检查模型是否支持
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"不支持的 Embedding 模型: {model_key}。"
            f"可用模型: {list(EMBEDDING_MODELS.keys())}"
        )

    config = EMBEDDING_MODELS[model_key]
    provider = config["provider"]

    # 获取 API Key 和 Base URL（从环境变量）
    if api_key is None:
        if provider == "openai":
            api_key = os.getenv("GPT_API_KEY")
        elif provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")

    if not api_key:
        raise ValueError(
            f"{provider.upper()}_API_KEY 未设置。"
            f"请在 .env 文件中配置或通过参数传入。"
        )

    # 构建参数
    kwargs = {
        "model": config["model"],
        "openai_api_key": api_key,
    }

    # 读取 base_url（从环境变量，必需）
    if provider == "openai":
        base_url = os.getenv("GPT_API_URL")
        if not base_url:
            raise ValueError("GPT_API_URL 未设置，请在 .env 文件中配置")
    elif provider == "qwen":
        base_url = os.getenv("QWEN_API_URL")
        if not base_url:
            raise ValueError("QWEN_API_URL 未设置，请在 .env 文件中配置")

        # Qwen API 限制：
        # 1. 不支持预tokenization，禁用长度检查
        # 2. 批处理大小最大为 20
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 20

    # 设置 base_url
    kwargs["openai_api_base"] = base_url

    # 自定义维度（仅当用户显式传入时）
    if dimensions:
        kwargs["dimensions"] = dimensions

    logger.info(
        f"[Embedding] 创建模型: {model_key} | provider={provider} | "
        f"dimensions={dimensions or 'default'}"
    )

    return OpenAIEmbeddings(**kwargs)


def list_embedding_models() -> list[str]:
    """列出所有支持的 Embedding 模型"""
    return list(EMBEDDING_MODELS.keys())
