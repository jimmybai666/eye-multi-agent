"""VLM Factory - 创建视觉语言模型实例的工厂模块。

所有 API Key 和 URL 均从 .env 环境变量中加载，不硬编码敏感信息。
"""

import os
import base64
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 模块导入时加载 .env
load_dotenv()


@dataclass
class VLMConfig:
    """VLM 模型配置"""
    model_name: str      # 模型显示名
    model_id: str        # API 调用时的模型 ID
    api_url_env: str     # 存放 URL 的环境变量名
    api_key_env: str     # 存放 Key 的环境变量名
    max_tokens: int = 4096
    temperature: float = 0.1


# 模型注册表 —— VLM（支持图像输入的模型）
VLM_REGISTRY: dict[str, VLMConfig] = {
    "gpt5": VLMConfig(
        model_name="GPT-5",
        model_id="gpt-5",
        api_url_env="GPT_API_URL",
        api_key_env="GPT_API_KEY",
        max_tokens=4096,
        temperature=0.1,
    ),
    "sonnet4": VLMConfig(
        model_name="Claude Sonnet 4.6",
        model_id="claude-sonnet-4-6",
        api_url_env="CLAUDE_API_URL",
        api_key_env="CLAUDE_API_KEY",
        max_tokens=4096,
        temperature=0.1,
    ),
    "qwen3vl": VLMConfig(
        model_name="Qwen3-VL-8B-Instruct",
        model_id="qwen/qwen3-vl-8b-instruct",
        api_url_env="QWEN_API_URL",
        api_key_env="QWEN_API_KEY",
        max_tokens=4096,
        temperature=0.1,
    ),
    "qwen3vl_thinking": VLMConfig(
        model_name="Qwen3-VL-8B-Thinking",
        model_id="qwen/qwen3-vl-8b-thinking",
        api_url_env="QWEN_API_URL",
        api_key_env="QWEN_API_KEY",
        max_tokens=4096,
        temperature=0.6,
    ),
}


def _resolve_env(env_name: str) -> str:
    """读取环境变量，不存在则抛错。"""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(
            f"环境变量 '{env_name}' 未设置，请在 .env 文件中添加。"
        )
    return value


def encode_image(image_path: str) -> str:
    """将本地图片编码为 base64 data URL。

    Args:
        image_path: 图片文件路径

    Returns:
        base64 data URL 字符串 (data:image/xxx;base64,...)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(suffix, "image/png")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


def create_vlm(model_key: str, **overrides) -> ChatOpenAI:
    """根据 model_key 创建 VLM 实例。

    Args:
        model_key: 注册表中的模型键名，如 'gpt5', 'qwen3vl', 'qwen3vl_thinking'
        **overrides: 可覆盖任意配置字段

    Returns:
        配置好的 ChatOpenAI 实例

    Raises:
        ValueError: model_key 不存在或环境变量缺失
    """
    if model_key not in VLM_REGISTRY:
        available = ", ".join(VLM_REGISTRY.keys())
        raise ValueError(f"未知 VLM 模型: '{model_key}'。可用模型: {available}")

    config = VLM_REGISTRY[model_key]

    api_url = overrides.pop("api_url", None) or _resolve_env(config.api_url_env)
    api_key = overrides.pop("api_key", None) or _resolve_env(config.api_key_env)

    return ChatOpenAI(
        model=overrides.get("model_id", config.model_id),
        base_url=api_url,
        api_key=api_key,
        max_tokens=overrides.get("max_tokens", config.max_tokens),
        temperature=overrides.get("temperature", config.temperature),
    )


@lru_cache()
def get_vlm(model_key: str = "gpt5") -> ChatOpenAI:
    """获取缓存的 VLM 实例（同一 model_key 单例复用）。

    Args:
        model_key: 要使用的模型，默认 'gpt5'

    Returns:
        缓存的 ChatOpenAI 实例
    """
    return create_vlm(model_key)

