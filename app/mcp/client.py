"""eye-mcp 客户端封装

通过 MCP stdio 协议连接 eye-mcp 服务，调用图像分析工具。
eye-mcp 服务独立部署在 EYE_MCP_DIR，本模块只做客户端调用。

可用工具：
  - oct_segment: OCT 语义分割（视网膜/脉络膜/囊肿/黄斑裂孔）
  - fovea_locate: 中央凹定位
  - optic_disc_cup_seg: 视杯视盘分割 + CDR
  - vessel_seg: 动静脉血管分割
  - lesion_seg: 四类病灶分割（MA/HE/EX/SE）
"""

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.utils.logger import get_logger

logger = get_logger(__name__)

# eye-mcp 项目根目录，默认与本项目同级（../eye-mcp），可通过环境变量覆盖
EYE_MCP_DIR = os.getenv(
    "EYE_MCP_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent / "eye-mcp"),
)

# MCP 工具返回的 mask/overlay base64 图片，存到本地文件而非 state
_IMAGE_KEYS = {"mask_base64", "overlay_base64"}

# 分割结果图片默认存放目录
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "mcp_images"


def _save_images(result: dict, tool_name: str, output_dir: Path | None = None) -> dict:
    """将结果中的 base64 图片存为本地文件，替换为文件路径

    base64 图片数据太大不适合放在 LangGraph state 中传递，
    但下游 Critic (VLM) 需要查看分割图来验证诊断，所以存到本地文件。
    """
    save_dir = output_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    cleaned = {}
    for k, v in result.items():
        if k in _IMAGE_KEYS and isinstance(v, str):
            suffix = k.replace("_base64", "")  # mask / overlay
            file_path = save_dir / f"{tool_name}_{suffix}.png"
            try:
                file_path.write_bytes(base64.b64decode(v))
                cleaned[k.replace("_base64", "_path")] = str(file_path)
            except Exception as e:
                logger.warning(f"[MCP] Failed to decode/save {k} for {tool_name}: {e}")
        else:
            cleaned[k] = v
    return cleaned


def read_image_as_base64(image_path: str) -> str:
    """读取图片文件并返回 raw base64 字符串（非 data URL）

    注意：MCP 服务端接收 raw base64，与 VLM 的 encode_image（data URL 格式）不同
    """
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


class EyeMCPClient:
    """eye-mcp 服务的客户端"""

    def __init__(self, server_dir: str | None = None):
        self.server_dir = server_dir or EYE_MCP_DIR

    @asynccontextmanager
    async def open_session(self, image_base64: str, output_dir: Path | None = None):
        """保持一个 MCP 子进程存活，供 ReAct 循环中多次工具调用共享

        用法：
            async with client.open_session(image_b64, output_dir) as session:
                r1 = await session.call("vessel_seg")
                r2 = await session.call("lesion_seg")
            # 子进程在此自动关闭
        """
        server_params = StdioServerParameters(
            command="uv",
            args=["run", "eye-mcp"],
            cwd=self.server_dir,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("[MCP] 长连接 session 已建立")
                yield ManagedMCPSession(session, image_base64, output_dir)
                logger.info("[MCP] 长连接 session 关闭")


class ManagedMCPSession:
    """单个 MCP 子进程 session，供 ReAct 循环中多次工具调用共享

    image_base64 在创建时传入，后续每次 call() 自动附带，
    工具函数不需要关心图片传递。
    """

    def __init__(self, session: ClientSession, image_base64: str, output_dir: Path | None):
        self._session = session
        self._image_base64 = image_base64
        self._output_dir = output_dir
        self.results: dict[str, dict] = {}

    async def list_tools(self) -> list[dict]:
        """从 MCP 服务端动态获取可用工具列表（名称+描述）

        用于自动发现：vision agent 不需要硬编码工具列表，
        eye-mcp 新增工具后自动可用。
        """
        result = await self._session.list_tools()
        return [
            {"name": t.name, "description": t.description or ""}
            for t in result.tools
        ]

    async def call(self, tool_name: str) -> dict:
        """调用单个 MCP 工具，结果自动保存图片并记录到 self.results"""
        arguments = {"image_base64": self._image_base64}
        try:
            result = await self._session.call_tool(tool_name, arguments)
            response = json.loads(result.content[0].text)
            cleaned = _save_images(response, tool_name, self._output_dir)
            self.results[tool_name] = cleaned
            logger.info(f"[MCP] {tool_name} 完成")
            return cleaned
        except Exception as e:
            logger.error(f"[MCP] {tool_name} 失败: {e}")
            error_result = {"error": str(e)}
            self.results[tool_name] = error_result
            return error_result
