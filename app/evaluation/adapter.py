"""Adapter — 将评测任务适配为 Agent Pipeline 的输入输出

职责：
1. 模型切换：根据 model_key 替换 VLM/LLM registry 配置
2. Pipeline 调用：构建 initial_state，跑 graph.ainvoke()
3. 结果提取：从 pipeline 输出中提取可供 parse_response 解析的文本
"""

import asyncio
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── MCP 全局长连接（仅评测时使用）──────────────────────────

_mcp_context_stack = None  # stdio_client context
_mcp_session_ctx = None    # ClientSession context
_mcp_raw_session = None    # raw ClientSession instance
_mcp_lock = asyncio.Lock()


async def warmup_mcp():
    """预启动 MCP 子进程，后续所有 vision 调用复用此连接。"""
    global _mcp_context_stack, _mcp_session_ctx, _mcp_raw_session

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from app.mcp.client import EYE_MCP_DIR

    server_params = StdioServerParameters(
        command="uv", args=["run", "eye-mcp"], cwd=EYE_MCP_DIR,
    )
    _mcp_context_stack = stdio_client(server_params)
    read, write = await _mcp_context_stack.__aenter__()
    _mcp_session_ctx = ClientSession(read, write)
    _mcp_raw_session = await _mcp_session_ctx.__aenter__()
    await _mcp_raw_session.initialize()
    logger.info("[Adapter] MCP 全局长连接已建立")


async def shutdown_mcp():
    """关闭全局 MCP 连接。"""
    global _mcp_context_stack, _mcp_session_ctx, _mcp_raw_session
    if _mcp_session_ctx:
        await _mcp_session_ctx.__aexit__(None, None, None)
    if _mcp_context_stack:
        await _mcp_context_stack.__aexit__(None, None, None)
    _mcp_raw_session = None
    _mcp_session_ctx = None
    _mcp_context_stack = None
    logger.info("[Adapter] MCP 全局长连接已关闭")


def patch_mcp_reuse():
    """Monkey-patch EyeMCPClient.open_session，复用全局 MCP 子进程。"""
    from app.mcp.client import EyeMCPClient, ManagedMCPSession

    @asynccontextmanager
    async def _reuse_session(self, image_base64, output_dir=None):
        global _mcp_raw_session
        async with _mcp_lock:
            if _mcp_raw_session is None:
                await warmup_mcp()
        managed = ManagedMCPSession(_mcp_raw_session, image_base64, output_dir)
        yield managed

    EyeMCPClient.open_session = _reuse_session
    logger.info("[Adapter] MCP session 复用已启用")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PRESETS = {
    "agent_gpt5": {
        "display_name": "Agent (GPT-5)",
        "vlm": {
            "model_id": "gpt-5",
            "api_url_env": "GPT_API_URL",
            "api_key_env": "GPT_API_KEY",
        },
        "llm": {
            "model_id": "gpt-5",
            "api_url_env": "GPT_API_URL",
            "api_key_env": "GPT_API_KEY",
        },
    },
    "agent_qwen3vl": {
        "display_name": "Agent (Qwen3-VL)",
        "vlm": {
            "model_id": "qwen/qwen3-vl-8b-instruct",
            "api_url_env": "QWEN_API_URL",
            "api_key_env": "QWEN_API_KEY",
        },
        "llm": {
            "model_id": "qwen/qwen3.7-plus",
            "api_url_env": "QWEN_API_URL",
            "api_key_env": "QWEN_API_KEY",
        },
    },
}

def apply_model_preset(model_key: str):
    """替换 VLM/LLM registry 中 gpt5 的配置，实现模型切换。

    所有 agent 都用 get_vlm("gpt5") / get_llm("gpt5")，
    所以只需把 registry 里 gpt5 条目的 model_id 和 api env 替换掉，
    再清除 lru_cache，下次调用就会用新配置创建实例。
    """
    if model_key not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {model_key}. Available: {list(MODEL_PRESETS.keys())}")

    preset = MODEL_PRESETS[model_key]

    from app.models.vlm_factory import VLM_REGISTRY, get_vlm
    from app.models.llm_factory import LLM_REGISTRY, get_llm

    vlm_cfg = VLM_REGISTRY["gpt5"]
    vlm_cfg.model_id = preset["vlm"]["model_id"]
    vlm_cfg.api_url_env = preset["vlm"]["api_url_env"]
    vlm_cfg.api_key_env = preset["vlm"]["api_key_env"]

    llm_cfg = LLM_REGISTRY["gpt5"]
    llm_cfg.model_id = preset["llm"]["model_id"]
    llm_cfg.api_url_env = preset["llm"]["api_url_env"]
    llm_cfg.api_key_env = preset["llm"]["api_key_env"]

    get_vlm.cache_clear()
    get_llm.cache_clear()

    patch_mcp_reuse()

    logger.info(f"[Adapter] Model preset applied: {model_key} "
                f"(VLM={preset['vlm']['model_id']}, LLM={preset['llm']['model_id']})")


_graph = None

def get_graph():
    """懒加载编译好的 LangGraph（全局单例）"""
    global _graph
    if _graph is None:
        from app.graph.builder import build_graph
        _graph = build_graph()
    return _graph


async def run_pipeline(image_path: str, task_prompt: str) -> dict:
    """跑一次完整 agent pipeline，返回结果 dict + 逐节点计时。

    返回的 dict 除了正常 state 字段外，额外包含：
    - _node_outputs: {node_name: node_output_dict} 每个节点的原始输出
    - _node_timings: {node_name: seconds} 每个节点的耗时
    - _pipeline_time: 总耗时（秒）
    """
    from app.state.state import create_initial_state

    graph = get_graph()
    initial_state = create_initial_state(
        input_text=task_prompt,
        image_path=image_path,
    )

    node_outputs = {}
    node_timings = {}
    final_state = {}

    t_start = time.perf_counter()
    t_node_start = t_start

    async for chunk in graph.astream(initial_state):
        now = time.perf_counter()
        for node_name, node_output in chunk.items():
            node_timings[node_name] = round(now - t_node_start, 2)
            t_node_start = now

            # 保存节点原始输出（深拷贝避免引用问题）
            node_outputs[node_name] = _safe_serialize_node(node_name, node_output)

            if isinstance(node_output, dict):
                final_state.update(node_output)

    pipeline_time = round(time.perf_counter() - t_start, 2)

    final_state["_node_outputs"] = node_outputs
    final_state["_node_timings"] = node_timings
    final_state["_pipeline_time"] = pipeline_time

    return final_state


def _safe_serialize_node(node_name: str, output: dict) -> dict:
    """将节点输出转为可 JSON 序列化的 dict（去掉 LangChain message 对象）"""
    if not isinstance(output, dict):
        return {"_raw": str(output)}

    safe = {}
    for k, v in output.items():
        if k == "messages":
            safe[k] = [_msg_to_dict(m) for m in v if m is not None]
        elif k == "evidence" and v is not None:
            try:
                import json
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
        else:
            try:
                import json
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
    return safe


def _msg_to_dict(msg) -> dict:
    """将 LangChain BaseMessage 转为可序列化的 dict"""
    if msg is None:
        return {}
    content = getattr(msg, "content", str(msg))
    return {
        "type": type(msg).__name__,
        "content": content[:3000] if isinstance(content, str) else str(content)[:3000],
    }


def extract_text_for_parsing(result: dict) -> str:
    """从 pipeline 结果中提取完整报告文本。"""
    report = result.get("report")
    if isinstance(report, dict):
        report = report.get("report", "")
    if report and isinstance(report, str) and len(report.strip()) > 10:
        return report

    clinical = result.get("clinical_result")
    if clinical:
        diagnosis = clinical.get("diagnosis")
        if isinstance(diagnosis, dict):
            parts = []
            if diagnosis.get("primary_diagnosis"):
                parts.append(f"Primary diagnosis: {diagnosis['primary_diagnosis']}")
            if diagnosis.get("clinical_reasoning"):
                parts.append(diagnosis["clinical_reasoning"])
            if diagnosis.get("raw"):
                parts.append(diagnosis["raw"])
            if parts:
                return "\n".join(parts)
        elif isinstance(diagnosis, str):
            return diagnosis

    return ""


JUDGE_PROMPT = """You are a strict label extractor.

## Task
Read ONLY the "Final Diagnosis" section below and answer the original question with one label.

## Original question
{question}

## Final Diagnosis (from report)
{report_excerpt}

## Rules
- ONLY use information from the Final Diagnosis section above.
- Output EXACTLY one label from: {labels}
- If the Final Diagnosis does not clearly state an answer, output UNKNOWN.
- No explanation. Just the label.
"""


async def judge_extract_label(result: dict, task: dict) -> str:
    """用一次轻量 LLM 调用，从 pipeline 报告中提取分类标签。

    比正则匹配更可靠：LLM 能理解 "no glaucomatous damage" → NON-GLAUCOMA，
    "mild NPDR" → Grade 1 等语义映射。

    Args:
        result: pipeline 完整输出 state dict
        task: 评测任务配置（含 prompt、labels 等）

    Returns:
        提取的标签字符串，匹配失败返回 "UNKNOWN"
    """
    from app.models.llm_factory import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    report_text = extract_text_for_parsing(result)
    if not report_text:
        return "UNKNOWN"

    # 截取报告关键部分（Final Diagnosis 优先），避免超长
    excerpt = _extract_key_sections(report_text)

    labels_str = " / ".join(task["labels"])
    prompt = JUDGE_PROMPT.format(
        question=task["prompt"],
        report_excerpt=excerpt,
        labels=labels_str,
    )

    llm = get_llm()
    try:
        response = await llm.bind(max_tokens=32).ainvoke([
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        # 清理：去掉可能的引号、句号、多余空白
        raw = raw.strip('"\' .')
        label = _match_label(raw, task["labels"])
        return label
    except Exception as e:
        logger.warning(f"[Judge] LLM call failed: {e}")
        return "UNKNOWN"


def _extract_key_sections(report: str) -> str:
    """从 markdown 报告中只截取 Final Diagnosis 段落"""
    lines = report.split("\n")
    key_sections = []
    capturing = False

    for line in lines:
        header_lower = line.strip().lower()
        if "final diagnosis" in header_lower:
            capturing = True
        elif line.startswith("## ") and capturing:
            break
        if capturing:
            key_sections.append(line)

    if key_sections:
        return "\n".join(key_sections)[:1500]
    return "UNKNOWN"


def _match_label(raw: str, labels: list[str]) -> str:
    """将 LLM 输出匹配到最接近的合法标签"""
    raw_upper = raw.upper().strip()

    for label in labels:
        if raw_upper == label.upper():
            return label

    # 最长标签优先匹配，避免 "GLAUCOMA" 抢在 "NON-GLAUCOMA" 之前
    sorted_labels = sorted(labels, key=len, reverse=True)
    for label in sorted_labels:
        if label.upper() in raw_upper:
            return label

    return "UNKNOWN"
