"""Critic agent — VLM 多模态交叉审核

轻量设计：单次 VLM 调用，不做 RAG，不做循环。
输入：Vision findings + Clinical diagnosis + Evidence 摘要 + overlay 分割图
输出：结构化 JSON（approved, issues, suggestions, confidence_assessment）
"""

import asyncio
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.state.schema import AgentState
from app.agents.critic.prompt import CRITIC_SYSTEM_PROMPT
from app.models.vlm_factory import get_vlm
from app.rag.evidence import EvidencePackage, format_evidence_for_critic
from app.utils.logger import get_logger

logger = get_logger(__name__)

OVERLAY_PRIORITY = ["optic_disc_cup_seg", "lesion_seg", "vessel_seg"]
MAX_OVERLAY_IMAGES = 2


async def critic_node(state: AgentState) -> dict:
    """VLM 交叉审核：比对 Vision 和 Clinical 的输出，找矛盾和遗漏"""
    vlm = get_vlm()
    iteration = state.get("iteration_count", 0)

    if not state.get("clinical_result"):
        logger.info("[Critic] No clinical_result, auto-approve")
        return {
            "messages": [],
            "critic_result": {
                "approved": True,
                "issues": [],
                "suggestions": [],
                "confidence_assessment": "adequate",
                "feedback": "No clinical result to review.",
            },
            "iteration_count": iteration + 1,
        }

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=_build_critic_input(state)),
    ]

    max_attempts = 2
    response = None
    for attempt in range(max_attempts):
        try:
            response = await vlm.ainvoke(messages)
            if response.content and response.content.strip():
                break
            logger.warning(f"[Critic] VLM returned empty (attempt {attempt + 1})")
        except Exception as api_err:
            logger.error(f"[Critic] VLM API call failed (attempt {attempt + 1}): {api_err}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return {
                "messages": [],
                "critic_result": {
                    "approved": True,
                    "issues": [],
                    "suggestions": [],
                    "confidence_assessment": "insufficient",
                    "feedback": f"Critic VLM call failed: {api_err}. Auto-approved to avoid blocking pipeline.",
                },
                "iteration_count": iteration + 1,
            }

    parsed = _parse_critic_output(response.content if response else "")

    feedback = _build_feedback_text(parsed)

    return {
        "messages": [response] if response else [],
        "critic_result": {
            "approved": parsed["approved"],
            "issues": parsed.get("issues", []),
            "suggestions": parsed.get("suggestions", []),
            "confidence_assessment": parsed.get("confidence_assessment", "weak"),
            "feedback": feedback,
        },
        "iteration_count": iteration + 1,
    }


# ── 输入构建 ──────────────────────────────────────────


def _build_critic_input(state: AgentState) -> list:
    """构建 VLM 多模态输入：overlay 图片 + 文本"""
    content_parts = []

    overlay_paths = _collect_overlay_images(state.get("vision_result", {}))
    for path in overlay_paths:
        try:
            data_url = _encode_resized(path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        except Exception as e:
            logger.warning(f"[Critic] Failed to load overlay {path}: {e}")

    text = _build_critic_text(state)
    content_parts.append({"type": "text", "text": text})

    return content_parts


def _build_critic_text(state: AgentState) -> str:
    """构建审核文本：Vision + Clinical + Evidence"""
    parts = ["Please cross-check the following diagnostic outputs:\n"]

    if state.get("vision_result"):
        vr = state["vision_result"]
        parts.append(f"## Vision Agent Findings\n{vr.get('findings', '(no findings)')}")
        parts.append(f"Image type: {vr.get('image_type', 'unknown')}\n")

    if state.get("clinical_result"):
        cr = state["clinical_result"]
        diagnosis = cr.get("diagnosis", "(no diagnosis)")
        if isinstance(diagnosis, dict):
            parts.append(f"## Clinical Agent Diagnosis\n{json.dumps(diagnosis, indent=2, ensure_ascii=False)}\n")
        else:
            parts.append(f"## Clinical Agent Diagnosis\n{diagnosis}\n")

    if state.get("evidence"):
        try:
            package = EvidencePackage(**state["evidence"])
            evidence_text = format_evidence_for_critic(package)
            parts.append(f"## Evidence Summary\n{evidence_text}\n")
        except Exception as e:
            logger.warning(f"[Critic] Failed to parse evidence: {e}")

    return "\n".join(parts)


def _collect_overlay_images(vision_result: dict) -> list[str]:
    """从 mcp_results 中按优先级提取 overlay 图片路径"""
    mcp_results = vision_result.get("mcp_results", {})
    paths = []

    for tool_name in OVERLAY_PRIORITY:
        if tool_name in mcp_results:
            result = mcp_results[tool_name]
            overlay = result.get("overlay_path")
            if overlay and Path(overlay).exists():
                paths.append(overlay)
            if len(paths) >= MAX_OVERLAY_IMAGES:
                break

    return paths


def _encode_resized(image_path: str, max_side: int = 768) -> str:
    """将图片缩放到 max_side 以内并编码为 base64 data URL"""
    from PIL import Image
    import io
    import base64

    img = Image.open(image_path)
    img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── 输出解析 ──────────────────────────────────────────


def _parse_critic_output(content: str) -> dict:
    """三层防御解析 VLM 的 JSON 输出，失败默认 not approved"""
    text = content.strip()

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "approved" in data:
                data["approved"] = bool(data["approved"])
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"[Critic] JSON parse failed, defaulting to not approved: {content[:200]}")
    return {
        "approved": False,
        "issues": ["Critic output could not be parsed — manual review recommended"],
        "suggestions": [],
        "confidence_assessment": "insufficient",
    }


def _build_feedback_text(parsed: dict) -> str:
    """将结构化结果转为可读文本（兼容 report_node 的 critic_result['feedback']）"""
    if parsed["approved"]:
        return f"APPROVED — Confidence: {parsed.get('confidence_assessment', 'adequate')}."

    lines = [f"REVISION NEEDED — Confidence: {parsed.get('confidence_assessment', 'weak')}"]

    issues = parsed.get("issues", [])
    if issues:
        lines.append("\nIssues found:")
        for issue in issues:
            lines.append(f"- {issue}")

    suggestions = parsed.get("suggestions", [])
    if suggestions:
        lines.append("\nSuggestions:")
        for s in suggestions:
            lines.append(f"- {s}")

    return "\n".join(lines)
