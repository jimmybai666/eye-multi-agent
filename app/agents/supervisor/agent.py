"""Supervisor Agent — scene recognition (routing).

classify user input into qa / diagnosis / reject
using VLM (text + image). All downstream orchestration is handled by
LangGraph conditional edges, not by the Supervisor.
"""

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.supervisor.prompt import SUPERVISOR_SCENE_RECOGNITION_PROMPT
from app.state.schema import AgentState
from app.models.vlm_factory import get_vlm, encode_image
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def supervisor_node(state: AgentState) -> dict:
    """LangGraph node: VLM 场景识别，决定走 diagnosis / qa / reject 管线"""
    vlm = get_vlm()

    user_input = state.get("input", "")
    image_path = state.get("image_path")

    logger.info("[Supervisor] Running VLM scene recognition...")

    content_parts = [{"type": "text", "text": f"用户输入文字：{user_input}"}]

    if image_path:
        try:
            data_url = encode_image(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        except FileNotFoundError as e:
            logger.error(f"[Supervisor] Image not found: {e}")
            content_parts.append({"type": "text", "text": "（用户未上传图片）"})
    else:
        content_parts.append({"type": "text", "text": "（用户未上传图片）"})

    messages = [
        SystemMessage(content=SUPERVISOR_SCENE_RECOGNITION_PROMPT),
        HumanMessage(content=content_parts),
    ]

    max_attempts = 2
    last_error = None

    for attempt in range(max_attempts):
        try:
            try:
                response = await vlm.ainvoke(messages)
            except Exception as api_err:
                logger.error(f"[Supervisor] VLM API call failed (attempt {attempt + 1}): {api_err}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            raw_content = response.content.strip()

            result = _robust_json_parse(raw_content)

            if not all(k in result for k in ["route", "image_status", "reason"]):
                raise ValueError(f"Missing required fields, got: {list(result.keys())}")
            if result["route"] not in ["qa", "diagnosis", "reject"]:
                raise ValueError(f"Invalid route: {result['route']}")
            if result["image_status"] not in ["valid", "invalid", "none"]:
                raise ValueError(f"Invalid image_status: {result['image_status']}")

            logger.info(
                f"[Supervisor] Scene recognition OK (attempt {attempt + 1}): "
                f"route={result['route']}, image_status={result['image_status']}"
            )
            return {
                "route": result["route"],
                "image_status": result["image_status"],
                "reason": result["reason"],
            }

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                f"[Supervisor] Attempt {attempt + 1} parse failed: {e}, "
                f"response: {raw_content[:200]}..."
            )

            if attempt < max_attempts - 1:
                correction_msg = HumanMessage(content=(
                    f"你的输出格式有误，无法解析。\n\n"
                    f"错误信息：{e}\n你的输出：{raw_content}\n\n"
                    f"请严格按照以下 JSON 格式重新输出，不要添加任何其他内容（不要用 markdown 包裹）：\n\n"
                    f'{{"route": "qa", "image_status": "none", "reason": "简短理由"}}\n\n'
                    f"route 只能是：qa、diagnosis、reject\n"
                    f"image_status 只能是：valid、invalid、none"
                ))
                messages.append(response)
                messages.append(correction_msg)

    logger.error(f"[Supervisor] All {max_attempts} attempts failed, last error: {last_error}")
    return {
        "route": "reject",
        "image_status": "unknown",
        "reason": "System error, JSON parse failed",
    }


def _robust_json_parse(content: str) -> dict:
    """Three-layer JSON defense: strip markdown → regex extract → json.loads."""
    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:].strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        content = match.group()

    return json.loads(content)
