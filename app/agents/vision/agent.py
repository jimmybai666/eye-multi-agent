"""Vision agent — VLM ReAct 工具调用 + MCP 自动发现模式

流程：
1. 建立 MCP session → list_tools() 动态获取可用工具
2. 工具描述注入 prompt → VLM 看图后输出 JSON 指定调哪些工具
3. 代码解析 JSON → session.call() 执行 → 结果作为 HumanMessage 返回 VLM
4. VLM 综合分析 → 输出 finish JSON → 提取 findings
"""

import asyncio
import json as json_mod
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.state.schema import AgentState
from app.agents.vision.prompt import build_vision_prompt
from app.models.vlm_factory import get_vlm, encode_image
from app.mcp.client import EyeMCPClient, read_image_as_base64
from app.utils.logger import get_logger

logger = get_logger(__name__)

MCP_OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "mcp_images"

MAX_TOOL_ROUNDS = 3
MAX_TOOL_RETRIES = 3
MAX_OVERLAY_PER_ROUND = 2

_TRANSIENT_KEYWORDS = ("timeout", "connection", "reset", "broken pipe", "eof")
_PERSISTENT_KEYWORDS = ("invalid", "unsupported", "not found", "unknown")


async def vision_node(state: AgentState) -> dict:
    """VLM ReAct 循环：观察图片 → 选工具 → 看结果 → 综合判断"""
    logger.info("[Vision] Node started")
    image_path = state.get("image_path")
    vlm = get_vlm()
    mcp_results = {}

    if image_path:
        image_b64 = read_image_as_base64(image_path)
        output_dir = MCP_OUTPUT_ROOT / Path(image_path).stem
        client = EyeMCPClient()

        try:
            async with client.open_session(image_b64, output_dir) as mcp_session:
                tools = await mcp_session.list_tools()
                tools_desc = _format_tools_for_prompt(tools)
                system_prompt = build_vision_prompt(tools_desc)

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=_build_vision_input(state)),
                ]

                messages = await _react_loop(vlm, mcp_session, messages)
                mcp_results = mcp_session.results
        except Exception as e:
            logger.warning(f"[Vision] MCP session 失败，回退到纯 VLM 分析: {e}")
            system_prompt = build_vision_prompt("(No tools available — MCP connection failed)")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=_build_vision_input(state)),
            ]
            try:
                response = await vlm.ainvoke(messages)
                messages.append(response)
            except Exception as api_err:
                logger.error(f"[Vision] Fallback VLM call failed: {api_err}")
    else:
        system_prompt = build_vision_prompt("(No tools available — no image provided)")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=_build_vision_input(state)),
        ]
        try:
            response = await vlm.ainvoke(messages)
            messages.append(response)
        except Exception as api_err:
            logger.error(f"[Vision] No-image VLM call failed: {api_err}")

    final_msg = _get_last_ai_message(messages)
    raw_content = final_msg.content if final_msg else ""

    parsed = _parse_vision_output(raw_content)

    if parsed and parsed.get("action") == "finish":
        image_type = parsed.get("image_type", "other")
        findings = _json_to_findings_text(parsed)
    else:
        image_type = _extract_image_type_regex(raw_content)
        findings = raw_content

    # VLM 输出 finish JSON 时定量数据已包含在内，无需重复追加
    # 仅在 JSON 解析失败（回退到自由文本）时追加 MCP 定量摘要
    if not (parsed and parsed.get("action") == "finish") and mcp_results:
        summary = _format_mcp_summary(mcp_results)
        if summary:
            findings += "\n\n" + summary

    return {
        "messages": [final_msg] if final_msg else [],
        "vision_result": {
            "findings": findings,
            "image_path": image_path,
            "image_type": image_type,
            "mcp_results": mcp_results,
        },
    }


# ── ReAct 循环 ──────────────────────────────────────────


async def _react_loop(vlm, mcp_session, messages: list) -> list:
    """VLM ReAct 循环：VLM 输出 JSON action → 解析执行 → 反馈 → 重复"""
    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        try:
            response = await vlm.ainvoke(messages)
        except Exception as api_err:
            logger.error(f"[Vision] VLM API call failed (round {round_num}): {api_err}")
            break
        messages.append(response)

        parsed = _parse_vision_output(response.content)

        if not parsed or parsed.get("action") == "finish":
            logger.info(f"[Vision] 第 {round_num} 轮: VLM 完成分析")
            break

        tool_names = parsed.get("tool_calls", [])
        if not tool_names:
            logger.info(f"[Vision] 第 {round_num} 轮: 无工具调用，完成")
            break

        # 每轮工具执行前去掉历史中的图片，只保留文本
        _strip_images_from_history(messages)

        logger.info(f"[Vision] 第 {round_num} 轮: 调用工具 {tool_names}")

        overlay_count = 0
        for tool_name in tool_names:
            result = await _execute_tool(mcp_session, tool_name)
            result_text = _format_tool_result(tool_name, result)
            include_overlay = overlay_count < MAX_OVERLAY_PER_ROUND
            content = _build_tool_result_content(tool_name, result_text, result, include_overlay)
            if include_overlay and result.get("overlay_path"):
                overlay_count += 1
            messages.append(HumanMessage(content=content))
            
        if round_num == MAX_TOOL_ROUNDS:
            logger.info("[Vision] 达到最大轮次，请求最终总结")
            messages.append(HumanMessage(
                content='You have reached the maximum number of tool rounds. '
                        'Please provide your final assessment now as a JSON with "action": "finish".',
            ))
            try:
                final_response = await vlm.ainvoke(messages)
                messages.append(final_response)
            except Exception as api_err:
                logger.error(f"[Vision] Final VLM call failed: {api_err}")

    return messages


async def _execute_tool(mcp_session, tool_name: str) -> dict:
    """执行单个 MCP 工具调用，按错误类型决定是否重试"""
    for attempt in range(1, MAX_TOOL_RETRIES + 1):
        result = await mcp_session.call(tool_name)

        if "error" not in result:
            return result

        error_msg = result["error"].lower()

        if any(kw in error_msg for kw in _PERSISTENT_KEYWORDS):
            logger.warning(f"[Vision] {tool_name} 持久性错误，不重试: {result['error']}")
            return result

        if attempt < MAX_TOOL_RETRIES:
            logger.warning(f"[Vision] {tool_name} 第 {attempt} 次失败，重试: {result['error']}")
            await asyncio.sleep(1.0 * attempt)
        else:
            logger.error(f"[Vision] {tool_name} 第 {attempt} 次失败，放弃: {result['error']}")

    return result


# ── JSON 解析 ──────────────────────────────────────────


def _parse_vision_output(content: str) -> dict | None:
    """三层防御解析 VLM 的 JSON 输出

    1. 去掉 markdown 代码块
    2. 正则提取 {...} JSON 对象
    3. 解析失败返回 None（调用方回退到正则提取或安全终止）
    """
    text = content.strip()

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json_mod.loads(json_match.group())
            if isinstance(data, dict) and "action" in data:
                return data
        except (json_mod.JSONDecodeError, ValueError):
            pass

    logger.warning(f"[Vision] JSON 解析失败: {content[:200]}")
    return None


# ── 工具结果格式化 ──────────────────────────────────────────


def _format_tool_result(tool_name: str, result: dict) -> str:
    """将单个 MCP 工具的结果 dict 转为 VLM 可理解的文本"""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = []

    if tool_name == "optic_disc_cup_seg":
        if "cdr" in result:
            lines.append(f"Cup-to-Disc Ratio (CDR): {result['cdr']:.4f}")
        stats = result.get("statistics", {})
        for key in ("optic_disc", "optic_cup"):
            if key in stats:
                lines.append(f"{key} area ratio: {stats[key].get('ratio', 0) * 100:.1f}%")

    elif tool_name == "fovea_locate":
        lines.append(f"Fovea normalized: ({result.get('x', 0):.4f}, {result.get('y', 0):.4f})")
        lines.append(f"Fovea pixel: ({result.get('pixel_x', 0)}, {result.get('pixel_y', 0)})")

    elif tool_name == "vessel_seg":
        stats = result.get("statistics", {})
        for key in ("artery", "vein", "vessel"):
            if key in stats:
                lines.append(f"{key} area ratio: {stats[key].get('ratio', 0) * 100:.2f}%")

    elif tool_name == "lesion_seg":
        stats = result.get("statistics", {})
        label_map = {"MA": "Microaneurysms", "HE": "Hemorrhages", "EX": "Hard exudates", "SE": "Soft exudates"}
        for key in ("MA", "HE", "EX", "SE"):
            if key in stats:
                lines.append(f"{label_map[key]} ({key}): {stats[key].get('ratio', 0) * 100:.3f}%")

    elif tool_name == "oct_segment":
        stats = result.get("statistics", {})
        label_map = {"retina": "Retina", "choroid": "Choroid", "irc": "Intraretinal cyst (IRC)", "mh": "Macular hole (MH)"}
        for key in ("retina", "choroid", "irc", "mh"):
            if key in stats:
                lines.append(f"{label_map[key]}: {stats[key].get('ratio', 0) * 100:.1f}%")

    if not lines:
        lines.append(f"{tool_name} completed (no structured statistics)")

    return "\n".join(lines)


def _build_tool_result_content(tool_name: str, result_text: str, result: dict, include_overlay: bool = True) -> list:
    """构建包含文本 + overlay 图片的多模态 content 列表"""
    parts = [{"type": "text", "text": f"[Tool result: {tool_name}]\n{result_text}"}]

    if not include_overlay:
        return parts

    overlay_path = result.get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        try:
            data_url = _encode_resized(overlay_path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        except Exception as e:
            logger.warning(f"[Vision] Failed to encode overlay for {tool_name}: {e}")

    return parts


def _encode_resized(image_path: str, max_side: int = 768) -> str:
    """将图片缩放到 max_side 以内并编码为 base64 data URL（JPEG 质量 80）"""
    from PIL import Image
    import io
    import base64

    img = Image.open(image_path)
    img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _strip_images_from_history(messages: list) -> None:
    """去掉历史消息中的 image_url，只保留文本部分，减少后续轮次的 payload。"""
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else None
        if not isinstance(content, list):
            continue
        stripped = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                continue
            stripped.append(part)
        msg.content = stripped if stripped else [{"type": "text", "text": "(image omitted)"}]


def _format_tools_for_prompt(tools: list[dict]) -> str:
    """将 MCP list_tools() 结果格式化为 prompt 中的工具描述"""
    lines = []
    for t in tools:
        lines.append(f"- **{t['name']}**: {t['description']}")
    return "\n".join(lines) if lines else "(No tools available)"


# ── 辅助函数 ──────────────────────────────────────────


def _build_vision_input(state: AgentState) -> list:
    """构建 VLM 多模态输入（图片 data URL + 文字上下文）"""
    content_parts = []

    image_path = state.get("image_path")
    if image_path:
        try:
            data_url = encode_image(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
        except FileNotFoundError:
            content_parts.append({"type": "text", "text": "(Image file not found)"})

    text = "Please analyze the following ophthalmic image."
    if state.get("input"):
        text += f"\nClinical context: {state['input']}"
    content_parts.append({"type": "text", "text": text})

    return content_parts


def _get_last_ai_message(messages: list) -> AIMessage | None:
    """从消息列表中获取最后一条 AI 消息"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _json_to_findings_text(parsed: dict) -> str:
    """将 JSON 结构化结果转为可读 findings 文本"""
    lines = []

    lines.append(f"Image type: {parsed.get('image_type', 'unknown')}")

    findings = parsed.get("primary_findings", [])
    if findings:
        lines.append("\nPrimary findings:")
        for f in findings:
            lines.append(f"- {f}")

    quant = parsed.get("quantitative_summary", {})
    if quant:
        lines.append("\nQuantitative summary:")
        for k, v in quant.items():
            lines.append(f"- {k}: {v}")

    severity = parsed.get("severity")
    if severity:
        lines.append(f"\nSeverity: {severity}")

    confidence = parsed.get("confidence")
    if confidence is not None:
        lines.append(f"Confidence: {confidence}")

    diff = parsed.get("differential", [])
    if diff:
        lines.append("\nSuggested differential:")
        for d in diff:
            lines.append(f"- {d}")

    return "\n".join(lines)


def _extract_image_type_regex(findings: str) -> str:
    """正则回退：从 VLM 自由文本中提取图片类型"""
    match = re.search(
        r"(?:image\s*type|IMAGE_TYPE)\s*[:：]\s*(fundus|oct|other)",
        findings,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    logger.warning("[Vision] VLM 未输出 Image type，默认 other")
    return "other"


def _format_mcp_summary(mcp_results: dict) -> str:
    """将 MCP 定量结果转为可读文本，追加到 findings"""
    lines = ["## Quantitative Image Analysis (MCP Tools)"]

    if "optic_disc_cup_seg" in mcp_results:
        r = mcp_results["optic_disc_cup_seg"]
        if "error" not in r:
            lines.append("\n### Optic Disc/Cup Segmentation")
            if "cdr" in r:
                lines.append(f"- Cup-to-Disc Ratio (CDR): {r['cdr']:.4f}")
            stats = r.get("statistics", {})
            for key in ("optic_disc", "optic_cup"):
                if key in stats:
                    ratio = stats[key].get("ratio", 0)
                    lines.append(f"- {key}: {ratio*100:.1f}%")

    if "fovea_locate" in mcp_results:
        r = mcp_results["fovea_locate"]
        if "error" not in r:
            lines.append("\n### Fovea Location")
            lines.append(f"- Normalized: ({r.get('x', 0):.4f}, {r.get('y', 0):.4f})")
            lines.append(f"- Pixel: ({r.get('pixel_x', 0)}, {r.get('pixel_y', 0)})")

    if "vessel_seg" in mcp_results:
        r = mcp_results["vessel_seg"]
        if "error" not in r:
            lines.append("\n### Vessel Segmentation")
            stats = r.get("statistics", {})
            for key in ("artery", "vein", "vessel"):
                if key in stats:
                    ratio = stats[key].get("ratio", 0)
                    lines.append(f"- {key}: {ratio*100:.2f}%")

    if "lesion_seg" in mcp_results:
        r = mcp_results["lesion_seg"]
        if "error" not in r:
            lines.append("\n### Lesion Detection")
            stats = r.get("statistics", {})
            label_map = {"MA": "Microaneurysms", "HE": "Hemorrhages", "EX": "Hard exudates", "SE": "Soft exudates"}
            for key in ("MA", "HE", "EX", "SE"):
                if key in stats:
                    ratio = stats[key].get("ratio", 0)
                    lines.append(f"- {label_map.get(key, key)} ({key}): {ratio*100:.3f}%")

    if "oct_segment" in mcp_results:
        r = mcp_results["oct_segment"]
        if "error" not in r:
            lines.append("\n### OCT Layer Segmentation")
            stats = r.get("statistics", {})
            label_map = {"retina": "Retina", "choroid": "Choroid", "irc": "Intraretinal cyst (IRC)", "mh": "Macular hole (MH)"}
            for key in ("retina", "choroid", "irc", "mh"):
                if key in stats:
                    ratio = stats[key].get("ratio", 0)
                    lines.append(f"- {label_map.get(key, key)}: {ratio*100:.1f}%")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
