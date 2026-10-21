"""Report Agent — 生成最终诊断报告（markdown + PDF）或 QA 回答

在 LangGraph 图中的位置：
  diagnosis 管线：supervisor → [vision + clinical] → critic → ★ report ★ → END
  qa 管线：      supervisor → clinical → ★ report ★ → END

工作方式：
  1. 从 state 中读取所有上游 agent 的输出（vision、clinical、critic、evidence）
  2. 把这些输出组装成一段结构化文本，作为 LLM 的输入
  3. LLM 单次调用，按 prompt 模板生成 6 个 section 的英文 markdown 报告
  4. （仅 diagnosis 模式）把 markdown 报告 + 分割图片 → 交给 pdf_report.py 生成 PDF
  5. 返回 report（markdown 文本）、pdf_path（PDF 路径）、final_diagnosis 等字段

两种模式：
  - diagnosis: 完整 6 section 报告 + PDF（含图片）
  - qa: 轻量 3 section 回答，不生成 PDF
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.state.schema import AgentState
from app.agents.report.prompt import REPORT_SYSTEM_PROMPT, QA_REPORT_SYSTEM_PROMPT
from app.models.llm_factory import get_llm
from app.rag.evidence import EvidencePackage, format_evidence_for_clinical
from app.utils.pdf_report import generate_pdf_report
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def report_node(state: AgentState) -> dict:
    """Report Agent 的 LangGraph 节点函数。

    流程：
    1. 根据 route 选择 prompt（diagnosis 用完整模板，qa 用简洁模板）
    2. 调用 _build_report_input() 把上游各 agent 输出组装成 LLM 输入
    3. LLM 单次调用，生成 markdown 报告
    4. diagnosis 模式额外调用 generate_pdf_report() 生成 PDF
    5. 返回结果写入 state

    写入 state 的字段：
    - messages: LLM 响应消息
    - report: markdown 文本（前端可以直接渲染）
    - pdf_path: PDF 文件路径（仅 diagnosis，qa 时为 None）
    - final_diagnosis: 从 clinical_result 中提取的主诊断（便于前端展示）
    - is_complete: True（标记管线已走完）
    """
    llm = get_llm()
    route = state.get("route", "diagnosis")

    # 根据场景选 prompt
    if route == "qa":
        system_prompt = QA_REPORT_SYSTEM_PROMPT
    else:
        system_prompt = REPORT_SYSTEM_PROMPT

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_report_input(state)),
    ]

    # LLM 调用 — 最多 2 次（首次 + 1 次重试），防止空响应和 API 错误
    report_text = ""
    response = None
    for attempt in range(2):
        try:
            response = await llm.ainvoke(messages)
            if response.content and response.content.strip():
                report_text = response.content
                break
            logger.warning(f"[Report] LLM returned empty (attempt {attempt + 1})")
        except Exception as api_err:
            logger.error(f"[Report] LLM API call failed (attempt {attempt + 1}): {api_err}")
            if attempt < 1:
                await asyncio.sleep(2)
                continue
            report_text = f"# Report Generation Failed\n\nThe report could not be generated due to an API error: {api_err}"

    # diagnosis 模式：额外生成 PDF（包含分割图片）
    pdf_path = None
    if route == "diagnosis":
        try:
            pdf_path = generate_pdf_report(
                markdown_text=report_text,
                original_image=state.get("image_path"),     # 用户上传的原始图片
                vision_result=state.get("vision_result"),    # 包含 mcp_results（overlay 图片路径）
            )
            logger.info(f"[Report] PDF generated: {pdf_path}")
        except Exception as e:
            # PDF 生成失败不影响 markdown 报告返回 — 降级处理
            logger.error(f"[Report] PDF generation failed: {e}")

    return {
        "messages": [response] if response else [],
        "report": report_text,
        "pdf_path": pdf_path,
        "final_diagnosis": _extract_diagnosis(state),
        "is_complete": True,
    }


# ══════════════════════════════════════════════════════════
# 输入构建 — 把上游 agent 输出组装成 LLM 输入文本
# ══════════════════════════════════════════════════════════


def _build_report_input(state: AgentState) -> str:
    """把 state 中所有上游 agent 的输出组装成一段文本，作为 LLM 的输入。

    组装的内容（按顺序）：
    1. 用户原始问题
    2. Vision Agent 的图像分析发现（findings + image_type）
    3. Clinical Agent 的诊断结论（JSON 转可读文本）
    4. Critic Agent 的审核意见（feedback 文本）
    5. 证据摘要（EvidencePackage 格式化）

    LLM 会根据这些输入 + prompt 中的 section 模板，生成结构化的报告。
    """
    parts = [f"Patient query: {state.get('input', '')}"]

    # Vision Agent 输出：图像分析发现
    if state.get("vision_result"):
        vr = state["vision_result"]
        parts.append(f"\n## Vision Agent Findings\n{vr.get('findings', '(no findings)')}")
        parts.append(f"Image type: {vr.get('image_type', 'unknown')}")

    # Clinical Agent 输出：诊断结论
    # diagnosis 字段可能是结构化 dict（正常解析）或 {"raw": 原始文本}（解析失败的兜底）
    if state.get("clinical_result"):
        cr = state["clinical_result"]
        diagnosis = cr.get("diagnosis", "(no diagnosis)")
        if isinstance(diagnosis, dict):
            if "raw" in diagnosis:
                # JSON 解析失败时的兜底格式，直接用原始文本
                formatted = diagnosis["raw"]
            else:
                # 正常的结构化诊断 → 转为可读文本（LLM 读文本比读 JSON 更高效）
                formatted = _format_clinical_diagnosis(diagnosis)
        else:
            formatted = str(diagnosis)
        parts.append(f"\n## Clinical Agent Diagnosis\n{formatted}")

    # Critic Agent 输出：审核意见（可读文本，由 critic 的 _build_feedback_text 生成）
    if state.get("critic_result"):
        parts.append(f"\n## Critic Agent Review\n{state['critic_result']['feedback']}")

    # 证据：Clinical Agent 检索的指南/百科/文献证据
    if state.get("evidence"):
        try:
            package = EvidencePackage(**state["evidence"])
            evidence_text = format_evidence_for_clinical(package)
            parts.append(f"\n## Evidence\n{evidence_text}")
        except Exception as e:
            logger.warning(f"[Report] Failed to parse evidence: {e}")

    return "\n".join(parts)


def _format_clinical_diagnosis(diagnosis: dict) -> str:
    """把 Clinical Agent 输出的结构化诊断 dict 转为可读文本。

    Clinical Agent 正常输出格式（JSON 解析成功时）：
    {
        "primary_diagnosis": "Mild NPDR",
        "confidence": "medium",
        "clinical_reasoning": "Based on...",
        "differential_diagnosis": [
            {"condition": "Hypertensive retinopathy", "likelihood": "low", "reasoning": "..."}
        ],
        "recommended_examinations": ["OCT", "FFA"],
        "missing_information": ["patient age", "blood sugar history"],
        "input_quality": "brief"
    }

    转为可读文本后，LLM 能更好地理解和整合到报告中。
    """
    lines = []

    if "primary_diagnosis" in diagnosis:
        lines.append(f"Primary diagnosis: {diagnosis['primary_diagnosis']}")
    if "confidence" in diagnosis:
        lines.append(f"Confidence: {diagnosis['confidence']}")
    if "clinical_reasoning" in diagnosis:
        lines.append(f"Clinical reasoning: {diagnosis['clinical_reasoning']}")

    # 鉴别诊断列表
    if "differential_diagnosis" in diagnosis:
        dd = diagnosis["differential_diagnosis"]
        if isinstance(dd, list):
            lines.append("Differential diagnoses:")
            for item in dd:
                if isinstance(item, dict):
                    cond = item.get("condition", "Unknown")
                    likelihood = item.get("likelihood", "")
                    reasoning = item.get("reasoning", "")
                    line = f"  - {cond}"
                    if likelihood:
                        line += f" ({likelihood})"
                    if reasoning:
                        line += f": {reasoning}"
                    lines.append(line)
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"Differential: {dd}")

    # 建议检查项目
    if "recommended_examinations" in diagnosis:
        exams = diagnosis["recommended_examinations"]
        if isinstance(exams, list) and exams:
            lines.append("Recommended examinations:")
            for exam in exams:
                lines.append(f"  - {exam}")

    # 缺失信息（用户没提供但诊断需要的）
    if "missing_information" in diagnosis:
        missing = diagnosis["missing_information"]
        if isinstance(missing, list) and missing:
            lines.append("Missing information:")
            for m in missing:
                lines.append(f"  - {m}")

    if "input_quality" in diagnosis:
        lines.append(f"Input quality: {diagnosis['input_quality']}")

    # 兜底：如果 dict 里没有任何已知字段，直接 JSON dump
    if not lines:
        return json.dumps(diagnosis, indent=2, ensure_ascii=False)
    return "\n".join(lines)


def _extract_diagnosis(state: AgentState) -> str | None:
    """从 clinical_result 中提取主诊断文本，方便前端直接展示。

    优先取 primary_diagnosis 字段，如果不存在则取 raw 兜底文本。
    """
    clinical = state.get("clinical_result")
    if not clinical:
        return None
    diagnosis = clinical.get("diagnosis")
    if isinstance(diagnosis, dict):
        return diagnosis.get("primary_diagnosis") or diagnosis.get("raw")
    return diagnosis
