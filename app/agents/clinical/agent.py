"""Clinical Agent — 迭代式证据检索 + 临床推理（ReAct 循环）

工作方式：LLM 分析 → 生成检索关键词 → 检索 → LLM 再分析 → ... → 输出结论
最多 3 轮检索，每轮 LLM 决定"继续查"还是"够了，出结论"。

与 Vision Agent 并行执行，各自独立。
后续由 Critic Agent 汇总比对两者结果。
"""

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.state.schema import AgentState
from app.agents.clinical.prompt import (
    CLINICAL_THINK_PROMPT,
    CLINICAL_QA_THINK_PROMPT,
    CLINICAL_CONCLUDE_PROMPT,
    CLINICAL_QA_CONCLUDE_PROMPT,
)
from app.models.llm_factory import get_llm
from app.rag.evidence import EvidenceRetriever, EvidencePackage, format_evidence_for_clinical
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 最多思考轮数（不含初始检索和最终结论）
MAX_THINK_ROUNDS = 2


async def clinical_node(state: AgentState) -> dict:
    """Clinical Agent 的 LangGraph 节点函数 — ReAct 循环。

    流程：
    1. 用用户原始文本做第一轮检索（初始证据）
    2. 每轮让 LLM 分析已有证据，决定继续查还是出结论
       - "search": LLM 给出新的检索关键词，调 EvidenceRetriever 再查
       - "conclude": 证据够了，跳出循环
    3. 最多 MAX_THINK_ROUNDS 轮后强制结束
    4. 最终用所有累积证据 + 结论 prompt 生成诊断/回答

    写入 state：
    - messages: LLM 最终响应
    - clinical_result: {diagnosis, search_rounds, queries_used}
    - evidence: 合并后的 EvidencePackage 字典
    """
    llm = get_llm()
    retriever = EvidenceRetriever()
    route = state.get("route", "qa")
    is_diagnosis = route == "diagnosis"

    # 根据场景选 prompt
    think_prompt = CLINICAL_THINK_PROMPT if is_diagnosis else CLINICAL_QA_THINK_PROMPT
    conclude_prompt = CLINICAL_CONCLUDE_PROMPT if is_diagnosis else CLINICAL_QA_CONCLUDE_PROMPT

    # 累积所有证据
    all_evidence_items = []
    all_sources = []
    all_logs = []
    all_queries = []

    # ── 第一轮：用用户原始文本做初始检索 ─────────────
    try:
        user_input = state.get("input", "")
        initial = await retriever.retrieve(query=user_input)
        all_evidence_items.extend(initial.evidence)
        all_sources.extend(initial.sources)
        all_logs.extend(initial.retrieval_log)
        all_queries.append(user_input)
        logger.info(f"[Clinical] Round 0: {initial.total_evidence_count} items")
    except Exception as e:
        logger.warning(f"[Clinical] Initial retrieval failed: {e}")

    # ── 迭代思考轮 ───────────────────────────────────
    for round_num in range(1, MAX_THINK_ROUNDS + 1):
        current_pkg = _build_merged_package(state.get("input", ""), all_evidence_items, all_sources, all_logs)
        evidence_text = format_evidence_for_clinical(current_pkg)

        think_messages = [
            SystemMessage(content=think_prompt),
            HumanMessage(content=_build_think_input(state, evidence_text, round_num)),
        ]
        # LLM 分析当前证据，返回 JSON: {"action": "search"|"conclude", "queries": [...]}
        try:
            think_resp = await llm.ainvoke(think_messages)
        except Exception as api_err:
            logger.error(f"[Clinical] Think LLM call failed (round {round_num}): {api_err}")
            break
        # 容错解析：提取 JSON 中的 action 和 queries，解析失败默认 conclude
        decision = _parse_think_output(think_resp.content)

        if decision["action"] == "conclude":
            logger.info(f"[Clinical] Round {round_num}: conclude")
            break

        queries = decision.get("queries", [])
        if not queries:
            logger.info(f"[Clinical] Round {round_num}: no queries, conclude")
            break

        logger.info(f"[Clinical] Round {round_num}: searching {queries}")
        for q in queries:
            try:
                new_ev = await retriever.retrieve(query=q)
                existing = {e.claim for e in all_evidence_items}
                for item in new_ev.evidence:
                    if item.claim not in existing:
                        all_evidence_items.append(item)
                        existing.add(item.claim)
                all_sources.extend(new_ev.sources)
                all_logs.extend(new_ev.retrieval_log)
                all_queries.append(q)
            except Exception as e:
                logger.warning(f"[Clinical] Round {round_num} failed for '{q}': {e}")

    # ── 结论阶段 ─────────────────────────────────────
    final_pkg = _build_merged_package(state.get("input", ""), all_evidence_items, all_sources, all_logs)
    final_text = format_evidence_for_clinical(final_pkg)

    conclude_messages = [
        SystemMessage(content=conclude_prompt),
        HumanMessage(content=_build_conclude_input(state, final_text)),
    ]

    response = None
    for attempt in range(2):
        try:
            response = await llm.ainvoke(conclude_messages)
            if response.content and response.content.strip():
                break
            logger.warning(f"[Clinical] Conclude returned empty (attempt {attempt + 1}), retrying...")
        except Exception as api_err:
            logger.error(f"[Clinical] Conclude LLM call failed (attempt {attempt + 1}): {api_err}")
            if attempt < 1:
                await asyncio.sleep(2)
                continue

    # 解析结论 JSON（容错：解析失败则保留原始文本）
    conclusion = _parse_conclusion_output(response.content if response else "", is_diagnosis)

    result = {
        "messages": [response] if response else [],
        "clinical_result": {
            "diagnosis": conclusion,
            "search_rounds": len(all_queries),
            "queries_used": all_queries,
        },
        "evidence": final_pkg.model_dump(),
    }

    # QA 模式：Clinical 直接输出最终回答，跳过 Report Agent
    if not is_diagnosis:
        result["report"] = _build_qa_report(conclusion, final_pkg)
        result["final_diagnosis"] = None
        result["pdf_path"] = None
        result["is_complete"] = True

    return result


# ── 辅助函数 ──────────────────────────────────────────


def _build_merged_package(query, items, sources, logs):
    """多轮证据合并成一个 EvidencePackage。"""
    sorted_items = sorted(items, key=lambda e: e.level_rank)
    return EvidencePackage(
        query=query,
        evidence=sorted_items,
        sources=sources,
        retrieval_status="completed" if items else "insufficient_evidence",
        retrieval_log=logs,
    )


def _build_qa_report(conclusion: dict, evidence_pkg: EvidencePackage) -> str:
    """QA 模式下直接从 Clinical 输出构建 markdown 回答，跳过 Report Agent。"""
    parts = []

    if isinstance(conclusion, dict):
        raw = conclusion.get("raw")
        if raw:
            parts.append(raw)
        else:
            answer = conclusion.get("answer") or conclusion.get("primary_diagnosis")
            if answer:
                parts.append(f"## Answer\n{answer}")
            if conclusion.get("clinical_reasoning"):
                parts.append(conclusion["clinical_reasoning"])
    else:
        parts.append(str(conclusion))

    if evidence_pkg.evidence:
        parts.append("\n## Evidence Sources")
        for item in evidence_pkg.evidence:
            parts.append(f"- **{item.source}** ({item.evidence_level}): {item.claim}")

    return "\n\n".join(parts) if parts else "(No answer generated)"


def _build_think_input(state, evidence_text, round_num):
    """构建思考轮的 HumanMessage。"""
    parts = [f"User input: {state.get('input', '')}"]
    parts.append(f"\nCurrent round: {round_num}/{MAX_THINK_ROUNDS}")
    if evidence_text:
        parts.append(f"\n## Evidence collected so far\n{evidence_text}")
    else:
        parts.append("\n## Evidence collected so far\n(No evidence yet)")
    if round_num == MAX_THINK_ROUNDS:
        parts.append("\nThis is the LAST round. You must output 'conclude'.")
    return "\n".join(parts)


def _build_conclude_input(state, evidence_text):
    """构建结论轮的 HumanMessage。"""
    parts = [f"User input: {state.get('input', '')}"]
    if evidence_text:
        parts.append(f"\n## All evidence collected\n{evidence_text}")

    critic = state.get("critic_result")
    if critic and not critic.get("approved", True):
        parts.append("\n## Critic feedback (address these issues in your revised conclusion)")
        for issue in critic.get("issues", []):
            parts.append(f"- {issue}")
        for suggestion in critic.get("suggestions", []):
            parts.append(f"- Suggestion: {suggestion}")

    return "\n".join(parts)


def _parse_think_output(content):
    """解析 LLM 思考轮的 JSON 输出（容错）。"""
    content = content.strip()
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "action" in result:
                return result
        except json.JSONDecodeError:
            pass
    logger.warning(f"[Clinical] Parse failed, defaulting to conclude: {content[:200]}")
    return {"action": "conclude", "reasoning": "Parse failed"}


def _parse_conclusion_output(content: str, is_diagnosis: bool) -> dict:
    """解析结论轮的 JSON 输出（容错）。

    解析成功 → 返回结构化 dict
    解析失败 → 返回 {"raw": 原始文本} 保证下游不会崩
    """
    content = content.strip()
    if "```" in content:
        match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if is_diagnosis and "primary_diagnosis" in result:
                return result
            if not is_diagnosis and "answer" in result:
                return result
            return result
        except json.JSONDecodeError:
            pass
    logger.warning(f"[Clinical] Conclusion JSON parse failed, using raw text")
    return {"raw": content}
