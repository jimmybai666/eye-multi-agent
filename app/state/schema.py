"""AgentState 定义 — 多 Agent 系统的共享状态结构

所有 agent 节点读写的都是同一个 AgentState 字典。
LangGraph 在每个节点执行后，把节点返回的 dict 合并进这个 state。
"""

from typing import Annotated, Literal, TypedDict, Sequence
from operator import add

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """多 Agent 图的共享状态，所有节点通过这个字典交换数据。

    字段分为 5 组：消息历史、用户输入、路由控制、Agent 输出、迭代控制。
    """

    # ── 消息历史 ───────────────────────────────────────
    # Annotated[..., add] 表示 LangGraph 合并时用列表拼接（追加），而不是覆盖
    # 每个 agent 返回 {"messages": [response]}，会自动追加到这个列表里
    messages: Annotated[Sequence[BaseMessage], add]

    # ── 用户输入 ───────────────────────────────────────
    input: str               # 用户的文本输入（问题或症状描述）
    image_path: str | None   # 用户上传的眼科图片路径（没有图片时为 None）

    # ── 路由控制（由 supervisor 场景识别填写）──────────
    route: Literal["diagnosis", "qa", "reject"] | None  # 场景类型，决定走哪条管线
    image_status: Literal["valid", "invalid", "none"] | None  # 图片是否为有效眼科图片
    reason: str | None       # supervisor 的判断理由（reject 时告诉用户为什么被拒绝）
    is_complete: bool        # 管线是否已走完（report_node 设为 True）

    # ── Agent 输出（每个 agent 写自己的字段）──────────
    vision_result: dict | None     # Vision Agent 的图像分析结果（findings, image_path）
    clinical_result: dict | None   # Clinical Agent 的诊断结果（diagnosis, based_on_vision）
    critic_result: dict | None     # Critic Agent 的审核结果（feedback, approved）

    # ── 证据（由 Clinical Agent 调用 EvidenceRetriever 填写）──
    evidence: dict | None    # EvidencePackage 序列化后的字典，包含指南/百科/文献证据

    # ── 最终输出 ───────────────────────────────────────
    final_diagnosis: str | None  # 最终诊断结论（从 clinical_result 中提取）
    report: str | None           # Report Agent 生成的完整报告
    pdf_path: str | None         # PDF 报告文件路径（仅 diagnosis 模式）

    # ── 迭代控制（critic 循环用）──────────────────────
    iteration_count: int     # 当前 critic 迭代次数（每次 critic_node 执行后 +1）
    max_iterations: int      # 最大迭代上限（默认 3，超过后强制进入 report）
