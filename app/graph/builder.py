"""构建 LangGraph 状态图 — 多 Agent 诊断管线

三条场景管线：
  diagnosis: supervisor → [vision + clinical]（并行）→ critic → (report | clinical 循环) → report → END
  qa:        supervisor → clinical → END（直接结束，不经 report）
  reject:    supervisor → END
"""

from langgraph.graph import StateGraph, END

from app.state.schema import AgentState
from app.graph.nodes import register_nodes
from app.graph.edges import (
    route_after_supervisor,
    route_after_clinical,
    route_after_critic,
)


def build_graph() -> StateGraph:
    """构建并编译多 Agent 状态图。

    diagnosis 场景的并行设计：
    - supervisor 之后同时进入 vision 和 clinical（并行执行）
    - vision 和 clinical 都完成后，汇聚到 critic
    - critic 比对两者结果，判断是否矛盾
    """

    workflow = StateGraph(AgentState)

    # 注册 5 个 agent 节点
    register_nodes(workflow)

    # 入口：所有请求先经过 supervisor 做场景识别
    workflow.set_entry_point("supervisor")

    # ── 边 1：supervisor 之后 ──────────────────────────
    # diagnosis → ["vision", "clinical"] 并行分发
    # qa → "clinical" 单独走
    # reject → END
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "vision": "vision",
            "clinical": "clinical",
            "end": END,
        },
    )

    # ── 边 2：vision → critic（diagnosis 场景）─────────
    # vision 完成后等 clinical 也完成，两者汇聚到 critic
    workflow.add_edge("vision", "critic")

    # ── 边 3：clinical 之后 ───────────────────────────
    # diagnosis 场景 → critic（等 vision 也到达后触发）
    # qa 场景 → END（直接结束，Clinical 输出即最终回答）
    workflow.add_conditional_edges(
        "clinical",
        route_after_clinical,
        {
            "critic": "critic",
            "end": END,
        },
    )

    # ── 边 4：critic 之后 ────────────────────────────
    # approved=True → report
    # iteration_count >= max_iterations → report（强制）
    # 否则 → clinical（打回重新推理）
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "report": "report",
            "clinical": "clinical",
        },
    )

    # ── 边 5：report → END ───────────────────────────
    workflow.add_edge("report", END)

    return workflow.compile()
