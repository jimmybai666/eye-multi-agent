"""条件边函数 — 控制图的流转方向

每个函数读取当前 state，返回一个字符串键（或列表），
LangGraph 根据这个键在 add_conditional_edges 的映射表里找到下一个节点。
返回列表时，多个目标节点并行执行。
"""

from typing import Union

from app.state.schema import AgentState


def route_after_supervisor(state: AgentState) -> Union[str, list[str]]:
    """supervisor 之后的路由。

    读取 state["route"]（由 supervisor_node 的场景识别填写）：
      "diagnosis" → ["vision", "clinical"]  并行：vision 分析图片 + clinical 做推理
      "qa"        → "clinical"              只做临床推理
      其他/None   → "end"                   拒绝或异常，结束管线
    """
    route = state.get("route", "reject")
    if route == "diagnosis":
        return ["vision", "clinical"]
    elif route == "qa":
        return "clinical"
    else:
        return "end"


def route_after_clinical(state: AgentState) -> str:
    """clinical 之后的路由。

      qa 场景 → "end"  QA 直接结束，不需要 Report 二次包装
      diagnosis 场景 → "critic"  但只有 vision 也完成后才会触发
                       （因为 critic 需要等 vision 和 clinical 两条边都到达）
    """
    route = state.get("route")
    if route == "diagnosis":
        return "critic"
    return "end"


def route_after_critic(state: AgentState) -> str:
    """critic 之后的路由（diagnosis 场景专用）。

    三种情况：
    1. critic 审核通过（approved=True）→ "report" 生成报告
    2. 迭代次数达到上限（iteration_count >= max_iterations）→ "report" 强制出报告
    3. 否则 → "clinical" 打回重新推理（形成 clinical ↔ critic 循环）
    """
    critic = state.get("critic_result", {})
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    if critic.get("approved", False):
        return "report"
    if iteration >= max_iter:
        return "report"
    return "clinical"
