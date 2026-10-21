"""Register all agent nodes to the graph with exception boundaries."""

import functools

from langgraph.graph import StateGraph

from app.agents.supervisor.agent import supervisor_node
from app.agents.vision.agent import vision_node
from app.agents.clinical.agent import clinical_node
from app.agents.critic.agent import critic_node
from app.agents.report.agent import report_node
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 节点级兜底映射表，作用是当图里某个节点崩溃时，不让整条 pipeline 直接抛异常挂掉，而是用一段预定义的降级状态覆盖 State，让流程能继续跑下去或者安全结束
_NODE_FALLBACKS = {
    "supervisor": lambda state: {
        "route": "reject",
        "image_status": "unknown",
        "reason": "Supervisor node crashed — pipeline terminated safely",
    },
    "vision": lambda state: {
        "messages": [],
        "vision_result": {
            "findings": "(Vision analysis unavailable due to internal error)",
            "image_path": state.get("image_path"),
            "image_type": "unknown",
            "mcp_results": {},
        },
    },
    "clinical": lambda state: {
        "messages": [],
        "clinical_result": {
            "diagnosis": {"raw": "(Clinical analysis unavailable due to internal error)"},
            "search_rounds": 0,
            "queries_used": [],
        },
        "evidence": None,
    },
    "critic": lambda state: {
        "messages": [],
        "critic_result": {
            "approved": True,
            "issues": [],
            "suggestions": [],
            "confidence_assessment": "insufficient",
            "feedback": "Critic node crashed — auto-approved to avoid blocking pipeline.",
        },
        "iteration_count": state.get("iteration_count", 0) + 1,
    },
    "report": lambda state: {
        "messages": [],
        "report": "# Report Generation Failed\n\nAn internal error occurred during report generation.",
        "pdf_path": None,
        "final_diagnosis": None,
        "is_complete": True,
    },
}

# 节点异常包装器（装饰器），作用是把原本会直接抛异常中断整张图的节点函数，包一层 try-except，出错时打日志并返回上一段代码里定义的兜底状态，没定义兜底就继续往外抛
def _safe_node(node_name: str, node_func):
    """Wrap a node function with an exception boundary that returns a fallback state."""
    @functools.wraps(node_func)
    async def wrapper(state):
        try:
            return await node_func(state)
        except Exception as e:
            logger.error(f"[{node_name}] Node crashed: {e}", exc_info=True)
            fallback = _NODE_FALLBACKS.get(node_name)
            if fallback:
                return fallback(state)
            raise
    return wrapper


def register_nodes(workflow: StateGraph) -> None:
    """Register all agent nodes into the workflow with exception boundaries."""
    workflow.add_node("supervisor", _safe_node("supervisor", supervisor_node))
    workflow.add_node("vision", _safe_node("vision", vision_node))
    workflow.add_node("clinical", _safe_node("clinical", clinical_node))
    workflow.add_node("critic", _safe_node("critic", critic_node))
    workflow.add_node("report", _safe_node("report", report_node))
