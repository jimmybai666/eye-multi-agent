"""State 工具函数 — 创建初始状态"""

from app.state.schema import AgentState


def create_initial_state(input_text: str, image_path: str | None = None) -> AgentState:
    """为一次新的诊断请求创建初始 AgentState。

    所有 agent 输出字段初始化为 None，等各节点执行时填入。
    route=None 表示还没经过 supervisor 场景识别。

    Args:
        input_text: 用户的文本输入
        image_path: 眼科图片路径（可选）

    Returns:
        填好初始值的 AgentState 字典
    """
    return AgentState(
        messages=[],              # 空消息列表，agent 执行时追加
        input=input_text,         # 用户原始文本
        image_path=image_path,    # 图片路径（无图片时为 None）
        route=None,               # 待 supervisor 填写
        image_status=None,        # 待 supervisor 填写
        reason=None,              # 待 supervisor 填写
        is_complete=False,        # report_node 执行后设为 True
        vision_result=None,       # 待 vision_node 填写
        clinical_result=None,     # 待 clinical_node 填写
        critic_result=None,       # 待 critic_node 填写
        evidence=None,            # 待 clinical_node 的证据检索填写
        final_diagnosis=None,     # 待 report_node 填写
        report=None,              # 待 report_node 填写
        pdf_path=None,            # 待 report_node 填写（仅 diagnosis）
        iteration_count=0,        # critic 迭代计数器，从 0 开始
        max_iterations=3,         # 最多允许 critic 循环 3 次
    )
