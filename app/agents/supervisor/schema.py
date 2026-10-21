"""Supervisor 的输出模型定义（Pydantic）"""

from typing import Literal
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """场景识别决策输出"""

    route: Literal["qa", "diagnosis", "reject"] = Field(
        description="路由决策：qa（问答）、diagnosis（诊断）、reject（拒绝）"
    )

    image_status: Literal["valid", "invalid", "none"] = Field(
        description="图片状态：valid（有效眼科图片）、invalid（无效图片）、none（无图片）"
    )

    reason: str = Field(
        description="判断理由，简短说明"
    )

