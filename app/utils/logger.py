"""日志工具 - 使用 loguru 实现的日志记录器。"""

import sys
from loguru import logger

# 移除默认处理器
logger.remove()

# 添加控制台处理器（带格式化）
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# 添加文件处理器（持久化日志）
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="1 day",      # 每天轮转
    retention="30 days",   # 保留30天
    level="DEBUG",         # 文件记录 DEBUG 及以上级别
)


def get_logger(name: str = __name__):
    """获取命名的日志记录器实例。

    Args:
        name: 日志记录器名称（通常传入 __name__）

    Returns:
        绑定了名称的 logger 实例

    Example:
        logger = get_logger(__name__)
        logger.info("这是一条日志")
    """
    return logger.bind(name=name)
