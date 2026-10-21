"""EyeWiki Explorer - 探索和发现疾病相关文章

这是 RAG 的前置阶段，负责：
1. 检测输入类型（分类 vs 具体页面）
2. 获取文章列表
3. 提供结构化数据给 Agent 进行精选
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Literal
from dataclasses import dataclass

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ArticleInfo:
    """文章信息"""
    name: str           # URL 路径名
    title: str          # 显示标题
    url: str            # 完整 URL
    source_type: Literal["page", "category"]  # 来源类型


class EyeWikiExplorer:
    """EyeWiki 探索器 - 发现相关文章"""

    def __init__(self, base_url: str = "https://eyewiki.aao.org"):
        self.base_url = base_url

    def explore(self, name: str, max_articles: int = 50) -> Dict:
        """
        探索一个疾病/主题，返回相关文章列表

        Args:
            name: 疾病名称（可以是分类或具体页面）
            max_articles: 如果是分类，最多返回多少篇文章（0=不限制）

        Returns:
            {
                "input": 输入名称,
                "type": "page" | "category" | "none",
                "articles": [ArticleInfo, ...],
                "total_found": 总数
            }
        """
        input_type = self._detect_type(name)
        result = {
            "input": name,
            "type": input_type,
            "articles": [],
            "total_found": 0
        }

        if input_type == "page":
            # 具体页面，直接返回
            article = ArticleInfo(
                name=name,
                title=self._get_page_title(name) or name,
                url=f"{self.base_url}/{name}",
                source_type="page"
            )
            result["articles"] = [article]
            result["total_found"] = 1

        elif input_type =="category":
            # 分类页面，获取所有文章
            articles = self._get_category_articles(name, max_articles)
            result["articles"] = articles
            result["total_found"] = len(articles)

        logger.info(
            f"[Explorer] '{name}' 检测为 {input_type}，"
            f"找到 {result['total_found']} 篇文章"
        )

        return result

    def _detect_type(self, name: str) -> Literal["page", "category", "none"]:
        """
        检测输入类型（移除了 both 类型）

        EyeWiki 结构特点：
        - 顶级分类（如 Glaucoma）只存在 Category 页面
        - 具体文章（如 Diabetic_Retinopathy）只存在单页

        Returns:
            - "page": 只存在单页（具体文章）→ 可以直接做 RAG
            - "category": 只存在分类 → 需要继续探索
            - "none": 都不存在
        """
        # 优先检查 page（page 优先级更高）
        page_exists = self._check_page_exists(name)
        if page_exists:
            return "page"

        # 再检查分类
        category_exists = self._check_category_exists(name)
        if category_exists:
            return "category"

        return "none"

    def _check_page_exists(self, name: str) -> bool:
        """检查单页是否存在"""
        try:
            url = f"{self.base_url}/{name}"
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[Explorer] 检查页面失败: {e}")
            return False

    def _check_category_exists(self, name: str) -> bool:
        """检查分类页面是否存在"""
        try:
            url = f"{self.base_url}/Category:{name}"
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"[Explorer] 检查分类失败: {e}")
            return False

    def _get_page_title(self, name: str) -> str:
        """获取页面标题"""
        try:
            url = f"{self.base_url}/{name}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                title_tag = soup.find("h1", {"id": "firstHeading"})
                if title_tag:
                    return title_tag.get_text(strip=True)
        except Exception:
            pass
        return name

    def _get_category_articles(
        self,
        category: str,
        max_articles: int = 50
    ) -> List[ArticleInfo]:
        """
        获取分类下的所有文章

        Args:
            category: 分类名称
            max_articles: 最多获取数量（0=不限制）
        """
        url = f"{self.base_url}/Category:{category}"
        articles = []

        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return articles

            soup = BeautifulSoup(response.text, "html.parser")
            category_div = soup.find("div", {"id": "mw-pages"})

            if category_div:
                links = category_div.find_all("a", href=True)

                for link in links:
                    href = link.get("href", "")

                    # 过滤特殊页面
                    if href.startswith("/") and not any(x in href for x in [
                        "Category:", "Template:", "Special:", "Help:",
                        "File:", "MediaWiki:", "Talk:", "User:"
                    ]):
                        article_name = href.lstrip("/")
                        article_title = link.get_text(strip=True)

                        if article_name:
                            articles.append(ArticleInfo(
                                name=article_name,
                                title=article_title,
                                url=f"{self.base_url}/{article_name}",
                                source_type="category"
                            ))

                            # 如果设置了限制且达到限制，停止
                            if max_articles > 0 and len(articles) >= max_articles:
                                break

        except Exception as e:
            logger.error(f"[Explorer] 获取分类文章失败: {e}")

        return articles


def format_articles_for_agent(articles: List[ArticleInfo]) -> str:
    """
    将文章列表格式化为适合 Agent 阅读的文本

    Args:
        articles: 文章信息列表

    Returns:
        格式化的文本，供 Agent 分析
    """
    if not articles:
        return "未找到相关文章。"

    lines = ["可用的文章列表：", ""]

    for i, article in enumerate(articles, 1):
        lines.append(f"{i}. {article.title}")
        lines.append(f"   路径: {article.name}")
        lines.append(f"   URL: {article.url}")
        lines.append("")

    return "\n".join(lines)


def extract_article_titles(articles: List[ArticleInfo]) -> List[str]:
    """
    提取文章标题（清理后的，供 Agent 阅读）

    转换规则：
    - Primary_Open-Angle_Glaucoma → Primary Open-Angle Glaucoma
    - Home_Tonometry → Home Tonometry
    - 保留连字符内的大小写（Open-Angle 保持不变）

    Args:
        articles: 文章信息列表

    Returns:
        清理后的标题列表
    """
    titles = []
    for article in articles:
        # 使用 title 字段（已经是清理过的）
        # 如果 title 不存在或为空，从 name 转换
        if article.title:
            titles.append(article.title)
        else:
            # 从 name 转换：用空格替换下划线
            clean_title = article.name.replace("_", " ")
            titles.append(clean_title)

    return titles


def create_agent_prompt_for_selection(articles: List[ArticleInfo], user_intent: str = "") -> str:
    """
    为 Agent 创建选择提示（只包含标题）

    Args:
        articles: 文章列表
        user_intent: 用户意图描述

    Returns:
        格式化的 Agent prompt
    """
    titles = extract_article_titles(articles)

    prompt_lines = []

    if user_intent:
        prompt_lines.append(f"用户意图: {user_intent}")
        prompt_lines.append("")

    prompt_lines.append(f"从以下 {len(titles)} 篇文章中选择最相关的（返回标题）:")
    prompt_lines.append("")

    for i, title in enumerate(titles, 1):
        prompt_lines.append(f"{i}. {title}")

    return "\n".join(prompt_lines)
