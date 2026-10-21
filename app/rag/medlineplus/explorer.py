"""MedlinePlus Explorer - 通过 NLM Search API 发现健康主题

这是 MedlinePlus RAG 的前置阶段，负责：
1. 调用 NLM Health Topics Search API 搜索相关主题,解决搜索关键词与 MedlinePlus 的页面名字不一致的问题：原理:全文检索,按相关性排序
2. 解析 XML 响应提取标题、URL、摘要
3. 提供结构化数据给 Agent 进行精选

"""

import re
import requests
from typing import List
from dataclasses import dataclass
from xml.etree import ElementTree

from app.utils.logger import get_logger

logger = get_logger(__name__)

# NLM 公共领域声明
ATTRIBUTION = "Courtesy of MedlinePlus from the National Library of Medicine"


@dataclass
class TopicInfo:
    """健康主题信息"""
    name: str               # URL slug（如 "glaucoma"）
    title: str              # 显示标题（如 "Glaucoma"）
    url: str                # 完整 URL
    summary_preview: str    # 摘要预览（前 200 字符）


class MedlinePlusExplorer:
    """MedlinePlus 探索器 - 通过 NLM API 发现健康主题"""

    SEARCH_API = "https://wsearch.nlm.nih.gov/ws/query"

    def search_topics(
        self,
        keyword: str,
        max_results: int = 10
    ) -> List[TopicInfo]:
        """
        搜索健康主题

        Args:
            keyword: 搜索关键词（如 "glaucoma", "eye diseases"）
            max_results: 最多返回数量

        Returns:
            TopicInfo 列表
        """
        try:
            logger.info(f"[MedlinePlus] 搜索: {keyword}")
            response = requests.get(
                self.SEARCH_API,
                params={
                    "db": "healthTopics",
                    "term": keyword,
                    "retmax": max_results,
                },
                timeout=30,
            )

            if response.status_code != 200:
                logger.error(
                    f"[MedlinePlus] API 请求失败: {response.status_code}"
                )
                return []

            topics = self._parse_xml_response(response.text)
            logger.info(
                f"[MedlinePlus] 搜索 '{keyword}' 找到 {len(topics)} 个主题"
            )
            return topics

        except Exception as e:
            logger.error(f"[MedlinePlus] 搜索异常: {e}")
            return []

    def _parse_xml_response(self, xml_text: str) -> List[TopicInfo]:
        """
        解析 NLM Search API 的 XML 响应

        Args:
            xml_text: XML 响应文本

        Returns:
            TopicInfo 列表
        """
        topics = []

        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError as e:
            logger.error(f"[MedlinePlus] XML 解析失败: {e}")
            return []

        # 遍历 document 元素
        for doc in root.iter("document"):
            url = doc.get("url", "")

            # 只保留 medlineplus.gov 的健康主题页面
            if "medlineplus.gov/" not in url:
                continue

            title = ""
            summary = ""

            for content in doc.iter("content"):
                name = content.get("name", "")
                value = content.text or ""

                if name == "title":
                    # 去掉 XML 中的高亮标签
                    title = re.sub(r'<[^>]+>', '', value).strip()
                elif name == "FullSummary":
                    # 去掉 HTML 标签
                    summary = re.sub(r'<[^>]+>', '', value).strip()

            if not title or not url:
                continue

            # 从 URL 提取 slug
            # https://medlineplus.gov/glaucoma.html → glaucoma
            slug = self._url_to_slug(url)
            if not slug:
                continue

            topics.append(TopicInfo(
                name=slug,
                title=title,
                url=url,
                summary_preview=summary[:200] if summary else "",
            ))

        return topics

    @staticmethod
    def _url_to_slug(url: str) -> str:
        """
        从 MedlinePlus URL 提取 slug

        https://medlineplus.gov/glaucoma.html → glaucoma
        https://medlineplus.gov/diabeticeyeproblems.html → diabeticeyeproblems
        """
        match = re.search(r'medlineplus\.gov/([a-zA-Z0-9\-]+)\.html', url)
        if match:
            return match.group(1).lower()
        return ""


def format_topics_for_agent(topics: List[TopicInfo]) -> str:
    """
    将主题列表格式化为适合 Agent 阅读的文本

    Args:
        topics: 主题信息列表

    Returns:
        格式化的文本
    """
    if not topics:
        return "未找到相关健康主题。"

    lines = ["可用的 MedlinePlus 健康主题：", ""]

    for i, topic in enumerate(topics, 1):
        lines.append(f"{i}. {topic.title}")
        lines.append(f"   URL: {topic.url}")
        if topic.summary_preview:
            lines.append(f"   摘要: {topic.summary_preview}...")
        lines.append("")

    return "\n".join(lines)
