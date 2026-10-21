"""PDF 报告生成工具 — 专业医疗诊断报告排版

整体流程：
  1. report_node 调用 LLM 生成 markdown 文本（6 个 section 的诊断报告）
  2. 本模块接收 markdown 文本 + 图片路径
  3. 用 reportlab 的 platypus 排版引擎渲染成专业医疗报告 PDF
  4. 输出 A4 尺寸的 PDF 文件到 outputs/reports/

设计思路：
  - 页眉：系统名称 + 蓝色分割线，每页都有
  - 页脚：页码 + 生成时间 + 免责声明，每页都有
  - 患者/图像信息栏：浅蓝色背景的摘要卡片
  - 图片区域：带边框和浅灰背景的网格
  - Final Diagnosis 高亮：橙色边框的重点框
  - Section 之间有分割线，视觉层次清晰
"""

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)

# PDF 输出目录
REPORT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "reports"

# ── 颜色常量 ─────────────────────────────────────────────
# 统一管理所有颜色，方便后续调整主题
COLOR_PRIMARY = colors.HexColor("#1a5276")      # 深蓝 — 标题、页眉
COLOR_SECONDARY = colors.HexColor("#2c81b9")    # 中蓝 — 章节标题
COLOR_ACCENT = colors.HexColor("#e67e22")        # 橙色 — Final Diagnosis 高亮
COLOR_LIGHT_BG = colors.HexColor("#eaf2f8")      # 浅蓝背景 — 信息卡片
COLOR_LIGHT_GRAY = colors.HexColor("#f5f5f5")    # 浅灰背景 — 图片区域
COLOR_BORDER = colors.HexColor("#d5d8dc")        # 边框灰
COLOR_TEXT = colors.HexColor("#2c3e50")          # 正文深灰
COLOR_MUTED = colors.HexColor("#7f8c8d")         # 次要文字灰

# MCP 工具的优先级顺序 — 决定 overlay 图片在 PDF 中的排列顺序
OVERLAY_PRIORITY = [
    "optic_disc_cup_seg",
    "lesion_seg",
    "vessel_seg",
    "oct_segment",
    "fovea_locate",
]

# 工具名 → PDF 中显示的图片标签
TOOL_LABELS = {
    "optic_disc_cup_seg": "Optic Disc / Cup Segmentation",
    "lesion_seg": "Lesion Segmentation",
    "vessel_seg": "Vessel Segmentation",
    "oct_segment": "OCT Layer Segmentation",
    "fovea_locate": "Fovea Location",
}

# 单张图片在 PDF 中的最大尺寸
IMG_WIDTH = 7 * cm
IMG_HEIGHT = 5.5 * cm

# 页面尺寸
PAGE_W, PAGE_H = A4
MARGIN_LEFT = 2.2 * cm
MARGIN_RIGHT = 2.2 * cm
CONTENT_WIDTH = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT


# ══════════════════════════════════════════════════════════
# 样式定义
# ══════════════════════════════════════════════════════════


def _build_styles():
    """构建 PDF 中各种文本元素的样式。"""
    base = getSampleStyleSheet()
    styles = {
        # 报告大标题：20pt 深蓝，居中，加粗
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"],
            fontSize=20, textColor=COLOR_PRIMARY,
            fontName="Helvetica-Bold",
            spaceAfter=2, alignment=TA_CENTER,
        ),
        # 副标题：10pt 灰色，居中
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=10, textColor=COLOR_MUTED, alignment=TA_CENTER,
            spaceAfter=6,
        ),
        # 章节标题：13pt 中蓝，加粗，左侧有色块效果（通过 Table 实现）
        "heading": ParagraphStyle(
            "ReportHeading", parent=base["Heading2"],
            fontSize=13, textColor=COLOR_SECONDARY,
            fontName="Helvetica-Bold",
            spaceBefore=10, spaceAfter=6,
            borderWidth=0, borderPadding=0,
        ),
        # 正文段落：10pt，行距 16pt，段间距 6pt
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"],
            fontSize=10, leading=16, textColor=COLOR_TEXT,
            spaceAfter=6,
        ),
        # 信息卡片内文字：10pt
        "info_text": ParagraphStyle(
            "InfoText", parent=base["Normal"],
            fontSize=10, leading=14, textColor=COLOR_TEXT,
        ),
        # 信息卡片标签：10pt 加粗
        "info_label": ParagraphStyle(
            "InfoLabel", parent=base["Normal"],
            fontSize=10, leading=14, textColor=COLOR_PRIMARY,
            fontName="Helvetica-Bold",
        ),
        # 图片标签：9pt 灰色居中
        "caption": ParagraphStyle(
            "ImageCaption", parent=base["Normal"],
            fontSize=9, textColor=COLOR_MUTED, alignment=TA_CENTER,
            spaceBefore=2, spaceAfter=4,
        ),
        # Final Diagnosis 高亮框内文字：11pt，稍大
        "highlight_body": ParagraphStyle(
            "HighlightBody", parent=base["Normal"],
            fontSize=11, leading=17, textColor=COLOR_TEXT,
            spaceAfter=4,
        ),
        # 高亮框标题：14pt 橙色
        "highlight_heading": ParagraphStyle(
            "HighlightHeading", parent=base["Heading2"],
            fontSize=14, textColor=COLOR_ACCENT,
            fontName="Helvetica-Bold",
            spaceBefore=0, spaceAfter=6,
        ),
        # 页脚免责声明：8pt 灰色
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"],
            fontSize=8, textColor=COLOR_MUTED,
            spaceBefore=12, spaceAfter=0,
        ),
        # 图片区域标题
        "image_section_title": ParagraphStyle(
            "ImageSectionTitle", parent=base["Heading2"],
            fontSize=13, textColor=COLOR_SECONDARY,
            fontName="Helvetica-Bold",
            spaceBefore=6, spaceAfter=8,
        ),
    }
    return styles


# ══════════════════════════════════════════════════════════
# 页眉 / 页脚 — 每页都绘制
# ══════════════════════════════════════════════════════════


def _header_footer(canvas, doc):
    """每页都绘制的页眉和页脚。

    页眉：左侧系统名称 + 右侧 "CONFIDENTIAL"，下方蓝色分割线
    页脚：左侧免责 + 右侧页码，上方灰色分割线
    """
    canvas.saveState()
    w, h = PAGE_W, PAGE_H

    # ── 页眉 ──
    header_y = h - 1.2 * cm

    # 左侧系统名称
    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(COLOR_PRIMARY)
    canvas.drawString(MARGIN_LEFT, header_y, "Eye Multi-Agent Diagnostic System")

    # 右侧标记
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(COLOR_MUTED)
    canvas.drawRightString(w - MARGIN_RIGHT, header_y, "AI-ASSISTED CLINICAL REPORT")

    # 蓝色分割线
    canvas.setStrokeColor(COLOR_PRIMARY)
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN_LEFT, header_y - 4 * mm, w - MARGIN_RIGHT, header_y - 4 * mm)

    # ── 页脚 ──
    footer_y = 1.2 * cm

    # 灰色分割线
    canvas.setStrokeColor(COLOR_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_LEFT, footer_y + 3 * mm, w - MARGIN_RIGHT, footer_y + 3 * mm)

    # 左侧免责
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(COLOR_MUTED)
    canvas.drawString(MARGIN_LEFT, footer_y - 2 * mm,
                      "AI-assisted analysis — not a substitute for professional medical judgment")

    # 右侧页码
    canvas.drawRightString(w - MARGIN_RIGHT, footer_y - 2 * mm,
                           f"Page {doc.page}")

    canvas.restoreState()


# ══════════════════════════════════════════════════════════
# 公开 API — 唯一的入口函数
# ══════════════════════════════════════════════════════════


def generate_pdf_report(
    markdown_text: str,
    original_image: str | None = None,
    vision_result: dict | None = None,
) -> str:
    """生成 PDF 诊断报告 — 本模块唯一的公开函数。

    Args:
        markdown_text: LLM 生成的 markdown 报告文本
        original_image: 用户上传的原始眼科图片路径
        vision_result: Vision Agent 的完整输出 dict，包含 mcp_results

    Returns:
        生成的 PDF 文件的绝对路径
    """
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORT_OUTPUT_DIR / f"report_{timestamp_str}.pdf"

    styles = _build_styles()
    story = []

    # ── 1. 报告头部 ──────────────────────────────────────
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Ophthalmic Diagnostic Report", styles["title"]))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(
        f"Eye Multi-Agent Diagnostic System&nbsp;&nbsp;|&nbsp;&nbsp;Generated: {ts}",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 3 * mm))

    # ── 2. 患者/图像信息卡片 ─────────────────────────────
    _add_info_card(story, styles, original_image, vision_result)

    # ── 3. 报告正文（6 个 section） ──────────────────────
    # 图片区会在 Section 1 (Image Analysis Summary) 之后自动插入
    overlay_images = _collect_report_images(vision_result) if vision_result else []
    _add_markdown_sections(story, styles, markdown_text, original_image, overlay_images)

    # ── 4. 末尾免责声明 ──────────────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=COLOR_BORDER,
        spaceBefore=4, spaceAfter=8,
    ))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report was generated by an AI-assisted multi-agent "
        "diagnostic system designed for clinical decision support. It does not constitute "
        "a definitive medical diagnosis. All findings, assessments, and recommendations "
        "must be reviewed and validated by a qualified ophthalmologist before clinical "
        f"action is taken. Report generated at {ts}.",
        styles["disclaimer"],
    ))

    # ── 5. 构建 PDF 文档 ─────────────────────────────────
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    logger.info(f"[PDF] Report saved: {pdf_path}")
    return str(pdf_path)


# ══════════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════════


def _add_info_card(story, styles, original_image, vision_result):
    """添加患者/图像信息摘要卡片 — 浅蓝色背景的信息栏。

    显示内容：图像类型、分析工具数量、生成时间等摘要信息。
    布局：2 列 Table，左列标签右列值，浅蓝背景 + 圆角效果。
    """
    info_rows = []

    # 报告日期
    info_rows.append([
        Paragraph("<b>Report Date:</b>", styles["info_label"]),
        Paragraph(datetime.now().strftime("%B %d, %Y"), styles["info_text"]),
    ])

    # 图像类型（从 vision_result 提取）
    image_type = "Not available"
    if vision_result:
        image_type = vision_result.get("image_type", "Ophthalmic image")
    info_rows.append([
        Paragraph("<b>Image Type:</b>", styles["info_label"]),
        Paragraph(image_type.replace("_", " ").title() if image_type else "N/A", styles["info_text"]),
    ])

    # 分析工具数量
    tool_count = 0
    tools_used = []
    if vision_result:
        mcp_results = vision_result.get("mcp_results", {})
        tool_count = len(mcp_results)
        tools_used = [TOOL_LABELS.get(t, t) for t in mcp_results.keys() if t in TOOL_LABELS]
    info_rows.append([
        Paragraph("<b>Analysis Tools:</b>", styles["info_label"]),
        Paragraph(f"{tool_count} tools executed" + (f" ({', '.join(tools_used)})" if tools_used else ""),
                  styles["info_text"]),
    ])

    # 图像路径
    if original_image:
        img_name = Path(original_image).name
        info_rows.append([
            Paragraph("<b>Source Image:</b>", styles["info_label"]),
            Paragraph(img_name, styles["info_text"]),
        ])

    # 构建表格 — 浅蓝背景
    col1_w = 3.5 * cm
    col2_w = CONTENT_WIDTH - col1_w
    tbl = Table(info_rows, colWidths=[col1_w, col2_w])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, COLOR_BORDER),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))


def _collect_report_images(vision_result: dict) -> list[tuple[str, str]]:
    """从 vision_result.mcp_results 中按优先级收集 overlay 分割图路径。"""
    mcp_results = vision_result.get("mcp_results", {})
    paths = []
    for tool_name in OVERLAY_PRIORITY:
        if tool_name in mcp_results:
            result = mcp_results[tool_name]
            if isinstance(result, dict):
                overlay = result.get("overlay_path")
                if overlay and Path(overlay).exists():
                    paths.append((tool_name, overlay))
    return paths


def _make_image(path: str, max_w=IMG_WIDTH, max_h=IMG_HEIGHT) -> Image | None:
    """加载图片并按比例缩放到 max_w × max_h 以内。"""
    p = Path(path)
    if not p.exists():
        logger.warning(f"[PDF] Image not found: {path}")
        return None
    try:
        img = Image(str(p), width=max_w, height=max_h)
        img.hAlign = "CENTER"
        iw, ih = img.imageWidth, img.imageHeight
        if iw > 0 and ih > 0:
            ratio = min(max_w / iw, max_h / ih)
            img.drawWidth = iw * ratio
            img.drawHeight = ih * ratio
        return img
    except Exception as e:
        logger.warning(f"[PDF] Failed to load image {path}: {e}")
        return None


def _add_image_section(story, styles, original_image, overlay_images):
    """添加诊断图片区域 — 带边框和浅灰背景的网格布局。"""
    all_imgs = []

    if original_image:
        img = _make_image(original_image)
        if img:
            all_imgs.append(("Original Fundus Image", img))

    for tool_name, path in overlay_images:
        img = _make_image(path)
        if img:
            label = TOOL_LABELS.get(tool_name, tool_name)
            all_imgs.append((label, img))

    if not all_imgs:
        return

    # 章节标题
    story.append(Paragraph("Diagnostic Images", styles["image_section_title"]))

    # 构建 2 列表格：每对图片占 2 行（图片行 + 标签行）
    rows = []
    for i in range(0, len(all_imgs), 2):
        pair = all_imgs[i:i + 2]
        img_row = [p[1] for p in pair]
        cap_row = [Paragraph(p[0], styles["caption"]) for p in pair]
        if len(img_row) == 1:
            img_row.append("")
            cap_row.append("")
        rows.append(img_row)
        rows.append(cap_row)

    col_w = 7.8 * cm
    tbl = Table(rows, colWidths=[col_w, col_w])
    tbl.setStyle(TableStyle([
        # 浅灰背景
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT_GRAY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        # 外边框
        ("BOX", (0, 0), (-1, -1), 0.8, COLOR_BORDER),
        # 列分割线
        ("LINEBEFORE", (1, 0), (1, -1), 0.3, COLOR_BORDER),
        # 图片行和标签行之间的细线（每 2 行一组）
        *[("LINEBELOW", (0, i), (-1, i), 0.3, COLOR_BORDER)
          for i in range(1, len(rows) - 1, 2)],
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))


def _add_markdown_sections(story, styles, markdown_text: str,
                           original_image=None, overlay_images=None):
    """解析 markdown 并添加到 story。

    图片区插入在 Section 1（Image Analysis Summary）之后，
    这样读者先看到文字分析，紧接着看到对应的分割图，图文对照最自然。
    Final Diagnosis 用橙色高亮框特殊处理。
    """
    sections = _parse_sections(markdown_text)
    images_inserted = False

    for idx, (heading, body) in enumerate(sections):
        is_final_diagnosis = "final diagnosis" in heading.lower()

        if is_final_diagnosis:
            _add_highlight_box(story, styles, heading, body)
        else:
            _add_section_with_indicator(story, styles, heading, body)

        # Section 1 之后插入图片区（只插一次）
        if idx == 0 and not images_inserted:
            if overlay_images:
                _add_image_section(story, styles, original_image, overlay_images)
            images_inserted = True

        # Section 之间的分割线（最后一个 section 后面不加）
        if idx < len(sections) - 1 and not is_final_diagnosis:
            story.append(Spacer(1, 2 * mm))
            story.append(HRFlowable(
                width="100%", thickness=0.3, color=COLOR_BORDER,
                spaceBefore=2, spaceAfter=6,
            ))


def _parse_sections(markdown_text: str) -> list[tuple[str, str]]:
    """把 markdown 文本拆分成 (heading, body) 列表。

    每遇到 ## 开头的行就开始新 section，后续行累积为 body。
    没有 heading 的开头部分会被跳过。
    """
    sections = []
    current_heading = None
    current_body_lines = []

    for line in markdown_text.strip().split("\n"):
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_body_lines).strip()))
            current_heading = line.lstrip("#").strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_body_lines).strip()))

    return sections


def _add_section_with_indicator(story, styles, heading: str, body: str):
    """添加普通章节 — 左侧蓝色竖线指示器 + 标题 + 正文。

    用单行 Table 模拟左侧色块效果：
    ┌──┬─────────────────────────────┐
    │▋ │  1. Image Analysis Summary  │
    └──┴─────────────────────────────┘
    """
    # 标题行：用 Table 做左侧色块效果
    title_tbl = Table(
        [[Paragraph(_escape(heading), styles["heading"])]],
        colWidths=[CONTENT_WIDTH],
    )
    title_tbl.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # 左侧蓝色竖线
        ("LINEBEFORE", (0, 0), (0, -1), 3, COLOR_SECONDARY),
    ]))
    story.append(title_tbl)

    # 正文
    _render_body_text(story, styles, body)


def _add_highlight_box(story, styles, heading: str, body: str):
    """Final Diagnosis 高亮框 — 橙色左边框 + 浅橙背景。"""
    story.append(Spacer(1, 4 * mm))

    # 构建框内内容
    inner_story = [Paragraph(_escape(heading), styles["highlight_heading"])]

    # 解析正文
    for para_text in body.split("\n\n"):
        para_text = para_text.strip()
        if not para_text:
            continue
        html_lines = []
        for line in para_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                html_lines.append(f"&bull;&nbsp; {_escape(line[2:])}")
            elif line:
                html_lines.append(_escape(line))
        if html_lines:
            inner_story.append(Paragraph("<br/>".join(html_lines), styles["highlight_body"]))

    # 用 Table 包裹，做出高亮框效果
    highlight_bg = colors.HexColor("#fef9e7")
    cell_content = []
    for item in inner_story:
        cell_content.append([item])

    tbl = Table(cell_content, colWidths=[CONTENT_WIDTH])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), highlight_bg),
        ("LINEBEFORE", (0, 0), (0, -1), 4, COLOR_ACCENT),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#f0c27a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 4 * mm))

    # 高亮框后面加分割线
    story.append(HRFlowable(
        width="100%", thickness=0.3, color=COLOR_BORDER,
        spaceBefore=2, spaceAfter=6,
    ))


def _render_body_text(story, styles, body: str):
    """把正文文本渲染为 Paragraph 元素。

    处理：段落分割（空行）、bullet list（- 或 *）、普通文本。
    """
    if not body.strip():
        return

    for para_text in body.split("\n\n"):
        para_text = para_text.strip()
        if not para_text:
            continue
        html_lines = []
        for line in para_text.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                html_lines.append(f"&bull;&nbsp; {_escape(line[2:])}")
            elif line.startswith("  - ") or line.startswith("  * "):
                # 二级缩进列表
                html_lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp; {_escape(line.strip()[2:])}")
            elif line:
                html_lines.append(_escape(line))
        if html_lines:
            story.append(Paragraph("<br/>".join(html_lines), styles["body"]))


def _escape(text: str) -> str:
    """转义 XML 特殊字符（reportlab Paragraph 内部是 XML 解析的）。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
