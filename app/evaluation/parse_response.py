"""标签解析模块 — 从模型回复中提取分类标签

"""

import re


def parse_response(response: str, task: dict) -> str:
    """从模型原始回复文本中解析预测标签"""
    clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    clean_response = re.sub(
        r"^[A-Z\-\/\s]+;\s*Explanation:\s*<[A-Z]+>\.?",
        "", clean_response, flags=re.IGNORECASE
    ).strip()
    clean_response = re.sub(
        r"^either\s+.+?;\s*Explanation:.*?(?:\n|$)",
        "", clean_response, flags=re.IGNORECASE
    ).strip()
    clean_response = re.sub(
        r"^Format:.*?(?:\n|$)",
        "", clean_response, flags=re.IGNORECASE
    ).strip()

    response_upper = clean_response.upper().strip()
    task_id = task["task_id"]

    if task_id == "glaucoma_detection":
        pos_non = -1
        for pattern in ["NON-GLAUCOMA", "NON GLAUCOMA"]:
            p = response_upper.find(pattern)
            if p != -1 and (pos_non == -1 or p < pos_non):
                pos_non = p
        pos_gla = response_upper.find("GLAUCOMA")
        if pos_gla != -1 and pos_non != -1 and pos_gla >= pos_non and pos_gla <= pos_non + 4:
            next_gla = response_upper.find("GLAUCOMA", pos_non + 13)
            pos_gla = next_gla
        if pos_gla != -1 and (pos_non == -1 or pos_gla < pos_non):
            return "GLAUCOMA"
        if pos_non != -1:
            return "NON-GLAUCOMA"

    elif task_id == "cataract_detection":
        if re.search(r"\bNORMAL\s*[;:]", response_upper):
            return "NORMAL"
        if re.search(r"\bCATARACT\s*[;:]", response_upper):
            return "CATARACT"
        pos_normal = response_upper.find("NORMAL")
        pos_cataract = response_upper.find("CATARACT")
        if pos_normal != -1 and pos_cataract != -1:
            return "NORMAL" if pos_normal < pos_cataract else "CATARACT"
        elif pos_cataract != -1:
            return "CATARACT"
        elif pos_normal != -1:
            return "NORMAL"

    elif task_id == "amd_detection":
        pos_non = -1
        for pattern in ["NON-AMD", "NON AMD"]:
            p = response_upper.find(pattern)
            if p != -1 and (pos_non == -1 or p < pos_non):
                pos_non = p
        pos_amd = response_upper.find("AMD")
        if pos_amd != -1 and pos_non != -1 and pos_amd >= pos_non and pos_amd <= pos_non + 4:
            next_amd = response_upper.find("AMD", pos_non + 7)
            pos_amd = next_amd
        if pos_amd != -1 and (pos_non == -1 or pos_amd < pos_non):
            return "AMD"
        if pos_non != -1:
            return "NON-AMD"

    elif task_id == "macular_hole_staging":
        match = re.search(r"STAGE[:\s]*\**(\d)\**", response_upper)
        if match:
            return match.group(1)
        match = re.search(r"\b([1-4])\b", response_upper)
        if match:
            return match.group(1)

    elif task_id == "dr_grading":
        match = re.search(r"GRADE[:\s]*\**(\d)\**", response_upper)
        if match:
            return match.group(1)
        grade_keywords = {
            "0": ["NO DR", "NO DIABETIC", "GRADE 0"],
            "1": ["MILD", "GRADE 1"],
            "2": ["MODERATE", "GRADE 2"],
            "3": ["SEVERE", "GRADE 3"],
            "4": ["PROLIFERATIVE", "PDR", "GRADE 4"]
        }
        for grade, keywords in grade_keywords.items():
            for kw in keywords:
                if kw in response_upper:
                    return grade

    elif task_id == "macular_edema_risk":
        match = re.search(r"RISK[:\s]*\**(\d)\**", response_upper)
        if match:
            return match.group(1)
        if "NO RISK" in response_upper or "NO MACULAR EDEMA" in response_upper:
            return "0"
        elif "MACULAR EDEMA PRESENT" in response_upper or "EDEMA IS PRESENT" in response_upper:
            return "2"
        elif "AT RISK" in response_upper or "RISK OF" in response_upper:
            return "1"

    return "UNKNOWN"
