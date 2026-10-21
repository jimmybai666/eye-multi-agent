"""Agent Pipeline 评测主脚本

对 6 个眼科分类数据集跑完整 multi-agent pipeline，
输出格式兼容 eval_vlm/analyze_results.py。

用法:
    python -m app.evaluation.eval_agent --model agent_gpt5
    python -m app.evaluation.eval_agent --model agent_qwen3vl
    python -m app.evaluation.eval_agent --model agent_gpt5 --task glaucoma_detection
"""

import asyncio
import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.evaluation.adapter import (
    MODEL_PRESETS, apply_model_preset, run_pipeline,
    extract_text_for_parsing, judge_extract_label, PROJECT_ROOT,
    warmup_mcp, shutdown_mcp,
)
from app.evaluation.parse_response import parse_response
from app.utils.logger import get_logger

logger = get_logger(__name__)

EVAL_CONFIG_PATH = Path(__file__).parent / "config.json"
EVAL_CONFIG_SAMPLED = Path(__file__).parent / "config_sampled.json"
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "eval"


def _relocate_pdf(result: dict, task_id: str, image_name: str, output_dir: Path) -> str | None:
    """把 pipeline 生成的 PDF 重命名为 {task_id}_{image_stem}.pdf 并移到 output_dir/reports/"""
    src = result.get("pdf_path")
    if not src or not Path(src).exists():
        return None
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    dst = reports_dir / f"{task_id}_{stem}.pdf"
    Path(src).rename(dst)
    return str(dst)


async def evaluate_single_image(
    image_path: str,
    image_name: str,
    folder_name: str,
    ground_truth: str,
    task_prompt: str,
    task: dict,
    output_dir: Path,
) -> dict:
    """评测单张图片：跑 pipeline → Judge LLM 提取标签（降级到正则）"""
    t0 = time.perf_counter()
    try:
        result = await run_pipeline(image_path, task_prompt)
        elapsed_pipeline = time.perf_counter() - t0

        # 重命名 PDF：{task_id}_{image_stem}.pdf，移到 eval 输出目录
        pdf_path = _relocate_pdf(result, task["task_id"], image_name, output_dir)

        # 第一层：Judge LLM 从报告中提取标签
        prediction = await judge_extract_label(result, task)

        # 第二层降级：Judge 失败则用正则匹配
        if prediction == "UNKNOWN":
            text = extract_text_for_parsing(result)
            prediction = parse_response(text, task)

        elapsed = time.perf_counter() - t0
        is_correct = prediction == ground_truth
        text = extract_text_for_parsing(result)

        route = result.get("route", "unknown")
        status = "OK" if is_correct else "X"
        print(f"  [{status}] [{folder_name}] {image_name}: "
              f"GT={ground_truth}, Pred={prediction}, "
              f"route={route}, {elapsed:.1f}s")

        return {
            "image": image_name,
            "folder": folder_name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": is_correct,
            "route": route,
            "elapsed_seconds": round(elapsed, 1),
            "pipeline_time": result.get("_pipeline_time"),
            "node_timings": result.get("_node_timings", {}),
            "node_outputs": result.get("_node_outputs", {}),
            "raw_response": text[:2000],
            "pdf_path": pdf_path,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] {image_name}: {e} ({elapsed:.1f}s)")
        logger.error(f"[Eval] {image_name} failed: {e}", exc_info=True)
        return {
            "image": image_name,
            "folder": folder_name,
            "ground_truth": ground_truth,
            "prediction": "ERROR",
            "correct": False,
            "route": "error",
            "elapsed_seconds": round(elapsed, 1),
            "pipeline_time": None,
            "node_timings": {},
            "node_outputs": {},
            "raw_response": str(e),
        }
async def evaluate_task(task: dict, model_key: str, output_dir: Path) -> dict:
    """评测单个任务（断点续跑 + 逐条保存）"""
    task_id = task["task_id"]
    dataset_dir = PROJECT_ROOT / task["dataset_dir"]
    prompt = task["prompt"]
    folder_to_label = task["folder_to_label"]

    print(f"\n{'='*60}")
    print(f"Task: {task['task_name']}")
    print(f"{'='*60}")

    all_jobs = []
    for folder_name, ground_truth in folder_to_label.items():
        folder_path = dataset_dir / folder_name
        if not folder_path.exists():
            print(f"  [WARN] Folder not found: {folder_path}")
            continue
        images = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        for img_name in images:
            img_path = str((folder_path / img_name).resolve())
            all_jobs.append((img_path, img_name, folder_name, ground_truth))

    output_file = output_dir / f"{task_id}_results.json"
    existing_results = []
    finished = set()

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        details = existing.get("details", [])
        bad = sum(1 for d in details if d["prediction"] in ("ERROR", "UNKNOWN"))
        if len(details) >= len(all_jobs) and bad == 0:
            print(f"  [SKIP] Already complete ({len(all_jobs)} images).")
            return existing
        for d in details:
            if d["prediction"] not in ("ERROR", "UNKNOWN"):
                finished.add(f"{d['folder']}/{d['image']}")
                existing_results.append(d)

    jobs = [(p, n, f, g) for p, n, f, g in all_jobs if f"{f}/{n}" not in finished]
    print(f"  Total: {len(all_jobs)}, Done: {len(finished)}, Remaining: {len(jobs)}")

    display_name = MODEL_PRESETS[model_key]["display_name"]
    new_results = []

    def save_progress():
        current = existing_results + new_results
        progress = {
            "task_id": task_id,
            "task_name": task["task_name"],
            "model": display_name,
            "accuracy": 0,
            "correct": sum(1 for r in current if r["correct"]),
            "total": len(current),
            "class_stats": {},
            "details": current,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    for img_path, img_name, folder_name, ground_truth in jobs:
        result = await evaluate_single_image(
            img_path, img_name, folder_name, ground_truth, prompt, task,
            output_dir,
        )
        new_results.append(result)
        save_progress()

    results = existing_results + new_results

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r["ground_truth"]
        class_stats[gt]["total"] += 1
        if r["correct"]:
            class_stats[gt]["correct"] += 1

    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.2%}")

    task_result = {
        "task_id": task_id,
        "task_name": task["task_name"],
        "model": display_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_stats": dict(class_stats),
        "details": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(task_result, f, ensure_ascii=False, indent=2)

    return task_result
async def main():
    parser = argparse.ArgumentParser(description="Agent Pipeline 评测")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_PRESETS.keys()),
                        help="模型配置 (agent_gpt5 / agent_qwen3vl)")
    parser.add_argument("--task", type=str, default=None,
                        help="只跑指定任务 (如 glaucoma_detection)")
    parser.add_argument("--sampled", action="store_true",
                        help="使用抽样数据集 (每类5张, 共90张)")
    args = parser.parse_args()

    apply_model_preset(args.model)

    print("Warming up MCP connection...")
    await warmup_mcp()

    output_dir = OUTPUT_BASE / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = EVAL_CONFIG_SAMPLED if args.sampled else EVAL_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)["tasks"]

    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
        if not tasks:
            print(f"Task not found: {args.task}")
            return

    display_name = MODEL_PRESETS[args.model]["display_name"]
    print(f"Model: {display_name}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_dir}")

    all_results = []
    t_start = time.perf_counter()

    for task in tasks:
        result = await evaluate_task(task, args.model, output_dir)
        all_results.append(result)

    total_time = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"SUMMARY - {display_name}")
    print(f"{'='*60}")
    print(f"{'Task':<35} {'Accuracy':<12} {'N'}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['task_name']:<35} {r['accuracy']:<12.2%} {r['total']}")

    avg_acc = sum(r["accuracy"] for r in all_results) / len(all_results) if all_results else 0
    print("-" * 55)
    print(f"{'AVERAGE':<35} {avg_acc:<12.2%}")
    print(f"\nTotal time: {total_time:.0f}s")

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": display_name,
            "model_key": args.model,
            "average_accuracy": round(avg_acc, 4),
            "total_time_seconds": round(total_time, 1),
            "tasks": [{
                "task_id": r["task_id"],
                "task_name": r["task_name"],
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
            } for r in all_results],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")

    await shutdown_mcp()


if __name__ == "__main__":
    asyncio.run(main())
