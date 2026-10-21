"""FastAPI 服务入口

启动方式：python app/main.py
前端界面：http://localhost:8000/
API 文档：http://localhost:8000/docs
"""

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.graph.builder import build_graph
from app.state.state import create_initial_state

# ── 路径常量 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# ── 创建 FastAPI 应用 ──────────────────────────────────
app = FastAPI(
    title="Eye Multi-Agent System",
    description="LangGraph-based multi-agent system for ophthalmic diagnosis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


# ── 请求体定义 ─────────────────────────────────────────
class DiagnoseRequest(BaseModel):
    input: str
    image_path: str | None = None


# ── 健康检查 ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ── 图片上传 ──────────────────────────────────────────
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传眼科图片，保存到 uploads/ 目录，返回绝对路径和预览 URL"""
    ext = Path(file.filename).suffix or ".png"
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    content = await file.read()
    save_path.write_bytes(content)

    return {
        "image_path": str(save_path.resolve()),
        "image_url": f"/uploads/{safe_name}",
        "filename": file.filename,
    }


# ── 核心诊断端点（JSON 响应，保持兼容） ───────────────
@app.post("/diagnose")
async def diagnose(request: DiagnoseRequest):
    """运行多 Agent 诊断管线，返回完整 JSON 结果"""
    try:
        initial_state = create_initial_state(
            input_text=request.input,
            image_path=request.image_path,
        )
        result = await graph.ainvoke(initial_state)
        return {
            "route": result.get("route"),
            "report": result.get("report"),
            "pdf_path": result.get("pdf_path"),
            "final_diagnosis": result.get("final_diagnosis"),
            "reason": result.get("reason"),
            "is_complete": result.get("is_complete"),
            "iteration_count": result.get("iteration_count"),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Pipeline execution failed"},
        )


# ── 流式诊断端点（SSE，节点级进度推送） ──────────────
@app.post("/diagnose-stream")
async def diagnose_stream(request: DiagnoseRequest):
    """流式诊断：通过 SSE 逐节点推送进度，前端实时显示当前在哪个 Agent"""
    initial_state = create_initial_state(
        input_text=request.input,
        image_path=request.image_path,
    )

    async def event_generator():
        final_state = {}
        try:
            async for chunk in graph.astream(initial_state):
                # astream 每次 yield 一个 dict: {node_name: node_output}
                for node_name, node_output in chunk.items():
                    # 合并到最终状态
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    # 根据节点类型提取关键信息推送给前端
                    event_data = {"type": "node_complete", "node": node_name}

                    if node_name == "supervisor":
                        event_data["data"] = {
                            "route": node_output.get("route"),
                            "reason": node_output.get("reason"),
                            "image_status": node_output.get("image_status"),
                        }
                    elif node_name == "vision":
                        vr = node_output.get("vision_result", {})
                        event_data["data"] = {
                            "image_type": vr.get("image_type") if isinstance(vr, dict) else None,
                            "has_findings": bool(vr.get("findings")) if isinstance(vr, dict) else False,
                        }
                    elif node_name == "clinical":
                        cr = node_output.get("clinical_result", {})
                        event_data["data"] = {
                            "has_diagnosis": bool(cr.get("diagnosis")) if isinstance(cr, dict) else False,
                        }
                    elif node_name == "critic":
                        cr = node_output.get("critic_result", {})
                        event_data["data"] = {
                            "approved": cr.get("approved") if isinstance(cr, dict) else None,
                            "iteration_count": node_output.get("iteration_count"),
                        }
                    elif node_name == "report":
                        event_data["data"] = {
                            "has_report": bool(node_output.get("report")),
                            "has_pdf": bool(node_output.get("pdf_path")),
                        }
                    else:
                        event_data["data"] = {}

                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

            # 所有节点完成，推送最终结果
            done_data = {
                "type": "done",
                "data": {
                    "route": final_state.get("route"),
                    "report": final_state.get("report"),
                    "pdf_path": final_state.get("pdf_path"),
                    "final_diagnosis": final_state.get("final_diagnosis"),
                    "reason": final_state.get("reason"),
                    "is_complete": final_state.get("is_complete"),
                    "iteration_count": final_state.get("iteration_count"),
                },
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            error_data = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── PDF 下载 ──────────────────────────────────────────
@app.get("/download-report")
async def download_report(path: str = Query(..., description="PDF 文件路径")):
    """下载 PDF 诊断报告，带路径穿越防护"""
    pdf_path = Path(path).resolve()

    if not pdf_path.is_relative_to(REPORTS_DIR.resolve()):
        return JSONResponse(status_code=403, content={"error": "Access denied"})

    if not pdf_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


# ── 静态文件挂载（必须在所有路由之后） ─────────────────
# /uploads 挂载在 / 前面，避免被 catch-all 覆盖
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── 直接运行入口 ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
