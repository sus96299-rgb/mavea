"""FastAPI 路由。"""

from __future__ import annotations

import shutil
from pathlib import Path

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from mavea import __version__
from mavea.agents.graph import run_pipeline
from mavea.api.schemas import EditRequest, EditResponse, StatusResponse
from mavea.config import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/api/health", response_model=StatusResponse)
async def health():
    """健康检查。"""
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return StatusResponse(
        status="ok",
        version=__version__,
        ffmpeg_available=ffmpeg_ok,
    )


@router.post("/api/edit", response_model=EditResponse)
async def edit(request: EditRequest):
    """执行剪辑流水线（素材已在服务器上）。"""
    settings = get_settings()

    # 校验素材路径
    for p in request.material_paths:
        path = Path(p).resolve()
        if not path.exists():
            raise HTTPException(400, f"素材不存在: {p}")
        # 路径安全检查
        try:
            path.relative_to(settings.workspace_path)
        except ValueError:
            raise HTTPException(400, f"素材路径不在工作目录内: {p}") from None

    try:
        final_state = await run_pipeline(
            material_paths=request.material_paths,
            user_prompt=request.prompt,
            max_iterations=request.max_iterations,
        )

        def _get(key, default=None):
            if isinstance(final_state, dict):
                return final_state.get(key, default)
            return getattr(final_state, key, default)

        exec_result = _get("execution_result")
        eval_result = _get("evaluation_result")

        if exec_result and exec_result.success:
            return EditResponse(
                success=True,
                output_path=exec_result.output_path,
                overall_score=eval_result.overall if eval_result else None,
                iteration=_get("iteration", 1),
                summary=f"完成剪辑，共 {exec_result.tool_call_count} 次工具调用",
            )
        else:
            errors = _get("errors", [])
            return EditResponse(
                success=False,
                error="; ".join(errors) if errors else "剪辑失败",
                iteration=_get("iteration", 1),
            )
    except Exception as e:
        logger.error("api.edit.failed", error=str(e))
        raise HTTPException(500, str(e)) from e


@router.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """上传素材文件。"""
    settings = get_settings()
    upload_dir = settings.temp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 安全文件名
    filename = Path(file.filename).name
    dest = upload_dir / filename

    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info("api.uploaded", file=filename, size=len(content))
    return {"path": str(dest), "filename": filename, "size": len(content)}


@router.get("/api/download/{filename}")
async def download(filename: str):
    """下载成片。"""
    settings = get_settings()
    file_path = settings.output_path / filename
    if not file_path.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
    )
