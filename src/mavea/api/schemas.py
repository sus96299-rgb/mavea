"""FastAPI 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EditRequest(BaseModel):
    """剪辑请求。"""
    material_paths: list[str] = Field(description="素材文件路径列表")
    prompt: str = Field(description="用户需求描述", min_length=1)
    max_iterations: int = Field(default=3, ge=1, le=5)


class EditResponse(BaseModel):
    """剪辑响应。"""
    success: bool
    output_path: str | None = None
    overall_score: float | None = None
    iteration: int = 1
    summary: str = ""
    error: str | None = None


class StatusResponse(BaseModel):
    """服务状态。"""
    status: str
    version: str
    ffmpeg_available: bool
