"""MCP 工具共享辅助函数。"""
from __future__ import annotations

from pathlib import Path


def resolve_input_path(path: str) -> Path:
    """校验输入文件路径：必须存在且是文件。

    输入文件允许在工作目录外（用户显式提供的素材），
    但必须存在且不能是目录。输出路径的工作目录限制由 ffmpeg 层处理。
    """
    p = Path(path).resolve()
    if not p.exists():
        raise ValueError(f"文件不存在: {path}")
    if not p.is_file():
        raise ValueError(f"路径不是文件: {path}")
    return p


def make_result(success: bool, data: dict | None = None, error: str | None = None) -> dict:
    """构造统一的工具返回结构。"""
    result: dict = {"success": success}
    if data is not None:
        result["data"] = data
    if error is not None:
        result["error"] = error
    return result
