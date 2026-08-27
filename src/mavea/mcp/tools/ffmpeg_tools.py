"""MCP FFmpeg 工具：视频信息、裁剪、拼接、字幕、文字贴纸、BGM。

这些函数通过 MCP Server 注册为工具，供 Agent 通过 MCP Client 调用。
每个函数返回 dict（success/data/error），不抛异常给 MCP 层。
"""

from __future__ import annotations

from typing import Any

import structlog

from mavea.mcp.tools._helpers import make_result, resolve_input_path
from mavea.video import ffmpeg

logger = structlog.get_logger(__name__)


def get_video_info(path: str) -> dict[str, Any]:
    """获取视频元信息（分辨率、帧率、时长、编码、音轨）。

    Args:
        path: 视频文件路径
    """
    try:
        p = resolve_input_path(path)
        info = ffmpeg.probe_video(p)
        return make_result(True, {
            "path": info.path,
            "duration": round(info.duration, 3),
            "width": info.width,
            "height": info.height,
            "fps": round(info.fps, 2),
            "codec": info.codec,
            "bitrate": info.bitrate,
            "has_audio": info.has_audio,
            "audio_codec": info.audio_codec,
            "size_bytes": info.size_bytes,
        })
    except Exception as e:
        logger.error("mcp.get_video_info.failed", path=path, error=str(e))
        return make_result(False, error=str(e))


def cut_clip(path: str, start: float, end: float, output_path: str | None = None) -> dict[str, Any]:
    """裁剪视频片段。

    Args:
        path: 源视频路径
        start: 起始时间（秒）
        end: 结束时间（秒）
        output_path: 输出路径（可选，默认自动生成）
    """
    try:
        p = resolve_input_path(path)
        out = ffmpeg.cut_clip(p, start, end, output_path)
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.cut_clip.failed", path=path, error=str(e))
        return make_result(False, error=str(e))


def concat_videos(
    paths: list[str],
    transition: str = "cut",
    transition_duration: float = 0.5,
    target_resolution: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """拼接多个视频片段，支持转场效果。

    Args:
        paths: 视频路径列表（至少2个）
        transition: 转场类型 cut/fade/dissolve/wipe/zoom/slide
        transition_duration: 转场时长（秒）
        target_resolution: 统一分辨率如 1080x1920（可选）
        output_path: 输出路径（可选）
    """
    try:
        resolved = [resolve_input_path(p) for p in paths]
        out = ffmpeg.concat_videos(
            resolved, transition, transition_duration, target_resolution, output_path
        )
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.concat_videos.failed", error=str(e))
        return make_result(False, error=str(e))


def add_subtitle(
    path: str,
    srt_path: str,
    style: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """烧录 SRT 字幕到视频。

    Args:
        path: 视频路径
        srt_path: SRT 字幕文件路径
        style: 字幕样式（FFmpeg force_style 字符串）
        output_path: 输出路径（可选）
    """
    try:
        p = resolve_input_path(path)
        srt = resolve_input_path(srt_path)
        out = ffmpeg.add_subtitle(p, srt, style, output_path)
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.add_subtitle.failed", error=str(e))
        return make_result(False, error=str(e))


def add_text_overlay(
    path: str,
    text: str,
    start: float,
    end: float,
    font_size: int = 48,
    font_color: str = "white",
    position: str = "center",
    output_path: str | None = None,
) -> dict[str, Any]:
    """在视频上叠加文字贴纸。

    Args:
        path: 视频路径
        text: 文字内容
        start: 显示开始时间（秒）
        end: 显示结束时间（秒）
        font_size: 字体大小
        font_color: 字体颜色
        position: 位置 top/center/bottom/top_left/top_right/bottom_left/bottom_right
        output_path: 输出路径（可选）
    """
    try:
        p = resolve_input_path(path)
        out = ffmpeg.add_text_overlay(
            p, text, start, end, font_size, font_color, position, output_path
        )
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.add_text_overlay.failed", error=str(e))
        return make_result(False, error=str(e))


def add_bgm(
    path: str,
    audio_path: str,
    volume: float = 0.15,
    loop: bool = True,
    keep_original_audio: bool = True,
    output_path: str | None = None,
) -> dict[str, Any]:
    """添加背景音乐。

    Args:
        path: 视频路径
        audio_path: 背景音乐路径
        volume: BGM音量 0-1
        loop: 音乐短于视频时是否循环
        keep_original_audio: 是否保留原音轨（True=混音）
        output_path: 输出路径（可选）
    """
    try:
        p = resolve_input_path(path)
        audio = resolve_input_path(audio_path)
        out = ffmpeg.add_bgm(p, audio, volume, loop, keep_original_audio, output_path)
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.add_bgm.failed", error=str(e))
        return make_result(False, error=str(e))
