"""MCP 媒体工具：图片转视频、音频提取、音频转写。"""

from __future__ import annotations

from typing import Any

import structlog

from mavea.config import get_settings
from mavea.mcp.tools._helpers import make_result, resolve_input_path
from mavea.video import ffmpeg

logger = structlog.get_logger(__name__)


def image_to_video(
    image_paths: list[str],
    duration_per_image: float = 3.0,
    ken_burns: bool = True,
    resolution: str = "1080x1920",
    fps: int = 30,
    output_path: str | None = None,
) -> dict[str, Any]:
    """将图片序列转为视频片段（支持 Ken Burns 推拉摇移效果）。

    Args:
        image_paths: 图片路径列表
        duration_per_image: 每张图片展示时长（秒）
        ken_burns: 是否启用缓慢推拉效果
        resolution: 输出分辨率如 1080x1920
        fps: 帧率
        output_path: 输出路径（可选）
    """
    try:
        resolved = [resolve_input_path(p) for p in image_paths]
        out = ffmpeg.image_to_video(
            resolved, duration_per_image, ken_burns, resolution, fps, output_path
        )
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.image_to_video.failed", error=str(e))
        return make_result(False, error=str(e))


def extract_audio(
    path: str,
    format: str = "wav",
    output_path: str | None = None,
) -> dict[str, Any]:
    """从视频提取音频。

    Args:
        path: 视频路径
        format: 音频格式 wav/mp3/aac
        output_path: 输出路径（可选）
    """
    try:
        p = resolve_input_path(path)
        out = ffmpeg.extract_audio(p, output_path, fmt=format)
        return make_result(True, {"output_path": str(out)})
    except Exception as e:
        logger.error("mcp.extract_audio.failed", error=str(e))
        return make_result(False, error=str(e))


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: str = "zh",
) -> dict[str, Any]:
    """音频转文字（faster-whisper）。

    Args:
        audio_path: 音频文件路径
        model_size: 模型大小 tiny/base/small/medium/large-v3
        language: 语言代码 zh/en/auto
    """
    try:
        p = resolve_input_path(audio_path)
        settings = get_settings()

        # 延迟导入，避免未安装时影响其他工具
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_size,
            device=settings.video.whisper_device,
            compute_type="int8",
        )
        segments_iter, info = model.transcribe(
            str(p), language=language if language != "auto" else None
        )

        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        return make_result(True, {
            "segments": segments,
            "full_text": " ".join(full_text_parts),
            "language": info.language,
        })
    except Exception as e:
        logger.error("mcp.transcribe_audio.failed", error=str(e))
        return make_result(False, error=str(e))
