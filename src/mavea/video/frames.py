"""关键帧提取：按固定间隔或场景边界提取代表帧。"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import structlog

from mavea.config import get_settings
from mavea.video.scene_detect import detect_scenes

logger = structlog.get_logger(__name__)


def extract_frames_interval(
    path: str | Path,
    interval: float = 2.0,
    output_dir: Path | None = None,
) -> list[Path]:
    """按固定时间间隔提取帧。

    Args:
        path: 视频路径
        interval: 间隔秒数
        output_dir: 输出目录，None 则用 temp/frames_{random}

    Returns:
        提取的帧图片路径列表（按时间排序）
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.temp_path / f"frames_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / "frame_%04d.jpg")
    cmd = [
        settings.video.ffmpeg_path, "-y",
        "-i", str(Path(path).resolve()),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info("frames.extract_interval", count=len(frames), interval=interval)
    return frames


def extract_scene_keyframes(
    path: str | Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.6,
    output_dir: Path | None = None,
) -> list[tuple[float, float, Path]]:
    """检测场景并提取每个场景的代表帧（取场景中点）。

    Returns:
        list of (start_sec, end_sec, keyframe_path)
    """
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.temp_path / f"keyframes_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = detect_scenes(path, threshold, min_scene_len)
    results: list[tuple[float, float, Path]] = []

    for i, (start, end) in enumerate(scenes):
        mid = (start + end) / 2
        frame_path = output_dir / f"scene_{i:03d}.jpg"
        cmd = [
            settings.video.ffmpeg_path, "-y",
            "-ss", str(mid),
            "-i", str(Path(path).resolve()),
            "-frames:v", "1",
            "-q:v", "2",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0 and frame_path.exists():
            results.append((start, end, frame_path))
        else:
            logger.warning("frames.keyframe_failed", scene=i, stderr=result.stderr[-200:])

    logger.info("frames.extract_scene_keyframes", scenes=len(results))
    return results
