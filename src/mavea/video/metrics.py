"""视频质量指标：无参考画质评估、黑帧检测、时长准确性。

注意：剪辑成片质量不使用 PSNR/SSIM（全参考指标，需要像素级参考视频），
因为剪辑改变了内容结构。这里使用无参考指标 + 程序化检查。
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np
import structlog

from mavea.video.ffmpeg import probe_video

logger = structlog.get_logger(__name__)


def detect_black_frames(
    path: str | Path,
    threshold: float = 10.0,
    sample_interval: int = 10,
) -> list[float]:
    """检测视频中的黑帧。

    Args:
        path: 视频路径
        threshold: 平均亮度低于此值视为黑帧（0-255）
        sample_interval: 每隔 N 帧采样一次（加速检测）

    Returns:
        黑帧时间点列表（秒）
    """
    cap = cv2.VideoCapture(str(Path(path).resolve()))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    black_timestamps: list[float] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) < threshold:
                black_timestamps.append(frame_idx / fps)
        frame_idx += 1

    cap.release()
    return black_timestamps


def estimate_blur(path: str | Path, sample_count: int = 10) -> float:
    """估计视频模糊度（Laplacian 方差，越高越清晰）。

    Returns:
        平均 Laplacian 方差，<100 可能较模糊
    """
    cap = cv2.VideoCapture(str(Path(path).resolve()))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return 0.0

    # 均匀采样 sample_count 帧
    indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames), dtype=int)
    scores: list[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

    cap.release()
    return float(np.mean(scores)) if scores else 0.0


def check_duration_accuracy(
    path: str | Path,
    target_duration: float,
    tolerance: float = 0.1,
) -> tuple[bool, float]:
    """检查成片时长是否符合目标。

    Args:
        target_duration: 目标时长（秒）
        tolerance: 允许偏差比例（0.1 = ±10%）

    Returns:
        (是否通过, 实际时长)
    """
    info = probe_video(path)
    diff = abs(info.duration - target_duration)
    passed = diff <= target_duration * tolerance
    return passed, info.duration


def compute_no_reference_score(path: str | Path) -> float:
    """计算无参考画质综合评分（1-5）。

    综合模糊度和亮度异常，返回粗略评分。
    精确的 NIQE/BRISQUE 需要额外模型权重，这里用轻量启发式替代。
    """
    cap = cv2.VideoCapture(str(Path(path).resolve()))
    if not cap.isOpened():
        return 1.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return 1.0

    indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
    blur_scores: list[float] = []
    brightness_scores: list[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_scores.append(float(np.mean(gray)))

    cap.release()

    if not blur_scores:
        return 1.0

    # 模糊度评分：Laplacian 方差 >500 清晰(5分)，<50 模糊(1分)
    avg_blur = float(np.mean(blur_scores))
    blur_score = min(5.0, max(1.0, avg_blur / 100))

    # 亮度评分：平均亮度在 40-220 之间为正常
    avg_brightness = float(np.mean(brightness_scores))
    if avg_brightness < 15 or avg_brightness > 245:
        brightness_score = 1.0
    elif avg_brightness < 30 or avg_brightness > 230:
        brightness_score = 2.5
    else:
        brightness_score = 5.0

    return round((blur_score * 0.7 + brightness_score * 0.3), 1)


# ==================== 音频客观指标（v8） ====================

def analyze_audio_quality(path: str | Path) -> dict:
    """用 ffmpeg volumedetect 分析成片响度与爆音（削波）。

    Returns:
        {"has_audio": bool, "mean_volume_db": float|None,
         "max_volume_db": float|None, "too_quiet": bool, "clipping": bool}
    """
    import subprocess

    from mavea.config import get_settings

    settings = get_settings()
    try:
        probe = probe_video(path)
        if not probe.has_audio:
            return {"has_audio": False, "mean_volume_db": None,
                    "max_volume_db": None, "too_quiet": True, "clipping": False}
    except Exception:
        pass

    cmd = [
        settings.video.ffmpeg_path, "-i", str(path),
        "-af", "volumedetect", "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        text = proc.stderr or ""
    except Exception as e:  # noqa: BLE001
        logger.warning("metrics.audio_failed", error=str(e))
        return {"has_audio": True, "mean_volume_db": None, "max_volume_db": None,
                "too_quiet": False, "clipping": False}

    mean_vol = max_vol = None
    for line in text.splitlines():
        line = line.strip()
        if "mean_volume:" in line:
            with contextlib.suppress(ValueError, IndexError):
                mean_vol = float(line.split("mean_volume:")[1].strip().split()[0])
        elif "max_volume:" in line:
            with contextlib.suppress(ValueError, IndexError):
                max_vol = float(line.split("max_volume:")[1].strip().split()[0])

    # 经验阈值：平均响度低于 -30dB 视为过轻；峰值高于 -0.5dB 视为有削波/爆音风险
    too_quiet = mean_vol is not None and mean_vol < -30.0
    clipping = max_vol is not None and max_vol >= -0.5
    return {
        "has_audio": True,
        "mean_volume_db": mean_vol,
        "max_volume_db": max_vol,
        "too_quiet": bool(too_quiet),
        "clipping": bool(clipping),
    }
