"""进阶剪辑效果：自动卡点、歌词字幕、轻度美颜。

全部基于本地 ffmpeg + numpy 实现，不引入 librosa 等重依赖：
- detect_beats：用能量包络做简易节拍/起振检测，返回节拍时间点（秒）。
- snap_durations_to_beats：把各片段时长边界吸附到最近节拍，实现“卡点”。
- parse_lrc / segments_to_srt：解析 .lrc 或把 Whisper 分段写成 SRT。
- beautify：轻度磨皮 + 提亮 + 饱和的全局美颜滤镜（不做人脸替换）。

设计原则：任何一步失败都只返回 None / 原值，绝不阻断主成片流程。
"""
from __future__ import annotations

import re
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mavea.config import get_settings

# ==================== 节拍检测 / 卡点 ====================

def detect_beats(audio_path: str | Path, max_seconds: float | None = None) -> list[float]:
    """检测音频中的节拍/强起振时间点。

    做法：ffmpeg 解码为 22050Hz 单声道 PCM -> numpy 计算短时能量包络 ->
    对包络做一阶差分（起振强度）-> 自适应阈值 + 最小间隔峰选。

    Returns:
        节拍时间列表（秒），升序。检测失败返回空列表。
    """
    try:
        settings = get_settings()
        sr = 22050
        cmd = [
            settings.video.ffmpeg_path,
            "-v", "error",
            "-i", str(audio_path),
            "-ac", "1",            # 单声道
            "-ar", str(sr),        # 采样率
            "-f", "s16le",         # 原始 PCM
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if proc.returncode != 0 or not proc.stdout:
            return []

        audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < sr:  # 不足 1 秒，无法判断
            return []

        # 短时能量：帧长 ~23ms，帧移 ~12ms
        frame, hop = 512, 256
        n_frames = max(1, (audio.size - frame) // hop + 1)
        rms = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            seg = audio[i * hop: i * hop + frame]
            rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-9))
        times = np.arange(n_frames) * hop / sr

        # 起振强度 = 能量对数包络的正向差分
        env = np.log1p(rms * 8.0)
        onset = np.diff(env, prepend=env[0])
        onset = np.maximum(onset, 0.0)
        # 平滑，去掉毛刺
        kernel = np.ones(3) / 3.0
        onset = np.convolve(onset, kernel, mode="same")

        threshold = float(np.mean(onset) + 1.3 * np.std(onset))
        min_gap = 0.28  # 最快约 215 BPM，避免把一个鼓点拆成多个
        beats: list[float] = []
        last = -10.0
        order = np.argsort(onset)[::-1]  # 从强到弱挑峰
        for idx in order:
            if onset[idx] < threshold:
                break
            t = float(times[idx])
            if max_seconds is not None and t > max_seconds:
                continue
            if abs(t - last) < min_gap:
                continue
            # 与已选节拍保持最小间隔
            if any(abs(t - b) < min_gap for b in beats):
                continue
            beats.append(t)
            last = t
        beats.sort()
        return beats
    except Exception:
        return []


def snap_durations_to_beats(
    durations: Sequence[float],
    beats: Sequence[float],
    max_shift: float = 0.7,
    min_seg: float = 0.4,
) -> list[float]:
    """把片段边界吸附到最近节拍，保持总时长基本不变（末段吸收误差）。

    Args:
        durations: 原始各片段时长。
        beats: 节拍时间点（秒）。
        max_shift: 允许单个边界移动的最大秒数，超过就不动。
        min_seg: 单段最短时长，避免过快闪切。
    """
    durations = [float(d) for d in durations]
    if not durations:
        return []
    if not beats:
        return [round(d, 2) for d in durations]

    total = sum(durations)
    beats = [b for b in beats if 0.3 < b < total - 0.3]
    if not beats:
        return [round(d, 2) for d in durations]

    out: list[float] = []
    cum = 0.0
    for i, d in enumerate(durations):
        if i == len(durations) - 1:
            out.append(round(max(min_seg, total - cum), 2))
            break
        target = cum + d
        nearest = min(beats, key=lambda b: abs(b - target))
        if abs(nearest - target) <= max_shift and nearest - cum >= min_seg:
            new_d = nearest - cum
        else:
            new_d = d
        new_d = round(max(min_seg, new_d), 2)
        out.append(new_d)
        cum += new_d
    return out


# ==================== 歌词 / 字幕 ====================

_LRC_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")


def parse_lrc(lrc_text: str) -> list[tuple[float, str]]:
    """解析 LRC 歌词文本，返回 [(开始秒, 歌词), ...]，按时间升序。"""
    result: list[tuple[float, str]] = []
    for line in lrc_text.splitlines():
        tags = list(_LRC_RE.finditer(line))
        if not tags:
            continue
        text = _LRC_RE.sub("", line).strip()
        if not text:
            continue
        for m in tags:
            minute = int(m.group(1))
            second = int(m.group(2))
            frac = m.group(3) or "0"
            frac_value = int(frac) / (10.0 ** len(frac))
            t = minute * 60 + second + frac_value
            result.append((t, text))
    result.sort(key=lambda x: x[0])
    return result


def seconds_to_srt_time(seconds: float) -> str:
    """秒转 SRT 时间戳 HH:MM:SS,mmm。"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: Sequence[dict], max_duration: float | None = None) -> str:
    """把 [{start,end,text}, ...] 转成 SRT 字符串。"""
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 2.0))
        if max_duration is not None and start > max_duration:
            break
        if max_duration is not None:
            end = min(end, max_duration)
        if end <= start:
            end = start + 1.0
        lines.append(f"{idx}")
        lines.append(f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def lrc_to_srt(lrc_text: str, line_duration: float = 3.0, max_duration: float | None = None) -> str:
    """把 LRC 转成 SRT（每行歌词持续到下一行）。"""
    pairs = parse_lrc(lrc_text)
    segs: list[dict] = []
    for i, (t, text) in enumerate(pairs):
        end = pairs[i + 1][0] if i + 1 < len(pairs) else t + line_duration
        segs.append({"start": t, "end": end, "text": text})
    return segments_to_srt(segs, max_duration=max_duration)


# ==================== 美颜 ====================

# 轻度磨皮 + 微提亮 + 微增饱和对比；全局安全滤镜，不涉及人脸替换
_BEAUTY_VF = (
    "hqdn3d=1.5:1.5:3:3,"          # 降噪柔化（近似磨皮）
    "eq=brightness=0.025:saturation=1.08:contrast=1.03:gamma=1.02,"
    "unsharp=5:5:0.25"             # 轻微锐化抵消模糊
)


def beautify(video_path: str | Path) -> Path | None:
    """对成片施加轻度美颜滤镜，返回新文件路径；失败返回 None。"""
    try:
        settings = get_settings()
        src = Path(video_path)
        out = settings.output_path / f"beauty_{uuid.uuid4().hex[:8]}.mp4"
        settings.output_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            settings.video.ffmpeg_path, "-y", "-v", "error",
            "-i", str(src),
            "-vf", _BEAUTY_VF,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", settings.video.default_video_codec,
            "-crf", str(settings.video.default_crf),
            "-c:a", "aac",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        return None
    except Exception:
        return None


def mix_voiceover(
    video_path: str | Path,
    voice_path: str | Path,
    bgm_volume: float = 0.30,
    voice_volume: float = 1.0,
) -> Path | None:
    """把旁白混到已含背景音乐的成片上：压低 BGM、突出人声。失败返回 None。"""
    try:
        settings = get_settings()
        out = settings.output_path / f"voice_{uuid.uuid4().hex[:8]}.mp4"
        settings.output_path.mkdir(parents=True, exist_ok=True)
        fc = (
            f"[0:a]volume={bgm_volume}[bg];"
            f"[1:a]volume={voice_volume},apad[vo];"
            f"[bg][vo]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        base = [
            settings.video.ffmpeg_path, "-y", "-v", "error",
            "-i", str(video_path), "-i", str(voice_path),
            "-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]",
        ]
        # 先尝试直接复制视频流（快），失败再重编码
        for extra in (
            ["-c:v", "copy", "-c:a", "aac"],
            ["-c:v", settings.video.default_video_codec,
             "-crf", str(settings.video.default_crf), "-c:a", "aac"],
        ):
            cmd = base + extra + [str(out)]
            proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out
        return None
    except Exception:
        return None
