"""FFmpeg 命令构建与执行。

所有命令通过参数数组构建（subprocess.run 的 list 形式），禁止 shell=True。
每个函数返回 (output_path, command) 或直接执行并返回 output_path。
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

from mavea.config import get_settings

logger = structlog.get_logger(__name__)


class FFmpegError(RuntimeError):
    """FFmpeg 执行失败"""

    def __init__(self, message: str, cmd: list[str], stderr: str = ""):
        super().__init__(message)
        self.cmd = cmd
        self.stderr = stderr


@dataclass
class VideoProbeInfo:
    """ffprobe 返回的视频元信息"""

    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int | None
    has_audio: bool
    audio_codec: str | None
    size_bytes: int


def _settings():
    return get_settings()


def _generate_output_path(prefix: str, ext: str = "mp4") -> Path:
    """自动生成输出文件路径：{output_dir}/{prefix}_{timestamp}_{random}.{ext}"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:4]
    return _settings().output_path / f"{prefix}_{ts}_{rand}.{ext}"


def _resolve_output(output_path: str | Path | None, prefix: str, ext: str = "mp4") -> Path:
    """解析输出路径：None 则自动生成，否则校验在工作目录内并创建父目录。"""
    if output_path is None:
        return _generate_output_path(prefix, ext)
    p = Path(output_path)
    if not p.is_absolute():
        p = _settings().workspace_path / p
    p = p.resolve()
    # 校验在工作目录内
    _settings().validate_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _run_ffmpeg(cmd: list[str], timeout: int = 300) -> None:
    """执行 FFmpeg 命令，失败时抛出 FFmpegError。

    Args:
        cmd: 完整命令数组（第一个元素是 ffmpeg 路径）
        timeout: 超时秒数
    """
    settings = _settings()
    cmd[0] = settings.video.ffmpeg_path
    logger.info("ffmpeg.run", cmd=" ".join(str(c) for c in cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # Windows 中文环境下 ffmpeg 输出可能含非 GBK 字节，替换而非崩溃
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"FFmpeg 执行超时（{timeout}s）", cmd) from e
    except FileNotFoundError as e:
        raise FFmpegError(
            f"找不到 FFmpeg，请确认已安装且在 PATH 中（当前配置: {settings.video.ffmpeg_path}）",
            cmd,
        ) from e

    if result.returncode != 0:
        # 截取 stderr 最后 2000 字符避免日志过长
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        raise FFmpegError(
            f"FFmpeg 返回非零退出码 {result.returncode}",
            cmd,
            stderr_tail,
        )


def _ensure_audio_track(video_path: Path) -> Path:
    """确保视频有音频轨；没有则添加静音音轨。

    图片生成的视频默认无音频，拼接时会导致 xfade 滤镜链异常或播放器无声音轨。
    """
    try:
        probe = probe_video(video_path)
        if probe.has_audio:
            return video_path
    except Exception:
        return video_path

    tmp = video_path.with_name(video_path.stem + "_audio.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "copy",
        "-c:a", "aac", "-shortest",
        str(tmp),
    ]
    _run_ffmpeg(cmd)
    import os
    os.replace(str(tmp), str(video_path))
    return video_path


def probe_video(path: str | Path) -> VideoProbeInfo:
    """用 ffprobe 获取视频元信息。

    Raises:
        FFmpegError: ffprobe 执行失败或文件不是有效视频
    """
    settings = _settings()
    p = str(Path(path).resolve())
    cmd = [
        settings.video.ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        p,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30, check=False,
        )
    except FileNotFoundError as e:
        raise FFmpegError(f"找不到 ffprobe: {settings.video.ffprobe_path}", cmd) from e
    except subprocess.TimeoutExpired as e:
        raise FFmpegError("ffprobe 超时", cmd) from e

    if result.returncode != 0:
        raise FFmpegError(f"ffprobe 失败: {result.stderr[-500:]}", cmd, result.stderr)

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise FFmpegError(f"文件没有视频流: {p}", cmd)

    # 解析帧率 "30/1" 或 "30000/1001"
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0

    return VideoProbeInfo(
        path=p,
        duration=float(fmt.get("duration", 0)),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        codec=video_stream.get("codec_name", "unknown"),
        bitrate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        has_audio=audio_stream is not None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        size_bytes=int(fmt.get("size", 0)),
    )


def cut_clip(
    path: str | Path,
    start: float,
    end: float,
    output_path: str | Path | None = None,
) -> Path:
    """裁剪视频片段。

    Args:
        path: 源视频路径
        start: 起始时间（秒）
        end: 结束时间（秒）
        output_path: 输出路径，None 则自动生成
    """
    out = _resolve_output(output_path, "cut")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(Path(path).resolve()),
        "-t", str(end - start),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    # -c copy 在关键帧不对齐时可能黑屏，降级重新编码
    try:
        _run_ffmpeg(cmd)
    except FFmpegError:
        logger.warning("ffmpeg.cut_clip.copy_failed_retry_encode")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(Path(path).resolve()),
            "-t", str(end - start),
            "-c:v", _settings().video.default_video_codec,
            "-c:a", _settings().video.default_audio_codec,
            "-crf", str(_settings().video.default_crf),
            str(out),
        ]
        _run_ffmpeg(cmd)
    return out


def concat_videos(
    paths: list[str | Path],
    transition: str = "cut",
    transition_duration: float = 0.5,
    target_resolution: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """拼接多个视频。

    Args:
        paths: 视频路径列表
        transition: 转场类型（cut/fade/dissolve/wipe/zoom/slide）
        transition_duration: 转场时长（秒）
        target_resolution: 目标分辨率 "WxH"，None 则取第一个视频分辨率
        output_path: 输出路径
    """
    out = _resolve_output(output_path, "concat")
    resolved = [str(Path(p).resolve()) for p in paths]

    if transition == "cut" or len(paths) == 1:
        return _concat_cut(resolved, out, target_resolution)

    return _concat_xfade(resolved, transition, transition_duration, target_resolution, out)


def _concat_cut(paths: list[str], out: Path, target_resolution: str | None = None) -> Path:
    """硬切拼接：使用 concat demuxer，速度快但要求编码参数一致。"""
    settings = _settings()
    # 确定目标分辨率
    tw, th = 1920, 1080
    if target_resolution and "x" in target_resolution:
        tw, th = (int(x) for x in target_resolution.split("x"))
    normalized: list[str] = []
    for _i, p in enumerate(paths):
        norm = settings.temp_path / f"concat_norm_{uuid.uuid4().hex[:6]}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", p,
            "-c:v", settings.video.default_video_codec,
            "-preset", "ultrafast",
            "-c:a", settings.video.default_audio_codec,
            "-vf", f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                   f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black",
            "-r", str(settings.video.default_fps),
            "-crf", str(settings.video.default_crf),
            str(norm),
        ]
        _run_ffmpeg(cmd)
        normalized.append(str(norm))

    # 写 concat list 文件
    list_file = _settings().temp_path / f"concat_list_{uuid.uuid4().hex[:6]}.txt"
    list_content = "\n".join(f"file '{p}'" for p in normalized)
    list_file.write_text(list_content, encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out),
    ]
    try:
        _run_ffmpeg(cmd)
    finally:
        list_file.unlink(missing_ok=True)
    return out


def _concat_xfade(
    paths: list[str],
    transition: str,
    duration: float,
    target_resolution: str | None,
    out: Path,
) -> Path:
    """带转场拼接：使用 xfade 滤镜。

    xfade 要求所有输入分辨率和帧率一致，所以先统一规格。
    """
    settings = _settings()
    w, h = (1920, 1080)
    if target_resolution and "x" in target_resolution:
        w, h = (int(x) for x in target_resolution.split("x"))

    # 统一每个片段的分辨率和帧率
    normalized: list[str] = []
    for p in paths:
        norm = settings.temp_path / f"xfade_norm_{uuid.uuid4().hex[:6]}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", p,
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,fps={settings.video.default_fps}",
            "-c:v", settings.video.default_video_codec,
            "-preset", "ultrafast",
            "-c:a", settings.video.default_audio_codec,
            "-crf", str(settings.video.default_crf),
            str(norm),
        ]
        _run_ffmpeg(cmd)
        normalized.append(str(norm))

    # 构建 xfade + acrossfade filter_complex
    # 视频用 xfade，音频用 acrossfade 同步交叉淡化
    probe_infos = [probe_video(p) for p in normalized]

    inputs: list[str] = []
    for p in normalized:
        inputs.extend(["-i", p])

    filter_parts: list[str] = []
    prev_v = "0:v"
    prev_a = "0:a"
    offset = probe_infos[0].duration - duration

    for i in range(1, len(normalized)):
        v_out = f"v{i}" if i < len(normalized) - 1 else "vout"
        a_out = f"a{i}" if i < len(normalized) - 1 else "aout"
        filter_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition={transition}:duration={duration}:"
            f"offset={offset:.3f}[{v_out}]"
        )
        filter_parts.append(
            f"[{prev_a}][{i}:a]acrossfade=d={duration}:c1=tri:c2=tri[{a_out}]"
        )
        prev_v = v_out
        prev_a = a_out
        if i < len(normalized) - 1:
            offset += probe_infos[i].duration - duration

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", settings.video.default_video_codec,
        "-preset", "ultrafast",
        "-crf", str(settings.video.default_crf),
        "-c:a", "aac",
        str(out),
    ]
    _run_ffmpeg(cmd, timeout=600)
    return out


def add_subtitle(
    path: str | Path,
    srt_path: str | Path,
    style: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """烧录 SRT 字幕到视频。

    Args:
        path: 视频路径
        srt_path: SRT 字幕文件路径
        style: FFmpeg subtitles 滤镜的 force_style 参数，如
               "FontSize=24,PrimaryColour=&Hffffff,Alignment=2"
        output_path: 输出路径
    """
    out = _resolve_output(output_path, "subtitle")
    # Windows 盘符冒号需要转义，否则 FFmpeg 会把 D: 当成滤镜选项分隔符
    srt_posix = Path(srt_path).resolve().as_posix()
    if len(srt_posix) > 1 and srt_posix[1] == ":":
        srt_posix = srt_posix[0] + "\\:" + srt_posix[2:]
    vf = f"subtitles='{srt_posix}'"
    if style:
        vf += f":force_style='{style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(Path(path).resolve()),
        "-vf", vf,
        "-c:v", _settings().video.default_video_codec,
        "-c:a", "copy",
        "-crf", str(_settings().video.default_crf),
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def _sanitize_overlay_text(text: str) -> str:
    """清洗 drawtext 文案：剥掉 emoji 等微软雅黑不支持的字符，避免渲染成 □ 豆腐块。

    保留：各语言文字（含中文）、数字、空白、常用中英文标点；其余符号丢弃。
    """
    import unicodedata

    allowed_punct = set(
        "，。！？、：；""''（）《》【】「」｜|·—–-~～.!,?%‰<>→←↑↓ "
    )
    kept: list[str] = []
    for ch in str(text):
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N")) or ch in allowed_punct:
            kept.append(ch)
        elif ch in "\n\t":
            kept.append(" ")
        # 符号(So/Sk/C*)等一律丢弃（emoji、装饰图标等）
    return "".join(kept).strip()


def add_text_overlay(
    path: str | Path,
    text: str,
    start: float,
    end: float,
    font_size: int = 48,
    font_color: str = "white",
    position: str = "center",
    output_path: str | Path | None = None,
) -> Path:
    """添加文字贴纸（drawtext 滤镜），带半透明深色底条保证可读性。

    Args:
        position: top/center/bottom/top_left/top_right/bottom_left/bottom_right
    """
    out = _resolve_output(output_path, "text")
    text = _sanitize_overlay_text(text)
    if not text:
        # 清洗后为空（纯 emoji）：不烧字，直接复制输入，避免空文本导致滤镜报错
        import shutil
        shutil.copyfile(path, out)
        return out

    # 位置映射到 x/y 表达式
    pos_map = {
        "top": ("(w-text_w)/2", "40"),
        "center": ("(w-text_w)/2", "(h-text_h)/2"),
        "bottom": ("(w-text_w)/2", "h-text_h-40"),
        "top_left": ("40", "40"),
        "top_right": ("w-text_w-40", "40"),
        "bottom_left": ("40", "h-text_h-40"),
        "bottom_right": ("w-text_w-40", "h-text_h-40"),
    }
    x, y = pos_map.get(position, pos_map["center"])

    # 半透明深色底条（带货字幕标配，避免压在复杂背景上看不清）；
    # 角落位置空间小，不加整条底条
    box_h = int(font_size * 1.9)
    # 注意：drawbox 的盒子高度选项也叫 h，表达式里引用输入高度必须用 ih，
    # 否则 y=h-... 中的 h 会被当成盒子高度（实测底条会错位到画面顶部）
    box_map = {
        "top": f"drawbox=x=0:y=20:w=iw:h={box_h}:color=black@0.55:t=fill,",
        "center": f"drawbox=x=0:y=(ih-{box_h})/2:w=iw:h={box_h}:color=black@0.55:t=fill,",
        "bottom": f"drawbox=x=0:y=ih-{box_h}-24:w=iw:h={box_h}:color=black@0.55:t=fill,",
    }
    box_filter = box_map.get(position, "")

    # drawtext 文本转义：单引号、冒号、反斜杠
    safe_text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")

    # Windows 下需要指定 fontfile（用微软雅黑支持中文），Linux 用 DejaVu。
    # 注意 filtergraph 转义：盘符冒号只需单反斜杠转义（C\:），
    # 双反斜杠会被解析成"字面反斜杠+未转义冒号"导致 FFmpeg 退出码 -22。
    fontfile_arg = ""
    if Path("C:/Windows/Fonts/msyh.ttc").exists():
        fontfile_arg = "fontfile='C\\:/Windows/Fonts/msyh.ttc':"
    elif Path("C:/Windows/Fonts/arial.ttf").exists():
        fontfile_arg = "fontfile='C\\:/Windows/Fonts/arial.ttf':"
    elif Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").exists():
        fontfile_arg = "fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"

    drawtext = (
        f"{box_filter}"
        f"drawtext={fontfile_arg}"
        f"text='{safe_text}':"
        f"fontsize={font_size}:fontcolor={font_color}:"
        f"x={x}:y={y}:"
        f"enable='between(t,{start},{end})':"
        f"shadowcolor=black:shadowx=2:shadowy=2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(Path(path).resolve()),
        "-vf", drawtext,
        "-c:v", _settings().video.default_video_codec,
        "-c:a", "copy",
        "-crf", str(_settings().video.default_crf),
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def add_bgm(
    path: str | Path,
    audio_path: str | Path,
    volume: float = 0.15,
    loop: bool = True,
    keep_original_audio: bool = True,
    output_path: str | Path | None = None,
) -> Path:
    """添加背景音乐。

    Args:
        volume: BGM 音量（0-1）
        loop: 音乐短于视频时是否循环
        keep_original_audio: True=混音保留原音，False=替换为BGM
    """
    out = _resolve_output(output_path, "bgm")
    video = str(Path(path).resolve())
    bgm = str(Path(audio_path).resolve())

    # 获取视频时长用于裁剪/循环BGM
    info = probe_video(video)
    duration = info.duration

    if keep_original_audio and info.has_audio:
        # 混音：原音轨 + BGM
        if loop:
            cmd = [
                "ffmpeg", "-y",
                "-i", video,
                "-stream_loop", "-1", "-i", bgm,
                "-filter_complex",
                f"[1:a]volume={volume},atrim=duration={duration}[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", _settings().video.default_audio_codec,
                "-shortest",
                str(out),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video, "-i", bgm,
                "-filter_complex",
                f"[1:a]volume={volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", _settings().video.default_audio_codec,
                str(out),
            ]
    else:
        # 替换原音轨
        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-stream_loop", "-1" if loop else "0", "-i", bgm,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", _settings().video.default_audio_codec,
            "-af", f"volume={volume}",
            "-shortest",
            str(out),
        ]
    _run_ffmpeg(cmd)
    return out


# Ken Burns 运镜方向池：按片段序号轮换，避免"每张图都在缓慢放大"的单调感
_KEN_BURNS_MOTIONS = ("zoom_in", "zoom_out", "pan_right", "pan_left")


def _build_ken_burns_vf(
    w: int,
    h: int,
    fps: int,
    duration: float,
    motion: str = "auto",
    motion_index: int = 0,
) -> str:
    """构造 Ken Burns 滤镜串（纯函数，便于单测）。

    Args:
        motion: zoom_in/zoom_out/pan_right/pan_left/auto；
                auto 时按 motion_index 在四种运镜间轮换。
        motion_index: 片段在时间轴上的序号，决定 auto 轮换到哪种运镜。
    """
    if motion == "auto":
        motion = _KEN_BURNS_MOTIONS[motion_index % len(_KEN_BURNS_MOTIONS)]
    frames = max(1, int(duration * fps))
    center_x, center_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"

    if motion == "zoom_out":
        zoom = "max(1.14-0.0020*on,1.02)"
        x, y = center_x, center_y
    elif motion == "pan_right":
        # 固定放大倍率，视窗从左滑到右
        zoom = "1.12"
        x, y = f"(iw-iw/zoom)*on/{frames}", center_y
    elif motion == "pan_left":
        zoom = "1.12"
        x, y = f"(iw-iw/zoom)*(1-on/{frames})", center_y
    else:  # zoom_in（含未知取值的兜底）
        zoom = "min(1.0+0.0020*on,1.14)"
        x, y = center_x, center_y

    return (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,"
        f"crop={w*2}:{h*2},"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':"
        f"d={frames}:s={w}x{h}:fps={fps}"
    )


def image_to_video(
    image_paths: list[str | Path],
    duration_per_image: float = 3.0,
    ken_burns: bool = True,
    resolution: str = "1080x1920",
    fps: int = 30,
    output_path: str | Path | None = None,
    motion: str = "auto",
    motion_index: int = 0,
) -> Path:
    """将图片序列转为视频片段。

    Args:
        image_paths: 图片路径列表
        duration_per_image: 每张图片展示时长（秒）
        ken_burns: 是否启用 Ken Burns 缓慢推拉效果
        resolution: 输出分辨率 "WxH"
        fps: 输出帧率
        motion: 运镜方式 zoom_in/zoom_out/pan_right/pan_left/auto
        motion_index: auto 轮换的起始序号（时间轴逐段渲染时传片段 index）
    """
    out = _resolve_output(output_path, "img2vid")
    w, h = (int(x) for x in resolution.split("x"))
    settings = _settings()

    # 每张图片单独生成片段，然后拼接
    clips: list[str] = []
    for i, img in enumerate(image_paths):
        clip = settings.temp_path / f"img_{i}_{uuid.uuid4().hex[:6]}.mp4"
        img_str = str(Path(img).resolve())

        if ken_burns:
            # Ken Burns：推/拉/左移/右移按序号轮换，避免所有图片同方向缓推
            vf = _build_ken_burns_vf(
                w, h, fps, duration_per_image,
                motion=motion, motion_index=motion_index + i,
            )
        else:
            vf = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", img_str,
            "-vf", vf,
            "-c:v", settings.video.default_video_codec,
            "-preset", "ultrafast",
            "-t", str(duration_per_image),
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-crf", str(settings.video.default_crf),
            str(clip),
        ]
        _run_ffmpeg(cmd)
        clips.append(str(clip))

    if len(clips) == 1:
        # 单张图片直接重命名
        import os
        os.replace(str(clips[0]), str(out))
    else:
        _concat_cut(clips, out)

    # 给无声视频添加静音音轨（避免后续拼接/播放兼容性问题）
    out = _ensure_audio_track(out)
    return out


def generate_bgm(
    duration: float,
    style: str = "ambient",
    output_path: str | Path | None = None,
) -> Path:
    """用 FFmpeg 合成简单的背景音乐（免版权，无需外部素材）。

    使用多个正弦波叠加生成和弦铺底，加低通滤波和颤音。

    Args:
        duration: 音乐时长（秒）
        style: ambient/upbeat/calm/epic，影响和弦与音色
        output_path: 输出路径
    """
    out = _resolve_output(output_path, "bgm", "m4a")

    # 不同风格的和弦频率（Hz）：根音+三音+五音
    chords = {
        "upbeat":  [261.63, 329.63, 392.00, 523.25],   # C大7 明亮
        "epic":    [196.00, 246.94, 293.66, 392.00],   # G小 厚重
        "calm":    [220.00, 277.18, 329.63],            # A小 柔和
        "ambient": [261.63, 329.63, 392.00],            # C大 中性
    }
    freqs = chords.get(style, chords["ambient"])

    # 构建 aevalsrc 表达式：多个正弦波叠加，音量递减（整体抬高，避免配乐过轻）
    terms = [f"0.18*sin(2*PI*{freqs[0]}*t)"]
    for i, f in enumerate(freqs[1:], 1):
        terms.append(f"+{0.12 / i:.3f}*sin(2*PI*{f}*t)")
    expr = "".join(terms)

    # 颤音频率随风格变化
    tremolo_freq = "4" if style == "upbeat" else "2.5"

    # 淡入淡出时长
    fade_dur = min(2.0, duration / 4)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"aevalsrc={expr}:s=44100:d={duration}",
        "-af",
        f"lowpass=f=2000,tremolo=f={tremolo_freq}:d=0.3,"
        f"afade=t=in:st=0:d={fade_dur},afade=t=out:st={duration - fade_dur}:d={fade_dur}",
        "-c:a", "aac", "-b:a", "128k",
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out


def mix_audio(
    video_path: str | Path,
    audio_path: str | Path,
    bgm_volume: float = 0.15,
    output_path: str | Path | None = None,
) -> Path:
    """将背景音乐混入视频。

    视频本身有音轨则与 BGM 混合，没有则直接用 BGM（循环到视频长度）。
    - 优先用 amix normalize=0，避免按输入数自动衰减导致配乐过轻；
    - 若当前 ffmpeg 不支持 normalize（<4.4），自动退回普通 amix，
      并把音量翻倍补偿归一化衰减，保证旧版本也能听到声音。
    """
    out = _resolve_output(output_path, "mix")
    has_audio = False
    with contextlib.suppress(Exception):
        has_audio = probe_video(video_path).has_audio

    def _build_cmd(vol: float, normalize: bool):
        if has_audio:
            norm = ":normalize=0" if normalize else ""
            fc = (
                f"[1:a]volume={vol:.3f}[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0{norm}[aout]"
            )
        else:
            fc = f"[1:a]volume={vol:.3f},aresample=44100[aout]"
        return [
            "ffmpeg", "-y",
            "-i", str(Path(video_path).resolve()),
            "-stream_loop", "-1", "-i", str(Path(audio_path).resolve()),
            "-filter_complex", fc,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(out),
        ]

    try:
        _run_ffmpeg(_build_cmd(bgm_volume, normalize=True))
    except FFmpegError as e:
        logger.warning("ffmpeg.mix_audio.normalize_fallback", error=str(e)[:120])
        _run_ffmpeg(_build_cmd(min(1.0, bgm_volume * 2), normalize=False))
    return out


def extract_audio(
    path: str | Path,
    output_path: str | Path | None = None,
    fmt: str = "wav",
) -> Path:
    """从视频提取音频。

    Args:
        fmt: wav/mp3/aac
    """
    ext = fmt
    out = _resolve_output(output_path, "audio", ext)
    codec_map = {"wav": "pcm_s16le", "mp3": "libmp3lame", "aac": "aac"}
    cmd = [
        "ffmpeg", "-y",
        "-i", str(Path(path).resolve()),
        "-vn",
        "-acodec", codec_map.get(fmt, "pcm_s16le"),
        str(out),
    ]
    _run_ffmpeg(cmd)
    return out
