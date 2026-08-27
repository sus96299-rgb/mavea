"""MAVEA MCP Server。

使用 MCP v2 的 MCPServer API，注册全部视频处理工具。
支持 stdio（Claude Desktop/Cursor 本地调用）和 streamable-http（远程调用）两种传输。

用法：
    mavea-mcp                              # 默认 stdio
    mavea-mcp --transport sse --port 8765  # HTTP 模式
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import structlog
from mcp.server import MCPServer

from mavea.config import get_settings
from mavea.mcp.tools import analysis_tools, ffmpeg_tools, media_tools
from mavea.progress import report_progress

logger = structlog.get_logger(__name__)

# 创建 MCP Server 实例
mcp = MCPServer("mavea-mcp", instructions="MAVEA 视频剪辑工具集，提供视频裁剪、拼接、字幕、转场、BGM、图片转视频等能力。")


# ==================== 工具注册 ====================

@mcp.tool()
def get_video_info(path: str) -> dict[str, Any]:
    """获取视频元信息：分辨率、帧率、时长、编码、音轨。"""
    return ffmpeg_tools.get_video_info(path)


@mcp.tool()
def cut_clip(path: str, start: float, end: float, output_path: str | None = None) -> dict[str, Any]:
    """裁剪视频片段。start/end 为秒数。"""
    return ffmpeg_tools.cut_clip(path, start, end, output_path)


@mcp.tool()
def concat_videos(
    paths: list[str],
    transition: str = "cut",
    transition_duration: float = 0.5,
    target_resolution: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """拼接多个视频，支持 cut/fade/dissolve/wipe/zoom/slide 转场。"""
    return ffmpeg_tools.concat_videos(paths, transition, transition_duration, target_resolution, output_path)


@mcp.tool()
def add_subtitle(
    path: str,
    srt_path: str,
    style: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """烧录 SRT 字幕文件到视频。style 为 FFmpeg force_style 字符串。"""
    return ffmpeg_tools.add_subtitle(path, srt_path, style, output_path)


@mcp.tool()
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
    """在视频上叠加文字贴纸。position: top/center/bottom/top_left/top_right/bottom_left/bottom_right。"""
    return ffmpeg_tools.add_text_overlay(path, text, start, end, font_size, font_color, position, output_path)


@mcp.tool()
def add_bgm(
    path: str,
    audio_path: str,
    volume: float = 0.15,
    loop: bool = True,
    keep_original_audio: bool = True,
    output_path: str | None = None,
) -> dict[str, Any]:
    """添加背景音乐。keep_original_audio=True 时混音保留人声。"""
    return ffmpeg_tools.add_bgm(path, audio_path, volume, loop, keep_original_audio, output_path)


@mcp.tool()
def image_to_video(
    image_paths: list[str],
    duration_per_image: float = 3.0,
    ken_burns: bool = True,
    resolution: str = "1080x1920",
    fps: int = 30,
    output_path: str | None = None,
) -> dict[str, Any]:
    """将图片序列转为视频片段，支持 Ken Burns 推拉效果。resolution 如 1080x1920。"""
    return media_tools.image_to_video(image_paths, duration_per_image, ken_burns, resolution, fps, output_path)


@mcp.tool()
def extract_audio(path: str, format: str = "wav", output_path: str | None = None) -> dict[str, Any]:
    """从视频提取音频，支持 wav/mp3/aac。"""
    return media_tools.extract_audio(path, format, output_path)


@mcp.tool()
def transcribe_audio(audio_path: str, model_size: str = "base", language: str = "zh") -> dict[str, Any]:
    """音频转文字（Whisper）。model_size: tiny/base/small/medium/large-v3。"""
    return media_tools.transcribe_audio(audio_path, model_size, language)


@mcp.tool()
def detect_scenes(path: str, threshold: float = 27.0, min_scene_len: float = 0.6) -> dict[str, Any]:
    """检测视频镜头边界。threshold 越低越敏感（vlog建议20-25，快剪27-30）。"""
    return analysis_tools.detect_scenes_tool(path, threshold, min_scene_len)


@mcp.tool()
def create_video_from_timeline(
    plan_json: str,
    material_map: dict[str, str],
    output_path: str | None = None,
    bgm_file: str | None = None,
    bgm_query: str | None = None,
    beat_sync: bool = False,
    lyric_mode: str = "off",
    lrc_path: str | None = None,
    beauty: bool = False,
) -> dict[str, Any]:
    """一站式时间轴渲染：输入剪辑方案JSON和素材映射，自动完成全部剪辑步骤。

    plan_json 是 TimelinePlan 的 JSON 字符串。
    material_map 是 material_id 到文件路径的映射。
    简单场景可以只调这一个工具。
    """
    try:
        from mavea.mcp.tools._helpers import make_result, resolve_input_path
        from mavea.video import ffmpeg as ff

        plan_dict = json.loads(plan_json)
        settings = get_settings()

        _segs_all = plan_dict.get("segments", [])
        norm_style0 = plan_dict.get("bgm_style") or "ambient"
        norm_style0 = norm_style0 if norm_style0 in ("upbeat", "epic", "calm", "ambient") else "ambient"
        target_dur = float(
            plan_dict.get("target_duration")
            or sum(float(s["source_end"]) - float(s.get("source_start", 0)) for s in _segs_all)
        )

        # v6：卡点 / Whisper 歌词都需要在渲染前先拿到音乐
        pre_bgm = None
        if (beat_sync or lyric_mode == "whisper") and _segs_all:
            pre_bgm, _ = _resolve_bgm_once(ff, norm_style0, bgm_file, bgm_query, target_dur)

        # v6：自动卡点——检测节拍并把片段边界吸附到节拍上
        if beat_sync and _segs_all and pre_bgm:
            from mavea.editing import effects
            report_progress("executor", "自动卡点：正在检测节拍…", 0.06)
            beats = effects.detect_beats(pre_bgm, max_seconds=target_dur + 2)
            if beats:
                durs = [float(s["source_end"]) - float(s.get("source_start", 0)) for s in _segs_all]
                new_durs = effects.snap_durations_to_beats(durs, beats)
                for s, nd in zip(_segs_all, new_durs, strict=False):
                    src0 = Path(material_map[s["material_id"]])
                    is_img = src0.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    s0 = float(s.get("source_start", 0))
                    s["source_start"] = 0.0 if is_img else s0
                    s["source_end"] = round((0.0 if is_img else s0) + nd, 2)
                report_progress("executor", f"检测到 {len(beats)} 个节拍，剪辑点已对齐", 0.09)
            else:
                report_progress("executor", "未检测到明显节拍，按原节奏剪辑", 0.09)

        # 逐步执行时间轴
        temp_clips: list[Path] = []

        # 清理上一轮可能残留的片段文件（Windows 下 rename 不覆盖已存在文件）
        for old in settings.temp_path.glob("timeline_seg_*.mp4"):
            with contextlib.suppress(OSError):
                old.unlink()

        for seg in plan_dict.get("segments", []):
            mat_id = seg["material_id"]
            if mat_id not in material_map:
                return make_result(False, error=f"素材 {mat_id} 不在 material_map 中")

            src = resolve_input_path(material_map[mat_id])
            clip_path = settings.temp_path / f"timeline_seg_{seg['index']:03d}.mp4"
            duration = seg["source_end"] - seg["source_start"]
            _segs = plan_dict.get("segments", [])
            _total = max(len(_segs), 1)
            _is_img = src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            report_progress(
                "executor",
                f"渲染片段 {seg['index'] + 1}/{_total}（{'图片' if _is_img else '视频'}，{duration:.1f}s）",
                0.1 + 0.7 * (seg["index"] / _total),
            )

            # 判断素材类型：图片用 image_to_video，视频用 cut_clip
            ext = src.suffix.lower()
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            if ext in image_exts:
                out = ff.image_to_video(
                    [src],
                    duration_per_image=duration,
                    ken_burns=True,
                    resolution=plan_dict.get("aspect_ratio", "1080x1920"),
                    output_path=clip_path,
                    motion="auto",
                    motion_index=int(seg["index"]),
                )
            else:
                out = ff.cut_clip(src, seg["source_start"], seg["source_end"], clip_path)

            # 字幕（用临时文件，避免 FFmpeg 读写同一文件）
            if seg.get("subtitle"):
                srt_path = settings.temp_path / f"seg_{seg['index']:03d}.srt"
                duration = seg["source_end"] - seg["source_start"]
                srt_content = (
                    f"1\n00:00:00,000 --> {_sec_to_srt_time(duration)}\n"
                    f"{seg['subtitle']}\n"
                )
                srt_path.write_text(srt_content, encoding="utf-8")
                sub_tmp = settings.temp_path / f"timeline_seg_{seg['index']:03d}_sub.mp4"
                out = ff.add_subtitle(out, srt_path, output_path=sub_tmp)
                # 替换原始片段
                import os
                os.replace(str(out), str(clip_path))
                out = clip_path

            # 文字贴纸（同样用临时文件）
            if seg.get("text_overlay"):
                txt_tmp = settings.temp_path / f"timeline_seg_{seg['index']:03d}_txt.mp4"
                _overlay_style = seg.get("text_overlay_style") or {}
                out = ff.add_text_overlay(
                    out, seg["text_overlay"], 0, duration,
                    font_size=int(_overlay_style.get("font_size", 48)),
                    font_color=_overlay_style.get("color", "white"),
                    position=_overlay_style.get("position", "center"),
                    output_path=txt_tmp,
                )
                import os
                os.replace(str(out), str(clip_path))
                out = clip_path

            temp_clips.append(out)

        # 拼接
        if not temp_clips:
            return make_result(False, error="没有可拼接的片段")

        transition = plan_dict.get("style", "")
        # 默认用 cut（硬切），避免 xfade 转场重叠导致总时长缩短
        trans_type = "fade" if "smooth" in transition or "cinematic" in transition else "cut"
        final = ff.concat_videos(
            [str(c) for c in temp_clips],
            transition=trans_type,
            target_resolution=plan_dict.get("aspect_ratio"),
            output_path=output_path,
        )

        # 背景音乐：默认一定配乐。本地/在线快速尝试，失败则离线合成，混音后校验音轨。
        bgm_style = plan_dict.get("bgm_style") or "ambient"
        norm_style = bgm_style if bgm_style in ("upbeat", "epic", "calm", "ambient") else "ambient"
        try:
            final = _attach_music(ff, final, norm_style, bgm_file, bgm_query,
                                  pre_bgm_path=pre_bgm)
        except Exception as be:
            # 配乐任何一步失败都不应让整片无声：兜底离线合成再混一次
            logger.warning("mcp.bgm_retry", error=str(be))
            report_progress("executor", f"配乐重试（离线合成）: {str(be)[:60]}", 0.96)
            try:
                probe = ff.probe_video(final)
                synth = ff.generate_bgm(duration=probe.duration + 1, style=norm_style)
                final = ff.mix_audio(final, synth, bgm_volume=0.7)
                report_progress("executor", "已用离线合成配乐兜底", 0.99)
            except Exception as be2:
                logger.error("mcp.bgm_failed_final", error=str(be2))

        # v8：AI 配音（edge-tts 旁白，自动压低 BGM 突出人声）
        voice_text = plan_dict.get("voiceover_text")
        if voice_text:
            try:
                from mavea.audio.tts import synthesize_voice
                from mavea.editing import effects as _fx
                report_progress("executor", f"AI 配音生成中：{str(voice_text)[:20]}…", 0.97)
                voice = synthesize_voice(str(voice_text), settings.temp_path / "tts")
                if voice is not None:
                    mixed_voice = _fx.mix_voiceover(final, voice)
                    if mixed_voice is not None:
                        final = str(mixed_voice)
                        report_progress("executor", "AI 旁白已混入（BGM 自动压低）", 0.98)
            except Exception as ve:
                logger.warning("mcp.voiceover_skip", error=str(ve))

        # v6：自动歌词字幕（.lrc 上传 或 Whisper 识别人声），烧在画面顶部
        if lyric_mode in ("lrc", "whisper"):
            try:
                _lyric = _apply_lyrics(ff, final, settings, lyric_mode, lrc_path, pre_bgm)
                if _lyric:
                    final = _lyric
            except Exception as le:
                logger.warning("mcp.lyric_skip", error=str(le))
                report_progress("executor", f"歌词字幕已跳过: {str(le)[:50]}", 0.995)

        # v6：轻度美颜（磨皮+提亮），失败不影响成片
        if beauty:
            try:
                from mavea.editing import effects
                report_progress("executor", "正在施加美颜滤镜…", 0.997)
                _b = effects.beautify(final)
                if _b is not None:
                    final = str(_b)
            except Exception as be:
                logger.warning("mcp.beauty_skip", error=str(be))

        return make_result(True, {"output_path": str(final)})

    except Exception as e:
        logger.error("mcp.create_timeline.failed", error=str(e))
        return make_result(False, error=str(e))


def _resolve_bgm_once(ff, norm_style: str, bgm_file, bgm_query, duration: float):
    """渲染前提前选曲（卡点/歌词识别需要）。返回 (路径|None, 来源说明)。"""
    from mavea.audio.music import resolve_music
    try:
        path, label = resolve_music(
            style=norm_style, duration=duration + 1,
            local_file=bgm_file, url_or_keyword=bgm_query,
        )
        if path is not None:
            return str(path), label
    except Exception as e:
        logger.warning("mcp.early_bgm", error=str(e))
    try:
        synth = ff.generate_bgm(duration=duration + 1, style=norm_style)
        return str(synth), "离线合成配乐"
    except Exception:
        return None, "无配乐"


def _apply_lyrics(ff, final, settings, mode: str, lrc_path, bgm_path):
    """生成歌词 SRT 并烧录到画面顶部，返回新路径；无内容返回 None。"""
    from mavea.editing import effects

    probe = ff.probe_video(final)
    srt_text = None
    if mode == "lrc":
        if lrc_path and Path(lrc_path).exists():
            raw = Path(lrc_path).read_text(encoding="utf-8", errors="ignore")
            srt_text = effects.lrc_to_srt(raw, max_duration=probe.duration)
    elif mode == "whisper":
        if not bgm_path:
            return None
        report_progress("executor", "正在识别人声生成歌词字幕…", 0.98)
        r = media_tools.transcribe_audio(bgm_path, "base", "zh")
        if r.get("success"):
            segs = (r.get("data") or {}).get("segments", [])
            srt_text = effects.segments_to_srt(segs, max_duration=probe.duration)

    if not srt_text or not srt_text.strip():
        return None
    srt_file = settings.temp_path / "lyrics.srt"
    srt_file.write_text(srt_text, encoding="utf-8")
    # 顶部居中、白字黑边，避免和底部 AI 字幕撞在一起
    style = (
        "FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=0,Alignment=8,MarginV=30"
    )
    out = ff.add_subtitle(final, srt_file, style=style)
    return str(out)


def _attach_music(ff, final, norm_style: str,
                  bgm_file: str | None = None, bgm_query: str | None = None,
                  pre_bgm_path: str | None = None):
    """给成片可靠配乐，返回带音轨的最终路径。

    选曲优先级见 audio.music.resolve_music：用户上传/链接/在线搜索 →
    本地曲库 → 免版税 → 离线合成。混音后用 ffprobe 校验确有音轨。
    pre_bgm_path 为卡点/歌词阶段提前选好的同一首曲子，避免重复下载。
    """
    from mavea.audio.music import resolve_music

    probe = ff.probe_video(final)

    if pre_bgm_path:
        bgm_path, label = pre_bgm_path, "已选曲目"
    else:
        report_progress("executor", "正在准备背景音乐…", 0.90)
        bgm_path, label = resolve_music(
            style=norm_style,
            duration=probe.duration + 1,
            local_file=bgm_file,
            url_or_keyword=bgm_query,
        )
    report_progress("executor", f"背景音乐：{label}", 0.93)

    if bgm_path is not None:
        # 真实歌曲音量适中；人声演唱不被压低
        mixed = ff.mix_audio(final, bgm_path, bgm_volume=0.85)
    else:
        report_progress("executor", "无可用音源，离线合成配乐中…", 0.94)
        synth = ff.generate_bgm(duration=probe.duration + 1, style=norm_style)
        mixed = ff.mix_audio(final, synth, bgm_volume=0.7)

    try:
        if ff.probe_video(mixed).has_audio:
            report_progress("executor", f"背景音乐已混入 ✔（{label}）", 0.99)
            return mixed
        raise RuntimeError("混音后仍无音轨")
    except Exception:
        synth = ff.generate_bgm(duration=probe.duration + 1, style=norm_style)
        return ff.mix_audio(final, synth, bgm_volume=0.7)


def _sec_to_srt_time(seconds: float) -> str:
    """秒转 SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ==================== 入口 ====================

def main():
    """MCP Server 命令行入口。"""
    parser = argparse.ArgumentParser(description="MAVEA MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输模式：stdio（本地）或 sse（HTTP）",
    )
    parser.add_argument("--host", default=None, help="SSE 监听地址")
    parser.add_argument("--port", type=int, default=None, help="SSE 端口")
    args = parser.parse_args()

    settings = get_settings()

    if args.transport == "stdio":
        logger.info("mcp.server.start", transport="stdio")
        mcp.run(transport="stdio")
    else:
        host = args.host or settings.mcp.sse_host
        port = args.port or settings.mcp.sse_port
        logger.info("mcp.server.start", transport="streamable-http", host=host, port=port)
        mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
