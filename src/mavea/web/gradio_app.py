"""Gradio WebUI：上传素材 → 输入需求 → 浅色进度条 + 两路滚动日志 → 播放成片。

修复点：
1. 不使用 gr.Progress()（它会把自动进度条注入每个输出框，导致三块全是进度条、结束闪烁）。
   进度=自绘浅色进度条；实时进度/工具日志=两个白底滚动控制台，风格贴近原版 Gradio。
2. 阻塞的 ffmpeg 渲染放工作线程，界面线程每 0.2s 增量刷新，长任务不卡死、不闪烁。
3. 工具调用由 MCP 客户端实时上报，运行中跳秒，结束显示 ✅/❌ 与耗时。
4. 上传图片即时预览（Gallery）。
5. 背景音乐支持：上传本地歌曲 / 音频直链 / 歌名在线搜索，留空则按风格自动选曲。
"""
from __future__ import annotations

import contextlib
import html
import queue
import re
import shutil
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import structlog

from mavea import __version__
from mavea.agents.graph import run_pipeline
from mavea.config import get_settings
from mavea.progress import (
    PipelineCancelled,
    is_paused,
    request_cancel,
    request_pause,
    request_resume,
    reset_controls,
    set_progress_callback,
)

logger = structlog.get_logger(__name__)

STAGE_NAMES = {
    "analyzer": "1/4 素材分析",
    "planner": "2/4 剪辑规划",
    "executor": "3/4 剪辑执行",
    "evaluator": "4/4 质量评估",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}

# 浅色主题样式（白底、灰边框、深字），贴近原版 Gradio Soft 风格。
# 智能吸底：用户停在底部附近才自动滚动；一旦手动上翻看历史就不打扰，滚回底部恢复。
_AUTO_SCROLL_JS = """
() => {
  const THRESHOLD = 48;  // 距底部多少 px 内算“贴底”
  window.__maveaStick = window.__maveaStick || {};
  // scroll 事件不冒泡，必须用捕获阶段监听。
  // 按 DOM 顺序区分两个框：0=实时进度，1=工具调用日志。
  document.addEventListener('scroll', (e) => {
    const el = e.target;
    if (el && el.classList && el.classList.contains('mavea-console')) {
      const all = Array.from(document.querySelectorAll('.mavea-console'));
      const idx = all.indexOf(el);
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < THRESHOLD;
      window.__maveaStick['box' + idx] = nearBottom;
    }
  }, true);
  setInterval(() => {
    document.querySelectorAll('.mavea-console').forEach((el, idx) => {
      const key = 'box' + idx;
      const stick = window.__maveaStick[key];
      if (stick === false) return;   // 用户正在上翻看历史 → 保持位置，不强制下拉
      el.scrollTop = el.scrollHeight; // 其余情况（默认/贴底）保持最新行可见
    });
  }, 250);
}
"""

_HEAD_CSS = """
<style>
.mavea-wrap{font-family:ui-sans-serif,'Microsoft YaHei',sans-serif}
.mavea-progress-head{display:flex;justify-content:space-between;font-size:13px;margin:4px 0 6px;color:#374151}
.mavea-stage{font-weight:700;color:#4f46e5}
.mavea-track{height:12px;border-radius:8px;background:#eef2ff;overflow:hidden;border:1px solid #e0e7ff}
.mavea-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#818cf8,#6366f1);transition:width .25s ease}
.mavea-msg{font-size:12.5px;color:#6b7280;margin-top:6px;white-space:pre-wrap;word-break:break-all}
.mavea-console{height:220px;overflow-y:auto;background:#ffffff;color:#374151;
  border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;font-size:13px;line-height:1.75;
  display:flex;flex-direction:column;justify-content:flex-start}
.mavea-line{white-space:pre-wrap;word-break:break-all;margin:1px 0;color:#374151}
.mavea-line.ok{color:#15803d}.mavea-line.err{color:#b91c1c}
.mavea-line.run{color:#b45309}.mavea-line.dim{color:#6b7280}
</style>
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=False)


# 工具代码名 → 中文名（英文原名以灰色小字附注）
TOOL_CN = {
    "create_video_from_timeline": "一站式生成视频（转场+字幕+配乐）",
    "image_to_video": "图片转视频（Ken Burns 推拉）",
    "cut_clip": "裁剪视频片段",
    "concat_videos": "拼接视频",
    "add_subtitle": "烧录字幕",
    "add_text_overlay": "添加文字贴纸",
    "add_bgm": "添加背景音乐",
    "mix_audio": "混入背景音乐",
    "get_video_info": "读取视频信息",
    "detect_scenes": "检测镜头边界",
    "extract_audio": "提取音频",
    "transcribe_audio": "语音转文字",
}


def _tool_label(name: str) -> str:
    """中文为主，英文原名灰色小字附注。"""
    safe = _esc(name)
    cn = TOOL_CN.get(str(name))
    if cn:
        return f"<b>{cn}</b> <span style='color:#9ca3af;font-size:11px'>({safe})</span>"
    return f"<b>{safe}</b>"


def _progress_html(percent: float, stage_label: str, message: str, elapsed: float) -> str:
    w = max(0.0, min(1.0, percent)) * 100
    return (
        "<div class='mavea-wrap'>"
        "<div class='mavea-progress-head'>"
        f"<span class='mavea-stage'>{_esc(stage_label)}</span>"
        f"<span>{w:.0f}%　已用 {elapsed:.1f}s</span>"
        "</div>"
        f"<div class='mavea-track'><div class='mavea-fill' style='width:{w:.1f}%'></div></div>"
        f"<div class='mavea-msg'>{_esc(message)}</div>"
        "</div>"
    )


def _console_html(lines: list[str]) -> str:
    # 正常从上到下时间正序；超框由 JS 自动吸底
    body = "".join(lines[-200:])
    return f"<div class='mavea-console'>{body}</div>"


def _tool_lines(tools: list[dict], now: float) -> list[str]:
    if not tools:
        return [
            "<div class='mavea-line dim'>⏳ 当前还没有工具调用。<br>"
            "工具只在「3/4 剪辑执行」阶段运行（素材分析/规划阶段这里为空是正常的），"
            "开始后此处会实时滚动，结束显示 ✅/❌ 与耗时。</div>"
        ]
    out = []
    for t in tools:
        label = _tool_label(t["tool"])
        if t["status"] == "running":
            dt = now - t["t0"]
            out.append(f"<div class='mavea-line run'>⏳ {label} 运行中… {dt:.1f}s</div>")
        elif t["status"] == "success":
            out.append(
                f"<div class='mavea-line ok'>✅ {label} 完成"
                f"（{t.get('duration_ms', 0) / 1000:.2f}s）</div>"
            )
        else:
            detail = _esc(t.get("detail") or "失败")
            out.append(f"<div class='mavea-line err'>❌ {label} 失败：{detail}</div>")
    return out


def _file_path(f) -> str:
    if isinstance(f, str):
        return f
    for attr in ("name", "path", "orig_name"):
        v = getattr(f, attr, None)
        if isinstance(v, str) and v:
            return v
    if isinstance(f, dict):
        return f.get("path") or f.get("name") or ""
    return str(f)


def toggle_pause():
    """暂停/继续切换：在阶段与片段边界协作式停住，不强杀正在跑的 ffmpeg。"""
    if is_paused():
        request_resume()
        return gr.update(value="⏸ 暂停生成", variant="secondary")
    request_pause()
    return gr.update(value="▶ 继续生成", variant="primary")


def stop_generation():
    """全部停止：请求取消流水线（会在当前片段结束后中断）。"""
    request_cancel()
    return gr.update(value="⏹ 正在停止…")


def reset_run_buttons():
    """每次开跑时把两个控制按钮恢复为初始状态。"""
    return (
        gr.update(value="⏸ 暂停生成", variant="secondary", interactive=True),
        gr.update(value="⏹ 全部停止", variant="stop", interactive=True),
    )


def preview_images(files):
    """上传后立即预览其中的图片。"""
    if not files:
        return []
    items = []
    for f in files:
        p = _file_path(f)
        if p and Path(p).suffix.lower() in IMAGE_EXTS:
            items.append((p, Path(p).name))
    return items


def _copy_upload(src_path: str, settings) -> str:
    src = Path(src_path)
    dest = settings.temp_path / "uploads" / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
        return str(dest)
    except Exception:
        return str(src)


def _extract_song_query(prompt: str) -> str | None:
    """从“背景音乐用:安和桥 / BGM：xxx”里识别歌名，用于在线搜索。"""
    if not prompt or "背景音乐" not in prompt and "bgm" not in prompt.lower():
        return None
    idx = -1
    low = prompt.lower()
    for key in ("背景音乐", "bgm"):
        i = low.find(key) if key == "bgm" else prompt.find(key)
        if i >= 0:
            idx = i + len(key)
            break
    if idx < 0:
        return None
    tail = re.sub(r"^[\s用是:：]+", "", prompt[idx:])
    song = re.split(r"[，。,.；;\n!！?？]", tail)[0].strip()
    if 2 <= len(song) <= 20:
        return song
    return None


class DisplayClock:
    """显示时钟：暂停期间冻结，恢复后从冻结值接续，多次暂停累计扣除。

    与后台流水线状态解耦——界面只在收到 progress 的 paused/resumed 事件时
    调用 pause()/resume()；停止时不冻结（时间继续走到线程真正退出）。
    """

    def __init__(self, start: float):
        self.start = start
        self.paused_total = 0.0
        self.pause_wall: float | None = None

    def pause(self, now: float) -> None:
        if self.pause_wall is None:
            self.pause_wall = now

    def resume(self, now: float) -> None:
        if self.pause_wall is not None:
            self.paused_total += now - self.pause_wall
            self.pause_wall = None

    def read(self, now: float) -> float:
        """当前应显示的已用秒数。"""
        extra = self.paused_total
        if self.pause_wall is not None:
            extra += now - self.pause_wall
        return max(0.0, now - self.start - extra)

    def clock_now(self, now: float) -> float:
        """工具小秒表用的虚拟当前时刻（与 read 同一时钟基准）。"""
        return self.start + self.read(now)


def run_editing(files, prompt: str, max_iter: int, bgm_file, music_query,
                beat_sync: bool = False, beauty: bool = False,
                lyric_mode: str = "off", lrc_file=None, ai_enhance: bool = True):
    """Gradio 回调（普通生成器）：工作线程跑流水线，主线程轮询事件并增量重绘。"""
    reset_controls()  # 每次开跑先清除上一次的暂停/停止状态
    empty_status = _console_html(["<div class='mavea-line dim'>等待开始…</div>"])
    empty_tools = _console_html(_tool_lines([], time.time()))
    if not files:
        yield _progress_html(0, "未开始", "请先上传至少一个素材文件", 0), empty_status, None, empty_tools, ""
        return
    if not prompt or not prompt.strip():
        yield _progress_html(0, "未开始", "请输入剪辑需求", 0), empty_status, None, empty_tools, ""
        return

    settings = get_settings()
    material_paths = [_copy_upload(_file_path(f), settings) for f in files]

    # 用户自定义背景音乐
    custom_bgm_path = None
    if bgm_file is not None:
        bf = bgm_file[0] if isinstance(bgm_file, (list, tuple)) and bgm_file else bgm_file
        bf_path = _file_path(bf)
        if bf_path and Path(bf_path).suffix.lower() in AUDIO_EXTS:
            custom_bgm_path = _copy_upload(bf_path, settings)
    custom_bgm_query = music_query.strip() if music_query and music_query.strip() else None
    if not custom_bgm_query:
        custom_bgm_query = _extract_song_query(prompt)
        if custom_bgm_query:
            logger.info("webui.auto_song_query", song=custom_bgm_query)

    # 歌词文件（.lrc/.txt）
    lrc_path = None
    if lyric_mode == "lrc" and lrc_file is not None:
        lf = lrc_file[0] if isinstance(lrc_file, (list, tuple)) and lrc_file else lrc_file
        lf_path = _file_path(lf)
        if lf_path:
            lrc_path = _copy_upload(lf_path, settings)
    if lyric_mode not in ("off", "lrc", "whisper"):
        lyric_mode = "off"

    events: queue.Queue[tuple] = queue.Queue()

    def _on_event(event: dict):
        with contextlib.suppress(Exception):
            events.put_nowait(event)

    set_progress_callback(_on_event)
    result_box: dict = {}

    def _worker():
        try:
            import asyncio
            try:
                from mavea.mcp import client as _mcp_client
                _mcp_client._client = None
            except Exception:
                pass
            state = asyncio.run(run_pipeline(
                material_paths=material_paths,
                user_prompt=prompt,
                max_iterations=max_iter,
                custom_bgm_path=custom_bgm_path,
                custom_bgm_query=custom_bgm_query,
                beat_sync=bool(beat_sync),
                beauty=bool(beauty),
                lyric_mode=lyric_mode,
                lrc_path=lrc_path,
                ai_enhance=bool(ai_enhance),
            ))
            result_box["state"] = state
        except PipelineCancelled:
            result_box["cancelled"] = True
        except Exception as e:  # noqa: BLE001
            result_box["error"] = (e, traceback.format_exc())
        events.put({"kind": "__done__"})

    threading.Thread(target=_worker, daemon=True).start()

    start = time.time()
    if custom_bgm_path:
        music_hint = f"使用上传歌曲: {Path(custom_bgm_path).name}"
    elif custom_bgm_query:
        music_hint = f"在线选曲: {custom_bgm_query[:20]}"
    else:
        music_hint = "默认自动配乐"
    status_lines: list[str] = [
        f"<div class='mavea-line dim'>[{0:6.1f}s] 🚀 启动 MAVEA 流水线（{music_hint}）…</div>"
    ]
    tool_entries: list[dict] = []
    percent = 0.02
    stage_label = "启动中"
    message = "正在初始化…"

    # 显示时钟：暂停期间冻结，恢复后接续（详见 DisplayClock）
    clock = DisplayClock(start)

    while True:
        try:
            ev = events.get(timeout=0.2)
        except queue.Empty:
            now = time.time()
            se = clock.read(now)
            yield (_progress_html(percent, stage_label, message, se),
                   _console_html(status_lines), None,
                   _console_html(_tool_lines(tool_entries, clock.clock_now(now))), "")
            continue

        if ev.get("kind") == "__done__":
            break

        now = time.time()
        se = clock.read(now)
        kind = ev.get("kind")
        if kind == "pause_requested":
            message = "⏸ 暂停中：当前步骤收尾后暂停，点继续可恢复…"
            status_lines.append(
                f"<div class='mavea-line dim'>[{se:6.1f}s] ⏸ 暂停请求已收到，等待当前步骤收尾…</div>"
            )
        elif kind == "paused":
            clock.pause(time.time())
            message = "⏸ 已暂停，点“▶ 继续生成”恢复"
            status_lines.append(
                f"<div class='mavea-line'>[{clock.read(time.time()):6.1f}s] ⏸ 已暂停，计时已冻结</div>"
            )
        elif kind == "resumed":
            clock.resume(time.time())
            message = "已恢复生成"
            status_lines.append(
                f"<div class='mavea-line'>[{clock.read(time.time()):6.1f}s] ▶ 已继续</div>"
            )
        elif kind == "cancel_requested":
            message = "⏹ 正在停止：当前步骤收尾后退出（规划/评估阶段最多约 1 分钟）…"
            status_lines.append(
                f"<div class='mavea-line err'>[{se:6.1f}s] ⏹ 停止请求已收到，等待当前步骤收尾…</div>"
            )
        elif kind == "stage":
            stage_label = STAGE_NAMES.get(ev.get("stage"), ev.get("stage", ""))
            message = ev.get("message", "")
            percent = float(ev.get("percent", percent))
            status_lines.append(
                f"<div class='mavea-line'>[{se:6.1f}s] "
                f"<b>{_esc(stage_label)}</b> · {_esc(message)}</div>"
            )
        elif kind == "tool":
            status_ = ev.get("status")
            if status_ == "start":
                tool_entries.append({
                    "tool": ev.get("tool", "?"), "status": "running",
                    "t0": now, "duration_ms": None, "detail": None,
                })
                status_lines.append(
                    f"<div class='mavea-line run'>[{se:6.1f}s] "
                    f"🔧 调用工具 {_esc(ev.get('tool'))} …</div>"
                )
            else:
                for t in reversed(tool_entries):
                    if t["tool"] == ev.get("tool") and t["status"] == "running":
                        t["status"] = status_
                        t["duration_ms"] = ev.get("duration_ms") or int((now - t["t0"]) * 1000)
                        t["detail"] = ev.get("detail")
                        break

        yield (_progress_html(percent, stage_label, message, clock.read(now)),
               _console_html(status_lines), None,
               _console_html(_tool_lines(tool_entries, clock.clock_now(now))), "")

    set_progress_callback(None)
    elapsed = clock.read(time.time())

    if result_box.get("cancelled"):
        yield (
            _progress_html(percent, "已停止", "⏹ 已按你的要求停止本次生成", elapsed),
            _console_html(status_lines + [
                "<div class='mavea-line err'>⏹ 已停止：本次生成被手动中断，未产出成片。</div>"
            ]),
            None, _console_html(_tool_lines(tool_entries, time.time())), "",
        )
        return

    if "error" in result_box:
        e, tb = result_box["error"]
        logger.error("webui.failed", error=str(e))
        yield (_progress_html(percent, "异常", f"运行异常: {e}", elapsed),
               _console_html(status_lines + [f"<div class='mavea-line err'>❌ 异常：{_esc(e)}</div>"]),
               None, _console_html(_tool_lines(tool_entries, time.time())), tb)
        return

    state = result_box.get("state")

    def _get(key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    exec_result = _get("execution_result")
    eval_result = _get("evaluation_result")
    edit_plan = _get("edit_plan")

    if exec_result and exec_result.success:
        output_path = exec_result.output_path
        report = ["<div class='mavea-line ok'>🎉 剪辑完成！</div>"]
        report.append(f"<div class='mavea-line'>📁 输出文件：{_esc(output_path)}</div>")
        report.append(f"<div class='mavea-line'>🔧 工具调用：{exec_result.tool_call_count} 次</div>")
        report.append(f"<div class='mavea-line'>⏱️ 总耗时：{elapsed:.1f}s</div>")
        if edit_plan:
            report.append(
                f"<div class='mavea-line'>📋 方案：用了 {len(edit_plan.segments)} 个素材片段，"
                f"时长 {edit_plan.estimated_duration():.1f}s，配乐风格：{_esc(edit_plan.bgm_style or '无')}</div>"
            )
        if eval_result:
            report.append(f"<div class='mavea-line'>⭐ 质量评分：{eval_result.overall}/5.0</div>")
            if eval_result.issues:
                report.append(
                    "<div class='mavea-line err'>⚠️ 问题："
                    + _esc("；".join(eval_result.issues[:3])) + "</div>"
                )
        final_tools = [
            f"<div class='mavea-line {'ok' if c.status.value == 'success' else 'err'}'>"
            f"{'✅' if c.status.value == 'success' else '❌'} {_tool_label(c.tool_name)}"
            f"（{c.duration_ms / 1000:.2f}s）</div>"
            for c in exec_result.tool_calls
        ]
        yield (_progress_html(1.0, "完成", f"成片已生成：{output_path}", elapsed),
               _console_html(status_lines + report), output_path,
               _console_html(final_tools or _tool_lines(tool_entries, time.time())), "")
    else:
        errors = _get("errors", ["未知错误"])
        error_text = "; ".join(str(e) for e in errors)
        yield (_progress_html(percent, "失败", error_text, elapsed),
               _console_html(status_lines + [f"<div class='mavea-line err'>❌ 剪辑失败：{_esc(error_text)}</div>"]),
               None, _console_html(_tool_lines(tool_entries, time.time())),
               traceback.format_exc())


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面。"""
    with gr.Blocks(title="MAVEA - 多Agent智能视频剪辑助手") as demo:
        gr.HTML(_HEAD_CSS)
        gr.Markdown(f"""
        # 🎬 MAVEA
        **Multi-Agent Video Editing Assistant** v{__version__}
        上传图片/视频 + 输入需求（如“11张图做15秒，配伤感流行音乐”），AI 自动分析→规划→剪辑→配乐→评估。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(
                    label="上传素材（视频/图片，可多选）",
                    file_count="multiple",
                    file_types=["video", "image"],
                )
                gallery = gr.Gallery(label="已上传图片预览", columns=3, height=180)
                prompt = gr.Textbox(
                    label="剪辑需求",
                    placeholder="用精准提示词效果更好",
                    lines=2,
                )
                with gr.Accordion("🎵 背景音乐（可选，默认自动配乐）", open=False):
                    bgm_file = gr.File(
                        label="上传本地歌曲（mp3/m4a/wav，优先级最高）",
                        file_types=["audio"],
                    )
                    music_query = gr.Textbox(
                        label="或粘贴音频直链 / 输入歌名·歌手在线搜索",
                        placeholder="直链：https://xxx/song.mp3　或搜索：伤感 流行 / 歌名 歌手",
                        lines=1,
                    )
                    gr.Markdown(
                        "<span style='color:#9ca3af;font-size:12px'>"
                        "留空=按需求风格自动选曲；在线搜索用公开试听片段（约30秒，适合15秒视频），仅供私人使用。</span>"
                    )
                with gr.Accordion("✨ 进阶效果（可选）", open=False):
                    ai_enhance = gr.Checkbox(
                        label="🤖 AI 增强导演（自动决定卡点/美颜/AI配音，会多花少量 token）",
                        value=True,
                    )
                    beat_sync = gr.Checkbox(
                        label="🎵 自动卡点（按背景音乐节拍切换画面）", value=False,
                    )
                    beauty = gr.Checkbox(label="✨ 轻度美颜（磨皮+提亮，全局滤镜）", value=False)
                    lyric_mode = gr.Radio(
                        label="🎤 歌词字幕（显示在画面顶部）",
                        choices=[("关闭", "off"), ("上传 .lrc 歌词文件", "lrc"),
                                 ("AI 识别人声（Whisper，较慢）", "whisper")],
                        value="off",
                    )
                    lrc_file = gr.File(
                        label="歌词文件（选“上传 .lrc”时使用，.lrc/.txt）",
                        file_types=[".lrc", ".txt"],
                    )
                    gr.Markdown(
                        "<span style='color:#9ca3af;font-size:12px'>"
                        "卡点会自动检测鼓点；歌词识别对纯音乐/伴奏无效，"
                        "想要准确歌词建议上传同名 .lrc。</span>"
                    )
                max_iter = gr.Slider(
                    label="最大迭代轮数（1=最快，3=质量优先）",
                    minimum=1, maximum=5, value=1, step=1,
                )
                btn = gr.Button("🎬 开始剪辑", variant="primary", size="lg")
                with gr.Row():
                    pause_btn = gr.Button("⏸ 暂停生成", variant="secondary")
                    stop_btn = gr.Button("⏹ 全部停止", variant="stop")

            with gr.Column(scale=1):
                output_video = gr.Video(label="成片")
                gr.Markdown("<div style='margin-top:10px;font-weight:700;color:#374151'>📊 进度</div>")
                progress_box = gr.HTML(
                    value=_progress_html(0, "未开始", "上传素材并点击“开始剪辑”", 0),
                    show_label=False,
                )
                gr.Markdown("<div style='margin-top:10px;font-weight:700;color:#374151'>📝 实时进度（滚动）</div>")
                status_box = gr.HTML(
                    value=_console_html(["<div class='mavea-line dim'>等待开始…</div>"]),
                    show_label=False,
                )
                gr.Markdown("<div style='margin-top:10px;font-weight:700;color:#374151'>🔧 工具调用日志（滚动）</div>")
                tool_box = gr.HTML(
                    value=_console_html(_tool_lines([], time.time())),
                    show_label=False,
                )

        with gr.Accordion("错误详情", open=False):
            error_detail = gr.Textbox(label="Traceback", lines=4)

        # 日志超框时自动吸底
        demo.load(js=_AUTO_SCROLL_JS)
        files.change(fn=preview_images, inputs=files, outputs=gallery)
        pause_btn.click(fn=toggle_pause, outputs=pause_btn)
        stop_btn.click(fn=stop_generation, outputs=stop_btn)
        btn.click(
            fn=run_editing,
            inputs=[files, prompt, max_iter, bgm_file, music_query,
                    beat_sync, beauty, lyric_mode, lrc_file, ai_enhance],
            outputs=[progress_box, status_box, output_video, tool_box, error_detail],
        )
        # 开跑时把暂停/停止按钮复位
        btn.click(fn=reset_run_buttons, outputs=[pause_btn, stop_btn])

    return demo


def main():
    """启动 Gradio WebUI。"""
    settings = get_settings()
    demo = build_ui()
    demo.launch(
        server_name=settings.gradio.host,
        server_port=settings.gradio.port,
        share=settings.gradio.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
