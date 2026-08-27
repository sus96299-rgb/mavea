"""TTS 配音：用 edge-tts（免费、无需 key，但需联网）把旁白文本合成语音。

edge-tts 是异步库，而渲染工具链是同步调用、且可能已处在某个事件循环里，
因此这里用“独立线程 + 独立事件循环”的方式同步封装，避免嵌套 loop 报错。
任何失败都返回 None，由上层决定是否跳过配音（绝不阻断成片）。
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# 常用中文音色
VOICES = {
    "female": "zh-CN-XiaoxiaoNeural",   # 女声·温暖（默认）
    "male": "zh-CN-YunxiNeural",        # 男声·自然
    "story": "zh-CN-XiaoyiNeural",      # 女声·叙事
}


def synthesize_voice(
    text: str,
    out_dir: str | Path,
    voice: str = "female",
    rate: str = "+0%",
    timeout: int = 120,
) -> Path | None:
    """把文本合成为 mp3。成功返回路径，失败返回 None。

    Args:
        text: 旁白文本（建议不超过 60 字，太长会超过成片时长）。
        out_dir: 输出目录。
        voice: female/male/story 或直接给 edge-tts 音色名。
        rate: 语速，如 "-10%" 放慢、"+10%" 加快。
    """
    text = (text or "").strip()
    if not text:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
    voice_name = VOICES.get(voice, voice)

    box: dict = {}

    def _run() -> None:
        try:
            import edge_tts  # 延迟导入，没装/没网时不影响主流程

            async def _go() -> None:
                communicate = edge_tts.Communicate(text, voice_name, rate=rate)
                await communicate.save(str(out_path))

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_go())
            finally:
                loop.close()
            if out_path.exists() and out_path.stat().st_size > 0:
                box["ok"] = True
        except Exception as e:  # noqa: BLE001
            box["err"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if box.get("ok"):
        logger.info("tts.done", voice=voice_name, chars=len(text))
        return out_path
    logger.warning("tts.failed", error=box.get("err", "timeout"))
    return None
