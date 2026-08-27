"""背景音乐提供器：本地素材优先，其次快速尝试免版权音乐源，最后交给 FFmpeg 离线合成。

音乐来源：Kevin MacLeod (Incompetech) 的 CC BY 4.0 免版税音乐。
下载缓存在 workspace/bgm/ 目录。

设计原则（修复“成片无声”）：
- 在线下载只做“尽力而为”的快速尝试（短超时、少量候选），绝不长时间卡住渲染；
- 任何失败都返回 None，由调用方用 FFmpeg 离线合成配乐，保证成片一定有声音。
"""
from __future__ import annotations

import random
from pathlib import Path

import structlog

from mavea.config import get_settings

logger = structlog.get_logger(__name__)

# 免版税音乐直链（Kevin MacLeod / Incompetech，CC BY 4.0）
_FREE_MUSIC_URLS: dict[str, list[str]] = {
    "upbeat": [
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Carefree.mp3",
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Happy%20Rock.mp3",
    ],
    "calm": [
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Heartwarming.mp3",
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Quiet%20Moments.mp3",
    ],
    "epic": [
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Impact%20Moderato.mp3",
    ],
    "ambient": [
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Cipher.mp3",
        "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Deep%20Blue.mp3",
    ],
}

_STYLE_ALIASES: dict[str, str] = {
    "轻快": "upbeat", "活泼": "upbeat", "欢快": "upbeat", "动感": "upbeat",
    "安静": "calm", "舒缓": "calm", "温馨": "calm", "柔和": "calm",
    "史诗": "epic", "大气": "epic", "震撼": "epic", "紧张": "epic",
    "氛围": "ambient", "电子": "ambient", "科技": "ambient", "科幻": "ambient",
}

_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}


def _normalize_style(style: str) -> str:
    """把各种风格描述归一化到 upbeat/calm/epic/ambient。"""
    style_lower = (style or "").lower().strip()
    if style_lower in _FREE_MUSIC_URLS:
        return style_lower
    for alias, canonical in _STYLE_ALIASES.items():
        if alias in style_lower:
            return canonical
    return "ambient"


def _get_bgm_dir() -> Path:
    """获取 BGM 缓存目录。"""
    settings = get_settings()
    bgm_dir = settings.workspace_path / "bgm"
    bgm_dir.mkdir(parents=True, exist_ok=True)
    return bgm_dir


def _find_local_track(style: str) -> Path | None:
    """在本地 BGM 目录查找**风格匹配**的音乐。

    用户可把 mp3/m4a/wav 放到 workspace/bgm/，只有文件名含风格名
    （upbeat/calm/epic/ambient，如 upbeat_01.mp3）才会被选用。

    注意：风格匹配不到时必须返回 None，让调用方继续走在线下载；
    绝不能随机回退到任意旧音频——否则用户用过一次的歌会永久污染
    之后所有选曲（例如要"动感"配乐却放出上一次的舒缓歌曲）。
    """
    bgm_dir = _get_bgm_dir()
    audio_files = [f for f in bgm_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in _AUDIO_EXTS]
    if not audio_files:
        return None
    matched = [f for f in audio_files if style in f.stem.lower()]
    if not matched:
        logger.info("music.local_no_style_match", style=style, dir=str(bgm_dir))
        return None
    return random.choice(matched)


def _download_track(style: str) -> Path | None:
    """快速尝试从免版权音乐源下载一首；网络不佳时尽快放弃（离线合成兜底）。"""
    try:
        import httpx
    except Exception:
        return None

    urls = _FREE_MUSIC_URLS.get(style, _FREE_MUSIC_URLS["ambient"])
    bgm_dir = _get_bgm_dir()
    for url in urls[:2]:  # 只试前两个，避免在不可达网络上长时间空等
        try:
            filename = url.split("/")[-1].replace("%20", "_")
            dest = bgm_dir / f"{style}_{filename}"
            if dest.exists() and dest.stat().st_size > 100_000:
                return dest
            # 短超时：连接 3s、总时长 10s，失败立刻换下一个
            with httpx.Client(
                timeout=httpx.Timeout(10.0, connect=3.0),
                follow_redirects=True,
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            if dest.stat().st_size > 100_000:
                logger.info("music.downloaded", path=str(dest))
                return dest
            dest.unlink(missing_ok=True)
        except Exception as e:
            logger.info("music.download_skip", url=url, error=str(e)[:80])
            continue
    return None


def get_background_music(
    style: str = "ambient",
    duration: float = 30.0,
) -> Path | None:
    """获取背景音乐文件路径。

    优先级：
    1. 本地 workspace/bgm/ 目录中**风格匹配**的音乐文件；
    2. 快速在线下载 CC BY 4.0 免版税音乐（网络不佳会很快跳过）；
    3. 返回 None（调用方用 FFmpeg 离线合成，保证有声音）。

    Args:
        style: 音乐风格（upbeat/calm/epic/ambient 或中文描述）
        duration: 需要的时长（秒），仅用于日志，不裁剪

    Returns:
        音乐文件路径，获取失败返回 None
    """
    normalized = _normalize_style(style)
    logger.info("music.request", style=style, normalized=normalized)

    track = _find_local_track(normalized)
    if track:
        logger.info("music.local_found", path=str(track))
        return track

    track = _download_track(normalized)
    if track:
        return track

    logger.info("music.fallback_to_synth", style=normalized)
    return None


# ==================== 真实音乐：本地上传 / 直链 / 在线搜索 ====================

def _safe_filename(text: str, suffix: str = ".m4a") -> Path:
    keep = "".join(c for c in str(text) if c.isalnum() or c in " -_")[:40].strip()
    return _get_bgm_dir() / f"user_{keep or 'track'}{suffix}"


def _download_url(url: str, dest: Path, timeout: float = 15.0) -> Path | None:
    """下载一个可直链的音频地址。"""
    try:
        import httpx
        headers = {"User-Agent": "Mozilla/5.0 (MAVEA)"}
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=4.0),
            follow_redirects=True, headers=headers,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        if dest.stat().st_size > 20_000:
            return dest
        dest.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        logger.info("music.url_download_failed", url=url[:80], error=str(e)[:80])
    return None


def search_online_song(keyword: str) -> tuple[Path | None, str | None]:
    """用 iTunes Search API 在线搜索真实歌曲的 30 秒试听片段（无需登录/Key）。

    适合 15 秒短视频：试听片段本身约 30 秒，循环/裁剪即可。
    返回 (本地文件路径, 曲目名)；搜不到返回 (None, None)。
    """
    try:
        import httpx
        with httpx.Client(
            timeout=httpx.Timeout(12.0, connect=4.0),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (MAVEA)"},
        ) as client:
            resp = client.get(
                "https://itunes.apple.com/search",
                params={"term": keyword, "media": "music",
                        "entity": "song", "limit": 8},
            )
            resp.raise_for_status()
            tracks = resp.json().get("results", [])
        for tr in tracks:
            preview = tr.get("previewUrl")
            if not preview:
                continue
            name = f"{tr.get('trackName', 'song')} - {tr.get('artistName', '')}".strip()
            dest = _safe_filename(name, ".m4a")
            if dest.exists() and dest.stat().st_size > 20_000:
                return dest, name
            got = _download_url(preview, dest, timeout=15.0)
            if got is not None:
                return got, name
    except Exception as e:  # noqa: BLE001
        logger.info("music.online_search_failed", keyword=keyword, error=str(e)[:80])
    return None, None


def resolve_music(
    style: str = "ambient",
    duration: float = 30.0,
    local_file: str | Path | None = None,
    url_or_keyword: str | None = None,
) -> tuple[Path | None, str]:
    """统一选曲，返回 (音频路径或None, 来源说明)。路径为 None 时调用方离线合成。

    优先级：
    1. 用户上传的本地歌曲；
    2. 用户给的音频直链（http 开头）或在线搜索关键词（iTunes 真实歌曲试听）；
    3. workspace/bgm 本地曲库中**风格匹配**的音乐（匹配不到不随机回退）；
    4. 免版税纯音乐在线下载；
    5. None（调用方 FFmpeg 离线合成，保证有声音）。
    """
    # 1. 用户上传的歌曲
    if local_file:
        p = Path(local_file)
        if p.exists() and p.stat().st_size > 1000:
            return p, f"本地歌曲: {p.name}"

    # 2. 直链 / 关键词在线搜索
    if url_or_keyword and url_or_keyword.strip():
        q = url_or_keyword.strip()
        if q.lower().startswith(("http://", "https://")):
            suffix = Path(q.split("?")[0]).suffix or ".mp3"
            dest = _safe_filename("direct", suffix if suffix in (".mp3", ".m4a", ".wav", ".aac", ".ogg") else ".mp3")
            got = _download_url(q, dest)
            if got:
                return got, f"音频链接: {got.name}"
        else:
            got, name = search_online_song(q)
            if got:
                return got, f"在线歌曲: {name}"

    # 3. 本地曲库
    track = _find_local_track(_normalize_style(style))
    if track:
        return track, f"本地曲库: {track.name}"

    # 4. 免版税纯音乐
    track = _download_track(_normalize_style(style))
    if track:
        return track, f"免版税音乐: {track.name}"

    # 5. 交给离线合成
    return None, "离线合成配乐"
