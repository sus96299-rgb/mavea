"""music.py 本地曲库选曲逻辑测试。

回归保护：本地曲库只允许返回"风格匹配"的音乐；
风格匹配不到时必须返回 None 让调用方继续在线下载，
禁止随机回退到任意旧音频——否则用户用过一次的歌
会永久污染之后所有选曲（要"动感"却放出旧的舒缓歌曲）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mavea.audio import music


@pytest.fixture
def fake_bgm_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 BGM 缓存目录重定向到临时目录，不碰真实 workspace。"""
    monkeypatch.setattr(music, "_get_bgm_dir", lambda: tmp_path)
    return tmp_path


def _make_track(d: Path, name: str) -> Path:
    p = d / name
    p.write_bytes(b"fake-audio")
    return p


def test_find_local_track_returns_style_match(fake_bgm_dir: Path):
    _make_track(fake_bgm_dir, "upbeat_Carefree.mp3")
    track = music._find_local_track("upbeat")
    assert track is not None
    assert "upbeat" in track.stem.lower()


def test_find_local_track_no_match_returns_none(fake_bgm_dir: Path):
    # 只有 calm 音乐和一首用户旧歌时，请求 upbeat 绝不能随机返回它们
    _make_track(fake_bgm_dir, "calm_Heartwarming.mp3")
    _make_track(fake_bgm_dir, "user_old_song.m4a")
    assert music._find_local_track("upbeat") is None


def test_find_local_track_empty_dir(fake_bgm_dir: Path):
    assert music._find_local_track("upbeat") is None


def test_normalize_style_chinese_alias():
    # "动感/欢快/轻快" 都应归一化到 upbeat
    assert music._normalize_style("动感快节奏") == "upbeat"
    assert music._normalize_style("舒缓") == "calm"
    assert music._normalize_style("没听过的风格") == "ambient"
