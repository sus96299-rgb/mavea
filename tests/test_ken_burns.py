"""Ken Burns 运镜轮换测试（P1：不再所有图片统一缓慢放大）。"""
from __future__ import annotations

from mavea.video.ffmpeg import _build_ken_burns_vf


def test_auto_motion_cycles_four_directions():
    vfs = [
        _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="auto", motion_index=i)
        for i in range(4)
    ]
    # 连续 4 段运镜必须两两不同（推/拉/左移/右移）
    assert len(set(vfs)) == 4


def test_auto_motion_is_periodic():
    a = _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="auto", motion_index=0)
    b = _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="auto", motion_index=4)
    assert a == b


def test_explicit_motion_ignores_index():
    a = _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="zoom_in", motion_index=0)
    b = _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="zoom_in", motion_index=5)
    assert a == b


def test_pan_uses_frame_interpolation():
    # 2 秒 30fps = 60 帧，平移表达式应带帧进度
    vf = _build_ken_burns_vf(1080, 1920, 30, 2.0, motion="pan_right")
    assert "on/60" in vf
