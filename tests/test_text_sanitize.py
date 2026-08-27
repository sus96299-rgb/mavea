"""drawtext 文案清洗与底条渲染测试。"""

from mavea.video.ffmpeg import _sanitize_overlay_text


def test_strips_emoji_keeps_cjk_and_pipe():
    out = _sanitize_overlay_text("🎧 智能降噪｜专注每一刻 ✨")
    assert out == "智能降噪｜专注每一刻"


def test_keeps_normal_selling_copy():
    assert _sanitize_overlay_text("点击购买") == "点击购买"
    assert _sanitize_overlay_text("40h超长续航！") == "40h超长续航！"


def test_pure_emoji_becomes_empty():
    assert _sanitize_overlay_text("🎵🔥✨") == ""


def test_strips_dingbats_but_keeps_punctuation():
    # ❤/◆/▍ 等装饰符号应剥除，中文标点保留
    out = _sanitize_overlay_text("◆个性音质，沉浸体验")
    assert out == "个性音质，沉浸体验"
