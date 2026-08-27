"""DisplayClock 暂停冻结/恢复接续逻辑测试。"""

from mavea.web.gradio_app import DisplayClock


def test_running_clock_tracks_wall_time():
    c = DisplayClock(start=100.0)
    assert c.read(110.0) == 10.0
    assert c.clock_now(110.0) == 110.0


def test_paused_clock_freezes():
    c = DisplayClock(start=100.0)
    c.pause(110.0)
    # 暂停期间墙钟继续走，显示时间冻结在 10s
    assert c.read(115.0) == 10.0
    assert c.read(120.0) == 10.0


def test_resume_continues_from_frozen_value():
    c = DisplayClock(start=100.0)
    c.pause(110.0)   # 10s 处暂停
    c.resume(115.0)  # 暂停了 5s
    assert c.read(120.0) == 15.0  # 20s 墙钟 - 5s 暂停 = 15s


def test_multiple_pauses_accumulate():
    c = DisplayClock(start=0.0)
    c.pause(10.0); c.resume(11.0)   # 扣 1s
    c.pause(20.0); c.resume(23.0)   # 扣 3s
    assert c.read(30.0) == 26.0


def test_clock_now_uses_same_basis():
    c = DisplayClock(start=100.0)
    c.pause(110.0)
    assert c.clock_now(130.0) == 110.0  # 冻结在虚拟时刻 110
