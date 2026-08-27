"""电商带货节奏模板测试（P1）。

验证：识别带货场景后追加 CTA 段、总时长仍锁定用户目标、
主图比细节图长形成节奏对比、卖点字幕升级为大字卡。
"""
from __future__ import annotations

import pytest

from mavea.agents.planner import _apply_commerce_template
from mavea.models import (
    MaterialAnalysisReport,
    MaterialInfo,
    MaterialType,
    TimelinePlan,
    TimelineSegment,
    VideoType,
)


def _image_report(n: int = 5) -> MaterialAnalysisReport:
    mats = [
        MaterialInfo(
            id=f"mat_{i + 1:06d}",
            type=MaterialType.IMAGE,
            path=f"{i}.jpg",
            original_filename=f"{i}.jpg",
        )
        for i in range(n)
    ]
    return MaterialAnalysisReport(materials=mats, summary=f"{n}张耳机商品图")


def _commerce_plan(n: int = 5, target: float = 15.0) -> TimelinePlan:
    segs = [
        TimelineSegment(
            index=i, material_id=f"mat_{i + 1:06d}",
            source_start=0.0, source_end=3.0, subtitle=f"核心卖点{i}",
        )
        for i in range(n)
    ]
    return TimelinePlan(
        video_type=VideoType.ECOMMERCE_PROMO,
        target_duration=target,
        style="fast",
        segments=segs,
        rationale="test",
    )


def test_commerce_appends_cta_and_locks_duration():
    report = _image_report(5)
    plan = _commerce_plan(5, target=15.0)

    out = _apply_commerce_template(plan, VideoType.ECOMMERCE_PROMO, report, "结尾引导点击购买")

    # 5 张图 + 1 个结尾 CTA 段
    assert len(out.segments) == 6
    cta = out.segments[-1]
    assert cta.text_overlay == "点击购买"
    assert cta.material_id == "mat_000005"  # 复用最后一张商品图

    # 总时长仍锁定 15 秒
    total = sum(s.source_end - s.source_start for s in out.segments)
    assert abs(total - 15.0) < 0.1

    # 主图和 CTA 应比中间细节图长（长短节奏对比）
    mid = out.segments[2].source_end
    assert out.segments[0].source_end > mid
    assert cta.source_end > mid


def test_subtitle_promoted_to_big_card():
    out = _apply_commerce_template(
        _commerce_plan(3, 9.0), VideoType.ECOMMERCE_PROMO,
        _image_report(3), "带货",
    )
    first = out.segments[0]
    assert first.subtitle is None
    assert first.text_overlay == "核心卖点0"
    assert first.text_overlay_style is not None
    assert first.text_overlay_style.font_size >= 56


def test_cta_word_follows_prompt():
    out = _apply_commerce_template(
        _commerce_plan(3, 9.0), VideoType.ECOMMERCE_PROMO,
        _image_report(3), "引导立即下单",
    )
    assert out.segments[-1].text_overlay == "立即下单"


def test_non_commerce_unchanged():
    plan = _commerce_plan(3, 9.0)
    plan.video_type = VideoType.VLOG_HIGHLIGHT
    before = len(plan.segments)
    out = _apply_commerce_template(
        plan, VideoType.VLOG_HIGHLIGHT, _image_report(3), "日常vlog",
    )
    assert len(out.segments) == before
    assert out.segments[0].subtitle == "核心卖点0"
