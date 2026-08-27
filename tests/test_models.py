"""基础测试：验证模型定义和配置加载。"""

import pytest

from mavea.models import (
    AspectRatio,
    GraphState,
    MaterialInfo,
    MaterialType,
    TimelinePlan,
    TimelineSegment,
    TransitionType,
    VideoType,
)


def test_timeline_segment_duration():
    """验证片段时长计算。"""
    seg = TimelineSegment(
        index=0,
        material_id="mat_abc123",
        source_start=0.0,
        source_end=5.0,
    )
    assert seg.source_duration == 5.0
    assert seg.output_duration == 5.0

    seg_speed = TimelineSegment(
        index=1,
        material_id="mat_abc123",
        source_start=0.0,
        source_end=4.0,
        speed=2.0,
    )
    assert seg_speed.output_duration == 2.0


def test_timeline_plan_estimated_duration():
    """验证方案预估时长（含转场扣减）。"""
    segments = [
        TimelineSegment(index=0, material_id="mat_aaa111", source_start=0, source_end=3),
        TimelineSegment(
            index=1, material_id="mat_bbb222", source_start=0, source_end=4,
            transition_in=TransitionType.FADE, transition_duration=0.5,
        ),
    ]
    plan = TimelinePlan(
        video_type=VideoType.ECOMMERCE_PROMO,
        target_duration=6.5,
        style="fast",
        segments=segments,
        rationale="测试",
    )
    # 3 + 4 - 0.5 (fade转场) = 6.5
    assert plan.estimated_duration() == pytest.approx(6.5)


def test_graph_state_defaults():
    """验证 GraphState 默认值。"""
    state = GraphState(
        user_prompt="测试",
        material_paths=["test.mp4"],
    )
    assert state.iteration == 1
    assert state.max_iterations == 3
    assert state.messages == []
    assert state.errors == []


def test_material_info_video():
    """验证素材信息。"""
    mat = MaterialInfo(
        id="mat_abc123",
        type=MaterialType.VIDEO,
        path="test.mp4",
        original_filename="test.mp4",
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
    )
    assert mat.is_video is True
    assert mat.resolution == "1920x1080"


def test_aspect_ratio_enum():
    """验证宽高比枚举。"""
    assert AspectRatio.VERTICAL.value == "1080x1920"
    assert AspectRatio.HORIZONTAL.value == "1920x1080"
