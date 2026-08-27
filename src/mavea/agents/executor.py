"""工具执行 Agent（Plan-and-Execute + 失败降级）。

职责：
1. 接收 TimelinePlan
2. 优先使用 create_video_from_timeline 一站式渲染
3. 失败时降级为确定性的逐步工具调用（逐片段裁剪/图生视频 → 拼接 → 混音）
4. 输出 ExecutionResult

注意：本节点没有 LLM 推理循环，工具序列由方案确定性推导，
不是 ReAct；白名单内的自主工具选择见 agents/director.py。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from mavea.mcp.client import get_mcp_client
from mavea.models import (
    AgentName,
    AgentStatus,
    ExecutionResult,
    MaterialAnalysisReport,
    TimelinePlan,
    ToolCallRecord,
    ToolCallStatus,
)
from mavea.progress import report_progress

logger = structlog.get_logger(__name__)

_STAGE = "executor"


def _log(msg: str, fraction: float = 0.5):
    print(f"[Executor] {msg}", flush=True)
    logger.info("executor.progress", msg=msg)
    report_progress(_STAGE, msg, fraction)


def _gen_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _make_record(
    tool_name: str,
    args: dict,
    result: dict,
    duration_ms: int,
    thought: str = "",
) -> ToolCallRecord:
    """从工具返回构造 ToolCallRecord。"""
    success = result.get("success", False)
    return ToolCallRecord(
        call_id=_gen_call_id(),
        tool_name=tool_name,
        arguments=args,
        status=ToolCallStatus.SUCCESS if success else ToolCallStatus.ERROR,
        result=result.get("data") if success else None,
        error_message=None if success else result.get("error", "未知错误"),
        duration_ms=duration_ms,
        thought=thought,
    )


async def execute_plan(
    plan: TimelinePlan,
    analysis_report: MaterialAnalysisReport,
    bgm_file: str | None = None,
    bgm_query: str | None = None,
    beat_sync: bool = False,
    lyric_mode: str = "off",
    lrc_path: str | None = None,
    beauty: bool = False,
) -> ExecutionResult:
    """执行剪辑方案。

    策略：
    1. 优先使用 create_video_from_timeline 一站式渲染（最可靠）
    2. 失败时尝试逐步执行（cut → concat → subtitle）
    """
    if plan is None or not getattr(plan, "segments", None):
        # 规划失败时直接返回，避免对 None 调 model_dump_json 造成连环报错
        return ExecutionResult(
            success=False,
            output_path=None,
            tool_calls=[],
            intermediate_files=[],
            error_message="剪辑方案为空（规划阶段失败），请查看上方“规划失败”原因",
            total_duration_ms=0,
        )
    _log("连接 MCP 工具服务...", 0.05)
    client = await get_mcp_client()

    material_map = {m.id: m.path for m in analysis_report.materials}
    tool_calls: list[ToolCallRecord] = []
    intermediate_files: list[str] = []

    # ---- 策略1：一站式渲染 ----
    _log("一站式渲染时间轴...", 0.15)
    plan_json = plan.model_dump_json()

    start_time = time.time()
    result = await client.call_tool("create_video_from_timeline", {
        "plan_json": plan_json,
        "material_map": material_map,
        "bgm_file": bgm_file,
        "bgm_query": bgm_query,
        "beat_sync": bool(beat_sync),
        "lyric_mode": lyric_mode,
        "lrc_path": lrc_path,
        "beauty": bool(beauty),
    })
    duration_ms = int((time.time() - start_time) * 1000)
    tool_calls.append(_make_record(
        "create_video_from_timeline",
        {"plan_json": "...", "material_map": "..."},
        result,
        duration_ms,
        "一站式时间轴渲染",
    ))

    if result.get("success"):
        output_path = result["data"]["output_path"]
        _log(f"渲染完成: {output_path}", 0.9)
        return ExecutionResult(
            success=True,
            output_path=output_path,
            tool_calls=tool_calls,
            intermediate_files=intermediate_files,
            total_duration_ms=duration_ms,
        )

    _log(f"一站式渲染失败: {result.get('error', '未知')}，尝试逐步执行...", 0.3)

    # ---- 策略2：逐步执行 ----
    final_output = await _step_by_step_execute(
        client, plan, material_map, tool_calls, intermediate_files,
        bgm_file=bgm_file, bgm_query=bgm_query,
    )

    success = final_output is not None
    return ExecutionResult(
        success=success,
        output_path=final_output,
        tool_calls=tool_calls,
        intermediate_files=intermediate_files,
        error_message=None if success else "执行失败，查看 tool_calls 了解详情",
        total_duration_ms=sum(c.duration_ms for c in tool_calls),
    )


async def _step_by_step_execute(
    client,
    plan: TimelinePlan,
    material_map: dict[str, str],
    tool_calls: list[ToolCallRecord],
    intermediate_files: list[str],
    bgm_file: str | None = None,
    bgm_query: str | None = None,
) -> str | None:
    """逐步执行：每个片段裁剪 → 拼接。"""
    clip_paths: list[str] = []

    for i, seg in enumerate(plan.segments):
        fraction = 0.3 + 0.5 * (i / max(len(plan.segments), 1))
        mat_path = material_map.get(seg.material_id)
        if not mat_path:
            _log(f"片段 {seg.index}: 素材 {seg.material_id} 不存在", fraction)
            continue

        ext = Path(mat_path).suffix.lower()
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        duration = seg.source_end - seg.source_start

        try:
            if ext in image_exts:
                _log(f"片段 {seg.index}: 图片转视频 ({duration:.1f}s)", fraction)
                result = await client.call_tool("image_to_video", {
                    "image_paths": [mat_path],
                    "duration_per_image": duration,
                    "ken_burns": True,
                    "resolution": plan.aspect_ratio.value,
                })
            else:
                _log(f"片段 {seg.index}: 裁剪 {seg.source_start:.1f}-{seg.source_end:.1f}s", fraction)
                result = await client.call_tool("cut_clip", {
                    "path": mat_path,
                    "start": seg.source_start,
                    "end": seg.source_end,
                })

            duration_ms = 0
            tool_calls.append(_make_record(
                "image_to_video" if ext in image_exts else "cut_clip",
                {"material_id": seg.material_id},
                result,
                duration_ms,
            ))

            if result.get("success"):
                clip_path = result["data"]["output_path"]
                clip_paths.append(clip_path)
                intermediate_files.append(clip_path)
            else:
                _log(f"片段 {seg.index} 失败: {result.get('error')}", fraction)
                return None

        except Exception as e:
            _log(f"片段 {seg.index} 异常: {e}", fraction)
            return None

    if not clip_paths:
        return None

    # 拼接
    _log(f"拼接 {len(clip_paths)} 个片段...", 0.85)
    # 默认硬切，避免 xfade 转场重叠导致时长缩短
    transition = "cut"
    result = await client.call_tool("concat_videos", {
        "paths": clip_paths,
        "transition": transition,
        "target_resolution": plan.aspect_ratio.value,
    })
    tool_calls.append(_make_record("concat_videos", {"count": len(clip_paths)}, result, 0))

    if result.get("success"):
        final_path = result["data"]["output_path"]
        # 如果方案要求 BGM，优先下载真实音乐，失败则合成
        if plan.bgm_style:
            try:
                from mavea.audio.music import resolve_music
                from mavea.video import ffmpeg as ff
                probe = ff.probe_video(final_path)
                bgm_path, music_label = resolve_music(
                    style=plan.bgm_style or "ambient",
                    duration=probe.duration + 1,
                    local_file=bgm_file,
                    url_or_keyword=bgm_query,
                )
                _log(f"背景音乐来源: {music_label}", 0.92)
                if bgm_path is None:
                    bgm_path = ff.generate_bgm(
                        duration=probe.duration + 1,
                        style=plan.bgm_style if plan.bgm_style in ("upbeat", "epic", "calm", "ambient") else "ambient",
                    )
                    mix_vol = 0.7  # 离线合成乐需要更高音量
                    _log(f"使用离线合成配乐 ({plan.bgm_style})", 0.93)
                else:
                    mix_vol = 0.3  # 真实音乐音量适中
                    _log(f"使用背景音乐: {Path(bgm_path).name}", 0.93)
                final_path = str(ff.mix_audio(final_path, bgm_path, bgm_volume=mix_vol))
                _log("背景音乐已混入", 0.95)
            except Exception as be:
                _log(f"BGM 添加失败（不影响成片）: {be}", 0.95)
        return final_path
    return None


async def run(state) -> dict[str, Any]:
    """LangGraph 节点入口。"""
    logger.info("agent.executor.start")
    state.agent_status[AgentName.EXECUTOR.value] = AgentStatus.RUNNING

    try:
        result = await execute_plan(
            state.edit_plan, state.analysis_report,
            getattr(state, 'custom_bgm_path', None),
            getattr(state, 'custom_bgm_query', None),
            bool(getattr(state, 'beat_sync', False)),
            str(getattr(state, 'lyric_mode', 'off') or 'off'),
            getattr(state, 'lrc_path', None),
            bool(getattr(state, 'beauty', False)),
        )
        if result.success:
            state.agent_status[AgentName.EXECUTOR.value] = AgentStatus.COMPLETED
            _log("执行完成", 1.0)
        else:
            state.agent_status[AgentName.EXECUTOR.value] = AgentStatus.FAILED
            state.errors.append(f"执行失败: {result.error_message}")
        return {"execution_result": result, "agent_status": state.agent_status, "errors": state.errors}
    except Exception as e:
        state.agent_status[AgentName.EXECUTOR.value] = AgentStatus.FAILED
        state.errors.append(f"执行异常: {e}")
        logger.error("agent.executor.failed", error=str(e))
        return {"agent_status": state.agent_status, "errors": state.errors}
