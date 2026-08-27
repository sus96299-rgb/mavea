"""进度事件总线：让 WebUI 能实时、分类地显示 Agent 执行情况。

事件分两类（回调收到的是一个 dict）：
- 阶段事件：{"kind": "stage", "stage": ..., "message": ..., "percent": 0.0-1.0}
- 工具事件：{"kind": "tool",  "tool": ..., "status": "start/success/error",
             "duration_ms": int|None, "detail": str|None}

用法：
    from mavea.progress import set_progress_callback, report_progress, report_tool

    def on_event(event: dict):
        queue.put_nowait(event)

    set_progress_callback(on_event)
    report_progress("analyzer", "检测场景中...", 0.15)
    report_tool("cut_clip", "start")
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Any

# 暂停控制：set=运行中，clear=已暂停。report_progress 在每个进度检查点等待，
# 从而实现“协作式暂停”（不会强杀 ffmpeg，只在阶段/片段边界停住）。
_pause_event = threading.Event()
_pause_event.set()


def request_pause() -> None:
    """请求暂停：流水线在下一个进度检查点停住。"""
    _pause_event.clear()
    _emit({"kind": "pause_requested"})


def request_resume() -> None:
    """恢复运行。"""
    _pause_event.set()


def is_paused() -> bool:
    return not _pause_event.is_set()


# 取消控制：set=用户点了“全部停止”。用 BaseException 子类，以便穿过各 Agent
# 内部的 except Exception，真正中断整条流水线。
class PipelineCancelled(BaseException):
    """用户主动停止生成。"""


_cancel_event = threading.Event()


def request_cancel() -> None:
    """请求停止：流水线在下一个检查点抛出 PipelineCancelled 中止。"""
    _cancel_event.set()
    _pause_event.set()  # 同时解除暂停阻塞，让取消立即生效
    _emit({"kind": "cancel_requested"})


def is_cancelled() -> bool:
    return _cancel_event.is_set()


def reset_controls() -> None:
    """每次开跑前重置暂停/停止状态。"""
    _cancel_event.clear()
    _pause_event.set()

# 阶段名称映射到进度百分比范围
STAGE_RANGES = {
    "analyzer": (0.05, 0.35),
    "planner": (0.35, 0.50),
    "executor": (0.50, 0.85),
    "evaluator": (0.85, 1.0),
}

# 当前回调函数（接收单个 event dict）
_callback: Callable[[dict[str, Any]], None] | None = None


def set_progress_callback(callback: Callable[[dict[str, Any]], None] | None) -> None:
    """设置事件回调函数。传 None 清除。"""
    global _callback
    _callback = callback


def _emit(event: dict[str, Any]) -> None:
    """安全地发出一个事件，回调失败绝不影响主流程。"""
    if _callback is None:
        return
    with contextlib.suppress(Exception):
        _callback(event)


def report_progress(stage: str, message: str, fraction: float = 0.5) -> None:
    """报告阶段进度。

    Args:
        stage: 阶段名 analyzer/planner/executor/evaluator
        message: 进度消息
        fraction: 该阶段内的完成比例 0-1
    """
    # 协作式暂停：若用户点了暂停，这里阻塞直到恢复（不影响 ffmpeg 已在跑的原子步骤）
    if not _pause_event.is_set():
        _emit({"kind": "paused"})  # 通知界面：已到达暂停点，可以冻结时钟
        _pause_event.wait()
        _emit({"kind": "resumed"})
    # 用户点了“全部停止”：抛出取消异常中断流水线
    if _cancel_event.is_set():
        raise PipelineCancelled("用户停止了生成")
    lo, hi = STAGE_RANGES.get(stage, (0.0, 1.0))
    fraction = max(0.0, min(1.0, fraction))
    percent = lo + (hi - lo) * fraction
    _emit({
        "kind": "stage",
        "stage": stage,
        "message": str(message),
        "percent": percent,
    })


def report_tool(
    tool: str,
    status: str,
    duration_ms: int | None = None,
    detail: str | None = None,
) -> None:
    """报告一次工具调用的开始/结束。

    Args:
        tool: 工具名
        status: "start" / "success" / "error"
        duration_ms: 耗时（结束时）
        detail: 附加信息（如错误信息）
    """
    _emit({
        "kind": "tool",
        "tool": str(tool),
        "status": str(status),
        "duration_ms": duration_ms,
        "detail": str(detail) if detail is not None else None,
    })
