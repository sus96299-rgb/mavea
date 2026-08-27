"""MCP 分析工具：场景检测。"""

from __future__ import annotations

from typing import Any

import structlog

from mavea.mcp.tools._helpers import make_result, resolve_input_path
from mavea.video.scene_detect import detect_scenes

logger = structlog.get_logger(__name__)


def detect_scenes_tool(
    path: str,
    threshold: float = 27.0,
    min_scene_len: float = 0.6,
) -> dict[str, Any]:
    """检测视频中的镜头/场景边界。

    Args:
        path: 视频文件路径
        threshold: 内容变化阈值（27为默认值，越低越敏感，vlog建议20-25）
        min_scene_len: 最短镜头时长（秒）
    """
    try:
        p = resolve_input_path(path)
        scenes = detect_scenes(p, threshold, min_scene_len)
        return make_result(True, {
            "scene_count": len(scenes),
            "scenes": [
                {"start": round(s, 3), "end": round(e, 3)}
                for s, e in scenes
            ],
        })
    except Exception as e:
        logger.error("mcp.detect_scenes.failed", error=str(e))
        return make_result(False, error=str(e))
