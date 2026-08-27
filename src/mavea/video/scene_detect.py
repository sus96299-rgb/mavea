"""场景检测：基于 PySceneDetect 封装。"""

from __future__ import annotations

from pathlib import Path

import structlog
from scenedetect import ContentDetector, SceneManager, open_video

logger = structlog.get_logger(__name__)


def detect_scenes(
    path: str | Path,
    threshold: float = 27.0,
    min_scene_len: float = 0.6,
) -> list[tuple[float, float]]:
    """检测视频中的镜头边界。

    Args:
        path: 视频文件路径
        threshold: ContentDetector 阈值（HSV 色彩空间平均差异），
                   27.0 为官方默认，越低越敏感
        min_scene_len: 最短镜头时长（秒），小于此时长的相邻镜头合并

    Returns:
        list of (start_sec, end_sec) 元组
    """
    video = open_video(str(Path(path).resolve()))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * video.frame_rate))
    )
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    result: list[tuple[float, float]] = []
    for scene in scenes:
        start_tc = scene[0].get_seconds()
        end_tc = scene[1].get_seconds()
        result.append((start_tc, end_tc))

    logger.info("scene_detect.complete", path=str(path), scenes=len(result))
    return result
