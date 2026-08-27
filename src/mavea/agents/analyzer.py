"""素材分析 Agent。

职责：
1. 探测视频/图片元信息
2. 场景检测 + 关键帧提取
3. 视觉模型生成画面描述
4. Whisper 音频转写
5. 输出 MaterialAnalysisReport
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import structlog

from mavea.config import get_settings
from mavea.llm.base import get_vision_llm
from mavea.models import (
    MaterialAnalysisReport,
    MaterialInfo,
    MaterialType,
    SceneInfo,
    TranscriptSegment,
)
from mavea.progress import report_progress
from mavea.video.ffmpeg import probe_video
from mavea.video.frames import extract_scene_keyframes
from mavea.video.scene_detect import detect_scenes

logger = structlog.get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}


_STAGE = "analyzer"

def _log(msg: str, fraction: float = 0.5):
    """同时输出到终端和进度回调。"""
    print(f"[Analyzer] {msg}", flush=True)
    logger.info("analyzer.progress", msg=msg)
    report_progress(_STAGE, msg, fraction)


def _gen_material_id() -> str:
    return f"mat_{uuid.uuid4().hex[:6]}"


def _setup_hf_mirror(settings):
    """设置 HuggingFace 国内镜像（如果配置了）。"""
    mirror = settings.video.hf_mirror
    if mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror
        _log(f"设置 HuggingFace 镜像: {mirror}")


def _analyze_video(path: Path, settings) -> MaterialInfo:
    """分析单个视频素材。"""
    mat_id = _gen_material_id()

    _log(f"探测视频信息: {path.name}", 0.1)
    info = probe_video(path)
    _log(f"视频信息: {info.width}x{info.height}, {info.duration:.1f}s, {info.codec}, 音频={'有' if info.has_audio else '无'}")

    _log("检测场景边界...", 0.25)
    scene_boundaries = detect_scenes(
        path,
        threshold=settings.video.scene_threshold,
        min_scene_len=0.6,
    )
    _log(f"检测到 {len(scene_boundaries)} 个镜头")

    _log("提取关键帧...", 0.4)
    keyframes = extract_scene_keyframes(
        path,
        threshold=settings.video.scene_threshold,
    )
    _log(f"提取了 {len(keyframes)} 个关键帧")

    # 视觉描述
    _log("调用视觉模型描述画面（Qwen-VL）...", 0.5)
    vision_llm = get_vision_llm()
    scenes: list[SceneInfo] = []
    for i, (start, end, kf_path) in enumerate(keyframes):
        _log(f"  镜头 {i+1}/{len(keyframes)}: {start:.1f}s-{end:.1f}s")
        try:
            description = vision_llm.generate_vision(
                "请用一句话描述这个视频画面的内容，包括主体、场景、动作、色调。"
                "如果是商品，请描述商品特征。",
                [str(kf_path)],
            )
            _log(f"  描述: {description[:60]}")
        except Exception as e:
            logger.warning("analyzer.vision_failed", scene=i, error=str(e))
            _log(f"  视觉描述失败: {e}")
            description = f"镜头 {i+1}"
        tags = _extract_tags(description)
        scenes.append(SceneInfo(
            start=round(start, 3),
            end=round(end, 3),
            description=description,
            tags=tags,
            keyframe_path=str(kf_path),
        ))

    # 如果场景检测失败，至少创建一个覆盖全片的 scene
    if not scenes:
        scenes.append(SceneInfo(
            start=0,
            end=info.duration,
            description="视频内容",
            tags=[],
        ))

    # 音频转写
    transcript: list[TranscriptSegment] = []
    if info.has_audio and not settings.video.skip_transcribe:
        _log(f"加载 Whisper 模型 ({settings.video.whisper_model_size}) 并转写音频...")
        try:
            from mavea.mcp.tools.media_tools import transcribe_audio
            result = transcribe_audio(str(path), settings.video.whisper_model_size, "zh")
            if result.get("success"):
                for seg in result["data"]["segments"]:
                    transcript.append(TranscriptSegment(
                        start=seg["start"],
                        end=seg["end"],
                        text=seg["text"],
                    ))
                _log(f"转写完成: {len(transcript)} 段")
            else:
                _log(f"转写失败: {result.get('error', '未知')}")
        except Exception as e:
            logger.warning("analyzer.transcribe_failed", error=str(e))
            _log(f"转写异常（跳过）: {e}")
    elif info.has_audio and settings.video.skip_transcribe:
        _log("跳过音频转写（skip_transcribe=True）")

    return MaterialInfo(
        id=mat_id,
        type=MaterialType.VIDEO,
        path=str(path),
        original_filename=path.name,
        duration=round(info.duration, 3),
        width=info.width,
        height=info.height,
        fps=round(info.fps, 2),
        codec=info.codec,
        scenes=scenes,
        transcript=transcript,
    )


def _analyze_image(path: Path) -> MaterialInfo:
    """分析单个图片素材。"""
    mat_id = _gen_material_id()

    _log(f"分析图片: {path.name}")
    description = "图片素材"
    tags: list[str] = []
    try:
        vision_llm = get_vision_llm()
        description = vision_llm.generate_vision(
            "请用一句话描述这张图片的内容。如果是商品图片，请描述商品类型、颜色、特征。",
            [str(path)],
        )
        _log(f"图片描述: {description[:60]}")
        tags = _extract_tags(description)
    except Exception as e:
        logger.warning("analyzer.image_vision_failed", error=str(e))
        _log(f"视觉描述失败: {e}")

    # 获取图片尺寸
    width, height = 0, 0
    try:
        import cv2
        import numpy as np
        # cv2.imread 在 Windows 上不支持中文/非 ASCII 路径，用 imdecode 兜底
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
    except Exception:
        pass

    return MaterialInfo(
        id=mat_id,
        type=MaterialType.IMAGE,
        path=str(path),
        original_filename=path.name,
        width=width or None,
        height=height or None,
        scenes=[SceneInfo(start=0, end=3.0, description=description, tags=tags)],
    )


def _extract_tags(description: str) -> list[str]:
    """从描述中简单提取标签（后续可替换为NER模型）。"""
    keywords = [
        "产品特写", "白底", "使用场景", "户外", "室内", "人物", "美食",
        "风景", "运动", "办公", "居家", "街拍", "夜景", "航拍",
        "运动鞋", "服装", "美妆", "数码", "食品", "家居",
    ]
    tags = []
    for kw in keywords:
        if kw in description:
            tags.append(kw)
    return tags


def analyze_materials(material_paths: list[str]) -> MaterialAnalysisReport:
    """素材分析 Agent 主函数。

    Args:
        material_paths: 素材文件路径列表

    Returns:
        MaterialAnalysisReport
    """
    settings = get_settings()
    _setup_hf_mirror(settings)

    # 并发分析：图片的视觉大模型调用是网络 IO，视频是子进程，多线程可大幅提速。
    # 11 张图串行≈11 倍耗时，并发后≈1-2 倍。按原始顺序收集结果。
    import concurrent.futures

    def _analyze_one(item):
        idx, path_str = item
        path = Path(path_str).resolve()
        if not path.exists():
            return idx, None, f"文件不存在: {path_str}"
        ext = path.suffix.lower()
        total = max(1, len(material_paths))
        _log(f"[{idx + 1}/{total}] 并发分析: {path.name}", 0.05 + 0.85 * idx / total)
        try:
            if ext in VIDEO_EXTENSIONS:
                return idx, _analyze_video(path, settings), None
            if ext in IMAGE_EXTENSIONS:
                return idx, _analyze_image(path), None
            return idx, None, f"不支持的文件类型: {path.name}"
        except Exception as e:  # noqa: BLE001
            logger.error("analyzer.failed", file=path.name, error=str(e))
            return idx, None, f"分析失败 {path.name}: {e}"

    indexed = list(enumerate(material_paths))
    returned: list = [None] * len(indexed)
    max_workers = min(6, max(1, len(indexed)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_analyze_one, item) for item in indexed]
        for fut in concurrent.futures.as_completed(futures):
            idx, mat, warn = fut.result()
            returned[idx] = (mat, warn)
            if mat is not None:
                _log(f"已完成 {idx + 1}/{len(indexed)}：{Path(material_paths[idx]).name}",
                     0.1 + 0.85 * (idx + 1) / len(indexed))

    materials: list[MaterialInfo] = []
    warnings: list[str] = []
    for item in returned:
        if item is None:
            continue
        mat, warn = item
        if warn:
            warnings.append(warn)
        if mat is not None:
            materials.append(mat)

    if not materials:
        raise RuntimeError(f"没有可分析的素材。warnings: {warnings}")

    # 生成摘要
    summary_parts = []
    for mat in materials:
        if mat.is_video:
            summary_parts.append(
                f"视频素材 {mat.id}（{mat.duration}s，{len(mat.scenes)}个镜头）"
            )
        else:
            summary_parts.append(f"图片素材 {mat.id}（{mat.scenes[0].description[:30]}）")
    summary = "；".join(summary_parts)
    _log(f"素材分析完成: {summary}", 1.0)

    return MaterialAnalysisReport(
        materials=materials,
        summary=summary,
        warnings=warnings,
    )


def run(state) -> dict[str, Any]:
    """LangGraph 节点入口。"""
    from mavea.models import AgentName, AgentStatus

    logger.info("agent.analyzer.start", materials=len(state.material_paths))
    state.agent_status[AgentName.ANALYZER.value] = AgentStatus.RUNNING

    try:
        report = analyze_materials(state.material_paths)
        state.agent_status[AgentName.ANALYZER.value] = AgentStatus.COMPLETED
        logger.info("agent.analyzer.done", materials=len(report.materials))
        return {"analysis_report": report, "agent_status": state.agent_status}
    except Exception as e:
        state.agent_status[AgentName.ANALYZER.value] = AgentStatus.FAILED
        state.errors.append(f"分析失败: {e}")
        logger.error("agent.analyzer.failed", error=str(e))
        return {"agent_status": state.agent_status, "errors": state.errors}
