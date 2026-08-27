"""MAVEA 共享数据模型。

对应 docs/interfaces.md 中定义的全部 Pydantic 模型和枚举。
所有跨模块共享的数据结构必须从本文件导入，不得重复定义。
"""

from __future__ import annotations

from enum import Enum
from operator import add
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

# ==================== 枚举 ====================

class MaterialType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


class VideoType(str, Enum):
    ECOMMERCE_PROMO = "ecommerce_promo"
    VLOG_HIGHLIGHT = "vlog_highlight"
    TUTORIAL = "tutorial"
    BRAND_STORY = "brand_story"


class TransitionType(str, Enum):
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    ZOOM = "zoom"
    SLIDE = "slide"


class SubtitlePosition(str, Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class TextAlign(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AgentName(str, Enum):
    ANALYZER = "analyzer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    RETRY = "retry"


class EvaluationDimension(str, Enum):
    TECHNICAL_QUALITY = "technical_quality"
    CONTENT_MATCH = "content_match"
    DURATION_ACCURACY = "duration_accuracy"
    FLUENCY = "fluency"
    SUBTITLE_QUALITY = "subtitle_quality"


class AspectRatio(str, Enum):
    VERTICAL = "1080x1920"
    HORIZONTAL = "1920x1080"
    SQUARE = "1080x1080"


# ==================== 素材分析 ====================

class SceneInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    keyframe_path: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1)


class MaterialInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^mat_[0-9a-f]{6}$")
    type: MaterialType
    path: str
    original_filename: str
    duration: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    codec: str | None = None
    scenes: list[SceneInfo] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)

    @property
    def is_video(self) -> bool:
        return self.type == MaterialType.VIDEO

    @property
    def resolution(self) -> str | None:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None


class MaterialAnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    materials: list[MaterialInfo] = Field(min_length=1)
    summary: str
    warnings: list[str] = Field(default_factory=list)


# ==================== 剪辑方案 ====================

class TextStyle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    font: str = "Sans"
    font_size: int = Field(default=36, ge=8, le=200)
    color: str = "white"
    stroke_color: str | None = "black"
    stroke_width: int = Field(default=2, ge=0, le=10)
    position: SubtitlePosition = SubtitlePosition.BOTTOM
    alignment: TextAlign = TextAlign.CENTER
    margin_v: int = Field(default=40, ge=0)


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    material_id: str = Field(pattern=r"^mat_[0-9a-f]{6}$")
    source_start: float = Field(default=0.0, ge=0)
    source_end: float = Field(gt=0)
    transition_in: TransitionType = TransitionType.CUT
    transition_out: TransitionType = TransitionType.CUT
    transition_duration: float = Field(default=0.5, ge=0, le=3.0)
    subtitle: str | None = None
    subtitle_style: TextStyle | None = None
    text_overlay: str | None = None
    text_overlay_style: TextStyle | None = None
    speed: float = Field(default=1.0, gt=0, le=4.0)

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def output_duration(self) -> float:
        return self.source_duration / self.speed


class TimelinePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_type: VideoType
    target_duration: float = Field(gt=0, le=600)
    aspect_ratio: AspectRatio = AspectRatio.VERTICAL
    style: str
    segments: list[TimelineSegment] = Field(min_length=1)
    bgm_style: str | None = None
    bgm_volume: float = Field(default=0.15, ge=0, le=1)
    voiceover_text: str | None = None
    rationale: str
    rag_references: list[str] = Field(default_factory=list)

    def estimated_duration(self) -> float:
        total = sum(seg.output_duration for seg in self.segments)
        for seg in self.segments[1:]:
            if seg.transition_in != TransitionType.CUT:
                total -= seg.transition_duration
        return total


# ==================== 执行结果 ====================

class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    call_id: str = Field(pattern=r"^call_[0-9a-f]{8}$")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus
    result: dict[str, Any] | None = None
    error_message: str | None = None
    duration_ms: int = Field(ge=0)
    thought: str | None = None


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    success: bool
    output_path: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    intermediate_files: list[str] = Field(default_factory=list)
    error_message: str | None = None
    total_duration_ms: int = Field(default=0, ge=0)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


# ==================== 质量评估 ====================

class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: EvaluationDimension
    score: float = Field(ge=1.0, le=5.0)
    reason: str
    metric_value: float | None = None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scores: list[DimensionScore] = Field(min_length=1)
    overall: float = Field(ge=1.0, le=5.0)
    passed: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    iteration: int = Field(default=1, ge=1)


# ==================== Agent 消息 ====================

class AgentMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    from_agent: AgentName
    to_agent: AgentName
    content: str = Field(min_length=1)
    round: int = Field(default=1, ge=1)


# ==================== LangGraph 状态 ====================

class GraphState(BaseModel):
    """LangGraph 状态图共享状态。"""
    model_config = ConfigDict(extra="forbid")

    # 输入
    user_prompt: str
    material_paths: list[str] = Field(default_factory=list)
    # 用户自定义背景音乐：本地文件路径 / 在线链接或关键词
    custom_bgm_path: str | None = None
    custom_bgm_query: str | None = None

    # 进阶效果（v6）
    beat_sync: bool = False          # 自动卡点：片段边界对齐节拍
    lyric_mode: str = "off"          # off / lrc / whisper
    lrc_path: str | None = None      # lyric_mode=lrc 时的 .lrc 文件
    beauty: bool = False             # 轻度美颜滤镜
    ai_enhance: bool = True          # AI 增强导演：自动决定卡点/美颜/配音

    # 各 Agent 产出
    analysis_report: MaterialAnalysisReport | None = None
    edit_plan: TimelinePlan | None = None
    execution_result: ExecutionResult | None = None
    evaluation_result: EvaluationResult | None = None

    # 流程控制
    iteration: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=3, ge=1, le=5)
    # 历轮评估总分（最近一次在末尾），用于“分数收敛则提前停止返工”
    score_history: list[float] = Field(default_factory=list)

    # Agent 间消息
    messages: Annotated[list[AgentMessage], add] = Field(default_factory=list)

    # 可观测性
    agent_status: dict[str, AgentStatus] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
