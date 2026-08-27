# MAVEA 接口契约（docs/interfaces.md）

> **本文件是整个项目的"宪法"。** 所有模块的共享数据结构、枚举类型、函数签名以此为准。
> 任何模块不得私自修改此处定义；如需修改，必须先提出并说明原因、影响范围，经确认后统一更新。
>
> 最后更新：2026-08-25 | 适用版本：MAVEA v0.1.0

---

## 0. 通用约定

### 0.1 路径安全

- 所有文件路径在系统内部统一使用 `Path` 对象，跨模块传递时可转为 `str`。
- 所有用户可控路径必须经过工作目录校验：解析为绝对路径后，必须位于 `Settings.workspace_dir` 之内，禁止路径穿越（`..`）。
- MCP 工具接收的路径参数由 `mcp/tools/` 层负责校验。

### 0.2 ID 规范

- 素材 ID：`mat_{6位随机hex}`，如 `mat_a1b2c3`
- 片段 ID：`seg_{index:03d}`，如 `seg_001`
- 工具调用 ID：`call_{uuid4 hex前8位}`
- 所有 ID 在同一次任务内唯一。

### 0.3 时间表示

- 视频时间点统一用 **秒（float）**，精确到毫秒（如 `3.5` 表示 3.5 秒）。
- 不用 `timedelta`、不用 `MM:SS.mmm` 字符串，除非在展示层转换。

### 0.4 序列化

- 所有 Pydantic 模型使用 `model_config = ConfigDict(extra="forbid")`，防止拼写错误的字段被静默吞掉。
- 枚举序列化为值（`use_enum_values=False`，在 JSON 中通过 `model_dump(mode="json")` 自动转为字符串）。

### 0.5 输出路径约定

所有 MCP 工具的 `output_path` 参数遵循统一规则：

- **为 None 时**：自动生成，格式为 `{Settings.output_dir}/{工具名}_{YYYYMMDD_HHMMSS}_{4位随机hex}.mp4`，例如 `workspace/output/cut_clip_20260825_143022_a1b2.mp4`。
- **不为 None 时**：必须是位于工作目录内的绝对路径或相对路径；若父目录不存在则自动创建。
- 工具返回的 `ToolResult.data` 中必须包含 `output_path` 字段（绝对路径），供下游工具使用。
- 中间产物（临时片段）放在 `{Settings.temp_dir}` 下，由执行 Agent 在任务结束后统一清理。

---

## 1. 枚举类型

```python
from enum import Enum


class MaterialType(str, Enum):
    """素材类型"""
    VIDEO = "video"
    IMAGE = "image"


class VideoType(str, Enum):
    """目标视频类型"""
    ECOMMERCE_PROMO = "ecommerce_promo"   # 电商带货短视频（主场景）
    VLOG_HIGHLIGHT = "vlog_highlight"     # Vlog 精彩集锦
    TUTORIAL = "tutorial"                 # 教程/知识类
    BRAND_STORY = "brand_story"           # 品牌宣传


class TransitionType(str, Enum):
    """转场类型。当前仅保证 cut/fade/dissolve 三种稳定可用，
    其余由 FFmpeg xfade 滤镜实现，兼容性需运行时检测。"""
    CUT = "cut"           # 硬切（无转场）
    FADE = "fade"         # 淡入淡出
    DISSOLVE = "dissolve" # 溶解
    WIPE = "wipe"         # 擦除
    ZOOM = "zoom"         # 缩放过渡
    SLIDE = "slide"       # 滑动


class SubtitlePosition(str, Enum):
    """字幕/文字贴纸位置"""
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class TextAlign(str, Enum):
    """文字对齐方式"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AgentName(str, Enum):
    """Agent 名称标识"""
    ANALYZER = "analyzer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"


class AgentStatus(str, Enum):
    """Agent 执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCallStatus(str, Enum):
    """MCP 工具调用状态"""
    SUCCESS = "success"
    ERROR = "error"
    RETRY = "retry"


class EvaluationDimension(str, Enum):
    """质量评估维度"""
    TECHNICAL_QUALITY = "technical_quality"   # 技术画质（NIQE/BRISQUE）
    CONTENT_MATCH = "content_match"           # 内容匹配度（LLM-as-Judge）
    DURATION_ACCURACY = "duration_accuracy"   # 时长准确性
    FLUENCY = "fluency"                       # 流畅度（黑帧/卡顿/音画同步）
    SUBTITLE_QUALITY = "subtitle_quality"     # 字幕质量


class AspectRatio(str, Enum):
    """输出视频比例与分辨率"""
    VERTICAL = "1080x1920"    # 9:16 竖屏（抖音/快手）
    HORIZONTAL = "1920x1080"  # 16:9 横屏（B站/YouTube）
    SQUARE = "1080x1080"      # 1:1 方形（小红书/Instagram）


class LLMProvider(str, Enum):
    """LLM 提供方"""
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OPENAI = "openai"
```

---

## 2. 素材分析模型

### 2.1 SceneInfo —— 单个镜头/场景

```python
from pydantic import BaseModel, Field, ConfigDict


class SceneInfo(BaseModel):
    """视频中的一个连续镜头（场景检测切分结果）。
    图片素材没有场景概念，整体视为一个 scene。"""
    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., description="镜头起始时间（秒）", ge=0)
    end: float = Field(..., description="镜头结束时间（秒）", gt=0)
    description: str = Field(..., description="视觉模型生成的画面描述", min_length=1)
    tags: list[str] = Field(
        default_factory=list,
        description="画面标签，如 ['产品特写','白底','运动鞋']"
    )
    keyframe_path: str | None = Field(
        default=None,
        description="代表帧截图的绝对路径（分析阶段提取）"
    )

    @property
    def duration(self) -> float:
        return self.end - self.start
```

### 2.2 TranscriptSegment —— 音频转写片段

```python
class TranscriptSegment(BaseModel):
    """音频转写的一个时间片段"""
    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str = Field(..., min_length=1)
```

### 2.3 MaterialInfo —— 单个素材

```python
class MaterialInfo(BaseModel):
    """一个上传的素材文件（视频或图片）及其分析结果"""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="素材唯一标识，如 mat_a1b2c3", pattern=r"^mat_[0-9a-f]{6}$")
    type: MaterialType
    path: str = Field(..., description="素材文件绝对路径")
    original_filename: str = Field(..., description="用户上传时的原始文件名")

    # 视频元信息（图片素材为 None）
    duration: float | None = Field(default=None, description="视频时长（秒）", ge=0)
    width: int | None = Field(default=None, description="宽度（像素）", gt=0)
    height: int | None = Field(default=None, description="高度（像素）", gt=0)
    fps: float | None = Field(default=None, description="帧率", gt=0)
    codec: str | None = Field(default=None, description="视频编码，如 h264")

    # 分析结果
    scenes: list[SceneInfo] = Field(
        default_factory=list,
        description="镜头列表；图片素材固定为1个scene"
    )
    transcript: list[TranscriptSegment] = Field(
        default_factory=list,
        description="音频转写（仅视频且有音轨时）"
    )

    @property
    def is_video(self) -> bool:
        return self.type == MaterialType.VIDEO

    @property
    def resolution(self) -> str | None:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return None
```

### 2.4 MaterialAnalysisReport —— 素材分析总报告

```python
class MaterialAnalysisReport(BaseModel):
    """素材分析 Agent 的完整输出"""
    model_config = ConfigDict(extra="forbid")

    materials: list[MaterialInfo] = Field(..., min_length=1)
    summary: str = Field(
        ...,
        description="对全部素材的整体摘要，供规划 Agent 参考"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="分析过程中发现的问题（如无音轨、画质过低）"
    )
```

---

## 3. 剪辑方案模型

### 3.1 TextStyle —— 文字样式

```python
class TextStyle(BaseModel):
    """字幕或文字贴纸的样式。
    注意：TimelineSegment 中每个片段都有独立的 subtitle_style / text_overlay_style，
    可逐片段覆盖；此处默认值仅在 Agent 未指定样式时作为兜底。"""
    model_config = ConfigDict(extra="forbid")

    font: str = Field(default="Sans", description="字体名称")
    font_size: int = Field(default=36, ge=8, le=200)
    color: str = Field(default="white", description="文字颜色，支持颜色名或十六进制")
    stroke_color: str | None = Field(default="black", description="描边颜色")
    stroke_width: int = Field(default=2, ge=0, le=10)
    position: SubtitlePosition = Field(default=SubtitlePosition.BOTTOM)
    alignment: TextAlign = Field(default=TextAlign.CENTER)
    margin_v: int = Field(default=40, ge=0, description="垂直边距（像素）")
```

### 3.2 TimelineSegment —— 时间轴片段

```python
class TimelineSegment(BaseModel):
    """时间轴上的一个片段：从某素材取一段时间，经过转场/字幕/变速处理"""
    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="片段在时间轴上的序号，从0开始")
    material_id: str = Field(..., pattern=r"^mat_[0-9a-f]{6}$")

    # 素材内取段（图片素材 source_start=0, source_end=该图片展示时长）
    source_start: float = Field(default=0.0, ge=0)
    source_end: float = Field(..., gt=0, description="素材内结束时间（秒）")

    # 转场
    transition_in: TransitionType = Field(
        default=TransitionType.CUT,
        description="与前一个片段之间的转场"
    )
    transition_out: TransitionType = Field(
        default=TransitionType.CUT,
        description="与后一个片段之间的转场"
    )
    transition_duration: float = Field(
        default=0.5, ge=0, le=3.0,
        description="转场时长（秒），CUT 时忽略"
    )

    # 字幕（叠加在画面上的同期字幕）
    subtitle: str | None = Field(default=None, description="字幕文本")
    subtitle_style: TextStyle | None = Field(default=None)

    # 文字贴纸（独立于字幕的营销文案）
    text_overlay: str | None = Field(default=None, description="文字贴纸内容")
    text_overlay_style: TextStyle | None = Field(default=None)

    # 速度
    speed: float = Field(default=1.0, gt=0, le=4.0, description="播放倍速")

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def output_duration(self) -> float:
        return self.source_duration / self.speed
```

### 3.3 TimelinePlan —— 完整剪辑方案

```python
class TimelinePlan(BaseModel):
    """剪辑规划 Agent 的完整输出，工具执行 Agent 直接消费此结构"""
    model_config = ConfigDict(extra="forbid")

    video_type: VideoType
    target_duration: float = Field(..., gt=0, le=600, description="目标成片时长（秒）")
    aspect_ratio: AspectRatio = Field(default=AspectRatio.VERTICAL)
    style: str = Field(..., description="风格标识，如 fast_paced_energetic")

    segments: list[TimelineSegment] = Field(..., min_length=1)

    bgm_style: str | None = Field(
        default=None,
        description="背景音乐风格描述，如 upbeat_electronic；None 表示不加BGM"
    )
    bgm_volume: float = Field(default=0.15, ge=0, le=1.0)

    voiceover_text: str | None = Field(
        default=None,
        description="TTS 配音文本；None 表示不配音"
    )

    rationale: str = Field(
        ...,
        description="规划理由：为什么这样选素材、排顺序、定节奏"
    )
    rag_references: list[str] = Field(
        default_factory=list,
        description="RAG 检索到的模板名称/ID，用于追溯"
    )

    def estimated_duration(self) -> float:
        """估算成片时长（各片段输出时长之和减去转场重叠）"""
        total = sum(seg.output_duration for seg in self.segments)
        # 转场会重叠前后两个片段，减去转场时长
        for seg in self.segments[1:]:
            if seg.transition_in != TransitionType.CUT:
                total -= seg.transition_duration
        return total
```

---

## 4. 执行结果模型

### 4.1 ToolCallRecord —— 单次工具调用记录

```python
class ToolCallRecord(BaseModel):
    """工具执行过程中一次工具调用的完整记录"""
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(..., pattern=r"^call_[0-9a-f]{8}$")
    tool_name: str
    arguments: dict
    status: ToolCallStatus
    result: dict | None = Field(default=None, description="工具返回的结构化结果")
    error_message: str | None = Field(default=None)
    duration_ms: int = Field(..., ge=0)
    thought: str | None = Field(
        default=None,
        description="调用该工具前的意图说明（如一站式渲染/降级步骤）"
    )
```

### 4.2 ExecutionResult —— 执行阶段最终结果

```python
class ExecutionResult(BaseModel):
    """工具执行 Agent 的输出"""
    model_config = ConfigDict(extra="forbid")

    success: bool
    output_path: str | None = Field(default=None, description="成片文件绝对路径")
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    intermediate_files: list[str] = Field(
        default_factory=list,
        description="中间产物路径（临时片段等），用于清理"
    )
    error_message: str | None = Field(default=None)
    total_duration_ms: int = Field(default=0, ge=0)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def retry_count(self) -> int:
        return sum(1 for c in self.tool_calls if c.status == ToolCallStatus.RETRY)
```

---

## 5. 质量评估模型

### 5.1 DimensionScore —— 单维度评分

```python
class DimensionScore(BaseModel):
    """单个评估维度的评分"""
    model_config = ConfigDict(extra="forbid")

    dimension: EvaluationDimension
    score: float = Field(..., ge=1.0, le=5.0, description="1-5 分")
    reason: str = Field(..., description="评分理由")
    metric_value: float | None = Field(
        default=None,
        description="程序化指标值（如 NIQE 分数、时长偏差秒数）"
    )
```

### 5.2 EvaluationResult —— 评估结果

```python
class EvaluationResult(BaseModel):
    """质量评估 Agent 的输出"""
    model_config = ConfigDict(extra="forbid")

    scores: list[DimensionScore] = Field(..., min_length=1)
    overall: float = Field(..., ge=1.0, le=5.0, description="加权总分")
    passed: bool = Field(..., description="overall >= 3.5 视为通过")
    issues: list[str] = Field(
        default_factory=list,
        description="发现的具体问题，如 ['第2个转场处有0.3秒黑帧']"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="改进建议，传回规划 Agent 指导下一轮"
    )
    iteration: int = Field(default=1, ge=1, description="这是第几轮评估")
```

---

## 6. Agent 消息模型

```python
class AgentMessage(BaseModel):
    """Agent 之间传递的自然语言消息（通过 LangGraph state 共享）"""
    model_config = ConfigDict(extra="forbid")

    from_agent: AgentName
    to_agent: AgentName
    content: str = Field(..., min_length=1)
    round: int = Field(default=1, ge=1, description="对话轮次")
```
```

---

## 7. Agent 共享状态（LangGraph）

```python
from typing import Annotated
from operator import add
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    """LangGraph 状态图中所有节点共享的状态。
    每个 Agent 节点读取需要的字段、产出自己的字段。
    messages 字段使用 add reducer 实现追加而非覆盖。"""
    model_config = ConfigDict(extra="forbid")

    # ---- 输入 ----
    user_prompt: str = Field(..., description="用户的自然语言需求")
    material_paths: list[str] = Field(
        default_factory=list,
        description="用户上传的素材文件绝对路径列表"
    )

    # ---- 各 Agent 产出 ----
    analysis_report: MaterialAnalysisReport | None = None
    edit_plan: TimelinePlan | None = None
    execution_result: ExecutionResult | None = None
    evaluation_result: EvaluationResult | None = None

    # ---- 流程控制 ----
    iteration: int = Field(default=1, ge=1, description="当前迭代轮数")
    max_iterations: int = Field(default=3, ge=1, le=5)

    # ---- Agent 间消息（追加语义）----
    messages: Annotated[list[AgentMessage], add] = Field(default_factory=list)

    # ---- 可观测性 ----
    agent_status: dict[str, AgentStatus] = Field(
        default_factory=dict,
        description="各 Agent 当前状态，key 为 AgentName.value"
    )
    errors: list[str] = Field(default_factory=list)
```

**状态流转规则**：

```
material_paths + user_prompt
        │
        ▼
  [analyzer] ──→ analysis_report
        │
        ▼
  [planner]  ←── RAG 模板检索（外部调用，不写 state）
        │
        ▼
  [executor] ──→ execution_result
        │
        ▼
  [evaluator] ──→ evaluation_result
        │
        ├── passed=True  ──→ END
        │
        └── passed=False & iteration < max_iterations
                │
                ▼
           iteration += 1 → [planner]（携带 issues + suggestions）
```

---

## 8. MCP 工具输入输出模型

> 以下模型对应 MCP Server 注册的 12 个工具。
> 每个工具的输入模型即 MCP 工具的参数 schema（由 Pydantic 自动生成 JSON Schema）。
> 所有路径参数在工具入口处校验工作目录边界。

### 8.1 通用

```python
class ToolResult(BaseModel):
    """所有 MCP 工具统一返回结构"""
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: dict | None = Field(default=None, description="工具返回的业务数据")
    error: str | None = Field(default=None, description="失败时的错误信息")
```

### 8.2 get_video_info

```python
class GetVideoInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="视频文件路径")

class VideoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int | None = None
    has_audio: bool
    audio_codec: str | None = None
    size_bytes: int
```

### 8.3 cut_clip

```python
class CutClipInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="源视频路径")
    start: float = Field(..., ge=0, description="起始时间（秒）")
    end: float = Field(..., gt=0, description="结束时间（秒）")
    output_path: str | None = Field(
        default=None,
        description="输出路径；None 则自动生成在工作目录"
    )
```

### 8.4 concat_videos

```python
class ConcatVideosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paths: list[str] = Field(..., min_length=2, description="待拼接视频路径列表")
    transition: TransitionType = Field(default=TransitionType.CUT)
    transition_duration: float = Field(default=0.5, ge=0, le=3.0)
    target_resolution: AspectRatio | None = Field(
        default=None,
        description="统一输出分辨率；None 时取第一个视频的分辨率，"
                    "其余视频等比缩放+黑边填充。不同分辨率的视频必须统一后才能拼接"
    )
    output_path: str | None = None
```

### 8.5 add_subtitle

```python
class SubtitleEntry(BaseModel):
    """单条字幕"""
    model_config = ConfigDict(extra="forbid")
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str = Field(..., min_length=1)

class AddSubtitleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    entries: list[SubtitleEntry] = Field(..., min_length=1)
    style: TextStyle | None = None
    output_path: str | None = None
```

### 8.6 add_text_overlay

```python
class AddTextOverlayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    text: str = Field(..., min_length=1)
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    style: TextStyle | None = None
    output_path: str | None = None
```

### 8.7 add_transition

```python
class AddTransitionInput(BaseModel):
    """给单个视频在开头或结尾加转场效果（用于独立处理）。
    注意：多数转场在 concat 时通过 xfade 一并实现，此工具用于特殊场景。"""
    model_config = ConfigDict(extra="forbid")
    path: str
    transition: TransitionType
    duration: float = Field(default=0.5, ge=0.1, le=3.0)
    phase: str = Field(
        default="in",
        pattern="^(in|out)$",
        description="in=开头淡入，out=结尾淡出"
    )
    output_path: str | None = None
```

### 8.8 add_bgm

```python
class AddBgmInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="视频路径")
    audio_path: str = Field(..., description="背景音乐路径")
    volume: float = Field(default=0.15, ge=0, le=1.0, description="BGM 音量")
    loop: bool = Field(
        default=True,
        description="音乐短于视频时是否循环播放；False 则视频尾部无BGM"
    )
    keep_original_audio: bool = Field(
        default=True,
        description="是否保留视频原音轨。True=混音（原音+BGM），"
                    "False=替换原音轨（仅BGM）。带货视频有人声时必须为True"
    )
    output_path: str | None = None
```

**BGM 时长处理规则**：
- 音乐短于视频 + `loop=True`：音乐循环至视频时长
- 音乐短于视频 + `loop=False`：音乐播放完毕后视频尾部静音
- 音乐长于视频：音乐截断至视频时长
- `keep_original_audio=True`：使用 FFmpeg `amix` 滤镜混合原音轨和BGM，BGM 按 `volume` 缩放

### 8.9 image_to_video

```python
class ImageToVideoInput(BaseModel):
    """将一张或多张图片转为视频片段（支持 Ken Burns 推拉摇移效果）"""
    model_config = ConfigDict(extra="forbid")
    image_paths: list[str] = Field(..., min_length=1)
    duration_per_image: float = Field(default=3.0, gt=0, le=30.0)
    ken_burns: bool = Field(default=True, description="是否启用 Ken Burns 效果")
    resolution: AspectRatio = Field(default=AspectRatio.VERTICAL)
    fps: int = Field(default=30, ge=1, le=60)
    output_path: str | None = None
```

### 8.10 extract_audio

```python
class ExtractAudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    output_path: str | None = None
    format: str = Field(default="wav", pattern="^(wav|mp3|aac)$")
```

### 8.11 transcribe_audio

```python
class TranscribeAudioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_path: str
    model_size: str = Field(
        default="base",
        pattern="^(tiny|base|small|medium|large-v3)$",
        description="faster-whisper 模型大小"
    )
    language: str = Field(default="zh", description="语言代码，zh/en/auto")

class TranscriptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segments: list[TranscriptSegment]
    full_text: str
    language: str
```

### 8.12 detect_scenes

```python
class DetectScenesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    threshold: float = Field(
        default=27.0, ge=1.0, le=100.0,
        description="PySceneDetect ContentDetector 的内容变化分数阈值。"
                    "该分数衡量相邻帧在 HSV 色彩空间的平均差异，"
                    "27.0 是官方默认值：低于27会检出更多镜头（包括缓慢运镜），"
                    "高于27只检出剧烈切换。vlog/访谈建议20-25，快剪建议27-30"
    )
    min_scene_len: float = Field(
        default=0.6, ge=0.1,
        description="最短镜头时长（秒）"
    )

class SceneBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float
    end: float
```

### 8.13 create_video_from_timeline（一站式渲染）

```python
class CreateVideoFromTimelineInput(BaseModel):
    """一站式时间轴渲染：输入完整剪辑方案，内部组合调用上述工具。
    这是给 Agent 用的"高级接口"，简单场景可以只调这一个工具。
    output_path 为 None 时按 0.5 节约定自动生成。"""
    model_config = ConfigDict(extra="forbid")
    plan: TimelinePlan
    material_map: dict[str, str] = Field(
        ...,
        description="material_id → 文件绝对路径的映射"
    )
    output_path: str | None = None
```

---

## 9. 配置模型

```python
class LLMSettings(BaseModel):
    """LLM 相关配置"""
    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = Field(default=LLMProvider.DEEPSEEK)
    deepseek_api_key: str | None = None
    deepseek_model: str = Field(default="deepseek-chat")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")

    qwen_api_key: str | None = Field(default=None, description="DashScope API Key")
    qwen_model: str = Field(default="qwen-plus")
    qwen_vl_model: str = Field(default="qwen-vl-max")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")

    openai_api_key: str | None = None
    openai_model: str = Field(default="gpt-4o-mini")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    temperature: float = Field(default=0.1, ge=0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    timeout: int = Field(default=60, ge=5, description="API 超时（秒）")
    max_retries: int = Field(default=3, ge=0, le=10)


class VideoSettings(BaseModel):
    """视频处理相关配置"""
    model_config = ConfigDict(extra="forbid")

    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")
    default_fps: int = Field(default=30)
    default_crf: int = Field(default=23, ge=0, le=51, description="输出质量，越小越清晰")
    default_video_codec: str = Field(default="libx264")
    default_audio_codec: str = Field(default="aac")
    scene_threshold: float = Field(default=27.0)
    frame_extract_interval: float = Field(default=2.0, description="关键帧提取间隔（秒）")
    whisper_model_size: str = Field(default="base")
    whisper_device: str = Field(default="cpu", pattern="^(cpu|cuda)$")


class RAGSettings(BaseModel):
    """RAG 相关配置"""
    model_config = ConfigDict(extra="forbid")

    persist_dir: str = Field(default="data/vector_db")
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    collection_name: str = Field(default="mavea_templates")
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)
    vector_weight: float = Field(default=0.7, ge=0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0, le=1.0)


class MCPSettings(BaseModel):
    """MCP Server 相关配置"""
    model_config = ConfigDict(extra="forbid")

    server_name: str = Field(default="mavea-mcp")
    stdio_enabled: bool = Field(default=True)
    sse_enabled: bool = Field(default=False)
    sse_host: str = Field(default="127.0.0.1")
    sse_port: int = Field(default=8765, ge=1, le=65535)
    tool_timeout: int = Field(default=30, ge=1, description="单个工具执行超时（秒）")
    max_tool_steps: int = Field(default=20, ge=1, description="单次流水线的工具调用步数上限")


class Settings(BaseModel):
    """全局配置，由 config.py 从环境变量 + .env 文件加载。
    使用 pydantic-settings 的 BaseSettings 在代码中实现，
    此处仅定义结构契约。"""
    model_config = ConfigDict(extra="forbid")

    workspace_dir: str = Field(
        default="./workspace",
        description="工作目录，所有输入输出文件限制在此目录内"
    )
    output_dir: str = Field(default="./workspace/output")
    temp_dir: str = Field(default="./workspace/temp")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
```

---

## 10. 跨模块函数签名约定

### 10.1 LLM 抽象层（llm/base.py）

```python
from typing import Any, Protocol


class BaseLLM(Protocol):
    """所有 LLM 客户端的统一接口。具体实现在 llm/deepseek.py 等文件中。"""

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """纯文本生成，返回 assistant 回复文本。"""
        ...

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        *,
        temperature: float | None = None,
    ) -> BaseModel:
        """结构化输出，返回 response_model 的实例。
        实现方式优先使用 LLM 的 structured output / function calling 能力，
        降级方案为手动 JSON 解析 + 重试。"""
        ...

    def generate_vision(
        self,
        text: str,
        image_paths: list[str],
        *,
        temperature: float | None = None,
    ) -> str:
        """多模态：根据图片生成文本描述。
        不支持视觉的 provider 应抛出 NotImplementedError。"""
        ...
```

### 10.2 Agent 节点签名（agents/）

```python
from typing import Callable, TypeVar

# 每个 Agent 节点是一个同步函数，接收 GraphState，返回部分 state 的 dict
# LangGraph 会将返回的 dict merge 回 state
AgentNode = Callable[[GraphState], dict[str, Any]]
```

### 10.3 RAG 检索接口（rag/retriever.py）

```python
class TemplateRetriever(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """检索剪辑模板。
        返回 list of dict，每个 dict 至少包含:
        - name: str
        - content: str（模板 JSON 字符串）
        - score: float
        """
        ...
```

---

## 11. 变更记录

| 日期 | 版本 | 变更 | 影响模块 |
|---|---|---|---|
| 2026-08-25 | 0.1.0 | 初始版本，定义全部枚举、模型、状态、工具I/O | 全部 |
| 2026-08-25 | 0.1.1 | 新增0.5输出路径约定；ConcatVideosInput加target_resolution；AddBgmInput加keep_original_audio及BGM时长规则；detect_scenes阈值说明细化；TextStyle加逐片段覆盖说明；create_video_from_timeline引用0.5约定 | mcp/tools, video/ffmpeg |
