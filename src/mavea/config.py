"""MAVEA 全局配置。

通过 pydantic-settings 从环境变量和 .env 文件加载配置。
环境变量前缀 MAVEA_，嵌套配置用双下划线分隔，例如 MAVEA_LLM__DEEPSEEK_API_KEY。
"""

from __future__ import annotations

import os

# 必须在导入任何 huggingface 相关库之前设置国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM 相关配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_LLM__", env_file=".env", extra="ignore")

    provider: str = Field(default="deepseek")
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    qwen_api_key: str | None = None
    qwen_model: str = "qwen-plus"
    qwen_vl_model: str = "qwen-vl-max"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    timeout: int = Field(default=60, ge=5)
    max_retries: int = Field(default=3, ge=0, le=10)


class VideoSettings(BaseSettings):
    """视频处理相关配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_VIDEO__", env_file=".env", extra="ignore")

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    default_fps: int = 30
    default_crf: int = Field(default=23, ge=0, le=51)
    default_video_codec: str = "libx264"
    default_audio_codec: str = "aac"
    scene_threshold: float = 27.0
    frame_extract_interval: float = 2.0
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    skip_transcribe: bool = False
    hf_mirror: str = "https://hf-mirror.com"


class RAGSettings(BaseSettings):
    """RAG 相关配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_RAG__", env_file=".env", extra="ignore")

    persist_dir: str = "data/vector_db"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    collection_name: str = "mavea_templates"
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_top_n: int = Field(default=3, ge=1, le=10)
    vector_weight: float = Field(default=0.7, ge=0, le=1)
    bm25_weight: float = Field(default=0.3, ge=0, le=1)


class MCPSettings(BaseSettings):
    """MCP Server 相关配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_MCP__", env_file=".env", extra="ignore")

    server_name: str = "mavea-mcp"
    sse_host: str = "127.0.0.1"
    sse_port: int = Field(default=8765, ge=1, le=65535)
    tool_timeout: int = Field(default=30, ge=1)
    max_tool_steps: int = Field(default=20, ge=1)


class APISettings(BaseSettings):
    """FastAPI 服务配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_API__", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class GradioSettings(BaseSettings):
    """Gradio WebUI 配置"""

    model_config = SettingsConfigDict(env_prefix="MAVEA_GRADIO__", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=7860, ge=1, le=65535)
    share: bool = False


class Settings(BaseSettings):
    """全局配置。单例通过 get_settings() 获取。"""

    model_config = SettingsConfigDict(
        env_prefix="MAVEA_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    workspace_dir: str = "./workspace"
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    api: APISettings = Field(default_factory=APISettings)
    gradio: GradioSettings = Field(default_factory=GradioSettings)

    @property
    def workspace_path(self) -> Path:
        p = Path(self.workspace_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = self.workspace_path / "output"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def temp_path(self) -> Path:
        p = self.workspace_path / "temp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def validate_path(self, path: str | Path) -> Path:
        """校验路径在工作目录内，防止路径穿越。返回解析后的绝对路径。"""
        p = Path(path).resolve()
        ws = self.workspace_path
        if not str(p).startswith(str(ws)):
            raise ValueError(f"路径 '{p}' 不在工作目录 '{ws}' 内")
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。lru_cache 保证只加载一次。"""
    return Settings()
