"""阿里通义千问客户端（DashScope OpenAI 兼容模式）。"""

from __future__ import annotations

from mavea.llm.openai_compat_base import OpenAICompatClient


class QwenClient(OpenAICompatClient):
    """Qwen API 客户端。

    支持视觉理解（qwen-vl-max），用于视频关键帧描述。
    文本模型默认 qwen-plus，视觉模型默认 qwen-vl-max。
    """

    def __init__(self):
        from mavea.config import get_settings
        settings = get_settings()
        super().__init__(
            api_key=settings.llm.qwen_api_key,
            base_url=settings.llm.qwen_base_url,
            model=settings.llm.qwen_model,
            vision_model=settings.llm.qwen_vl_model,
            supports_vision=True,
        )

    def _get_api_key(self, settings):
        return settings.llm.qwen_api_key

    def _get_base_url(self, settings):
        return settings.llm.qwen_base_url

    def _get_model(self, settings):
        return settings.llm.qwen_model
