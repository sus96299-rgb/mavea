"""OpenAI 官方 API 客户端（备选，支持 GPT-4o 视觉）。"""

from __future__ import annotations

from mavea.llm.openai_compat_base import OpenAICompatClient


class OpenAIClient(OpenAICompatClient):
    """OpenAI API 客户端。

    支持视觉（gpt-4o-mini/gpt-4o），作为 Qwen-VL 的备选。
    """

    def __init__(self):
        from mavea.config import get_settings
        settings = get_settings()
        super().__init__(
            api_key=settings.llm.openai_api_key,
            base_url=settings.llm.openai_base_url,
            model=settings.llm.openai_model,
            vision_model=settings.llm.openai_model,
            supports_vision=True,
        )

    def _get_api_key(self, settings):
        return settings.llm.openai_api_key

    def _get_base_url(self, settings):
        return settings.llm.openai_base_url

    def _get_model(self, settings):
        return settings.llm.openai_model
