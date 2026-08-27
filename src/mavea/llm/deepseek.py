"""DeepSeek API 客户端。"""

from __future__ import annotations

from mavea.llm.openai_compat_base import OpenAICompatClient


class DeepSeekClient(OpenAICompatClient):
    """DeepSeek Chat API（OpenAI 兼容接口）。

    特点：便宜、中文强、不支持视觉。
    模型：deepseek-chat（V3）
    """

    def __init__(self):
        super().__init__(supports_vision=False)

    def _get_api_key(self, settings):
        return settings.llm.deepseek_api_key

    def _get_base_url(self, settings):
        return settings.llm.deepseek_base_url

    def _get_model(self, settings):
        return settings.llm.deepseek_model
