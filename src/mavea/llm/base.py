"""LLM 抽象层。

定义 BaseLLM 协议和通用消息类型，所有具体客户端（DeepSeek/Qwen/OpenAI）
都通过 OpenAI 兼容接口实现，共享 OpenAICompatClient 基类。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class BaseLLM(Protocol):
    """所有 LLM 客户端的统一接口。"""

    @property
    def model_name(self) -> str: ...

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

        优先使用 LLM 的 structured output / JSON mode，
        降级为手动 JSON 解析 + 重试。
        """
        ...

    def generate_vision(
        self,
        text: str,
        image_paths: list[str],
        *,
        temperature: float | None = None,
    ) -> str:
        """多模态：根据图片生成文本描述。

        Raises:
            NotImplementedError: 该 provider 不支持视觉
        """
        ...


def get_llm(provider: str | None = None) -> BaseLLM:
    """工厂函数：根据配置创建 LLM 客户端。

    Args:
        provider: deepseek/qwen/openai，None 则从配置读取
    """
    from mavea.config import get_settings

    settings = get_settings()
    p = (provider or settings.llm.provider).lower()

    if p == "deepseek":
        from mavea.llm.deepseek import DeepSeekClient
        return DeepSeekClient()
    elif p == "qwen":
        from mavea.llm.qwen import QwenClient
        return QwenClient()
    elif p == "openai":
        from mavea.llm.openai_compat import OpenAIClient
        return OpenAIClient()
    else:
        raise ValueError(f"不支持的 LLM provider: {p}，可选 deepseek/qwen/openai")


def get_vision_llm() -> BaseLLM:
    """获取支持视觉的 LLM 客户端（优先 Qwen-VL，降级 GPT-4o）。"""
    from mavea.config import get_settings

    settings = get_settings()
    if settings.llm.qwen_api_key:
        from mavea.llm.qwen import QwenClient
        return QwenClient()
    elif settings.llm.openai_api_key:
        from mavea.llm.openai_compat import OpenAIClient
        return OpenAIClient()
    else:
        raise RuntimeError(
            "视觉理解需要 Qwen API Key（MAVEA_LLM__QWEN_API_KEY）"
            "或 OpenAI API Key（MAVEA_LLM__OPENAI_API_KEY）"
        )
