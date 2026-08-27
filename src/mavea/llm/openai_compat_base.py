"""OpenAI 兼容客户端基类。

DeepSeek、Qwen（DashScope 兼容模式）、OpenAI 都使用相同的 OpenAI SDK 接口，
区别仅在 base_url、api_key、model 名称和是否支持视觉。
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import structlog
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from mavea.config import get_settings

logger = structlog.get_logger(__name__)


class OpenAICompatClient:
    """OpenAI 兼容客户端基类。子类设置以下类属性即可。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        supports_vision: bool = False,
    ):
        settings = get_settings()
        self._api_key = api_key or self._get_api_key(settings)
        self._base_url = base_url or self._get_base_url(settings)
        self._model = model or self._get_model(settings)
        self._vision_model = vision_model or self._model
        self._supports_vision = supports_vision
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens
        self._timeout = settings.llm.timeout
        self._max_retries = settings.llm.max_retries

        if not self._api_key:
            raise RuntimeError(
                f"{self.__class__.__name__} 缺少 API Key，"
                "请在 .env 中配置对应环境变量"
            )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    # 子类覆盖以下方法
    def _get_api_key(self, settings) -> str | None:
        raise NotImplementedError

    def _get_base_url(self, settings) -> str:
        raise NotImplementedError

    def _get_model(self, settings) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """纯文本生成。"""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except (APITimeoutError, RateLimitError, APIError) as e:
            logger.error("llm.generate.failed", model=self._model, error=str(e))
            raise

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        *,
        temperature: float | None = None,
    ) -> BaseModel:
        """结构化输出。

        策略：
        1. 尝试使用 response_format=json_object（OpenAI/DeepSeek 支持）
        2. 在 system prompt 中注入 JSON Schema 要求
        3. 解析返回的 JSON，失败时重试一次（附带修复提示）
        """
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        structured_prompt = (
            "你必须只返回一个合法的 JSON 对象，不要包含任何其他文字、解释或 markdown 代码块。\n"
            f"JSON 必须符合以下 Schema：\n{schema_str}\n"
            "直接返回 JSON，以 { 开头，以 } 结尾。"
        )

        # 把结构化要求注入 system message
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {
                "role": "system",
                "content": msgs[0]["content"] + "\n\n" + structured_prompt,
            }
        else:
            msgs.insert(0, {"role": "system", "content": structured_prompt})

        for attempt in range(2):
            try:
                # 第一次尝试用 response_format=json_object（部分兼容API可能不支持）
                create_kwargs = dict(
                    model=self._model,
                    messages=msgs,
                    temperature=temperature if temperature is not None else 0.05,
                    max_tokens=self._max_tokens,
                )
                if attempt == 0:
                    create_kwargs["response_format"] = {"type": "json_object"}
                response = self._client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content or ""
                return self._parse_json(content, response_model)
            except Exception as e:
                if attempt == 0:
                    logger.warning("llm.structured.retry", error=str(e)[:200])
                    # 追加修复提示（第二次尝试不带 response_format，兼容不支持 JSON mode 的 API）
                    msgs.append({"role": "assistant", "content": ""})
                    msgs.append({
                        "role": "user",
                        "content": f"上一次输出解析失败：{e}\n请严格按照 Schema 返回合法 JSON，不要包含 markdown 代码块。",
                    })
                else:
                    logger.error("llm.structured.failed", error=str(e))
                    raise

    def generate_vision(
        self,
        text: str,
        image_paths: list[str],
        *,
        temperature: float | None = None,
    ) -> str:
        """多模态：根据图片生成文本描述。"""
        if not self._supports_vision:
            raise NotImplementedError(
                f"{self.__class__.__name__} 不支持视觉理解，请使用 QwenClient 或 OpenAIClient"
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for img_path in image_paths:
            img_base64 = self._encode_image(img_path)
            mime = self._guess_mime(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_base64}"},
            })

        messages = [{"role": "user", "content": content}]
        response = self._client.chat.completions.create(
            model=self._vision_model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _encode_image(path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"图片不存在: {path}")
        return base64.b64encode(p.read_bytes()).decode("utf-8")

    @staticmethod
    def _guess_mime(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
            ext, "image/jpeg"
        )

    @staticmethod
    def _parse_json(content: str, model: type[BaseModel]) -> BaseModel:
        """从 LLM 输出中提取并解析 JSON。"""
        # 尝试直接解析
        try:
            return model.model_validate_json(content)
        except Exception:
            pass

        # 尝试提取 ```json ... ``` 代码块
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL)
        if match:
            try:
                return model.model_validate_json(match.group(1))
            except Exception:
                pass

        # 尝试找到第一个 { 到最后一个 }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            try:
                return model.model_validate_json(content[start : end + 1])
            except Exception as e:
                raise ValueError(f"JSON 解析失败: {e}\n原始内容: {content[:500]}") from e

        raise ValueError(f"未找到 JSON 内容: {content[:500]}")
