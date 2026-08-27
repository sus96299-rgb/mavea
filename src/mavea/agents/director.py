"""增强导演（受限工具自主决策 / 白名单 function-calling）。

主干编排仍是确定性的 Plan-and-Execute（保证成片稳定）；本模块在“后期增强”
这个**受限子任务**上引入 Agent 自主性：把一组已实现、可安全调用的增强能力
当作“工具池”，让大模型根据用户需求与素材分析结果，自行决定启用哪些、参数如何。

可用工具池（全部本地可落地、无外部 key）：
- beat_sync   自动卡点（按 BGM 节拍切换画面）
- beauty      轻度美颜（磨皮提亮）
- voiceover   AI 配音（edge-tts 旁白，需联网，免费）

这是“有限自主”：LLM 只能在上述白名单内选择，不能任意调用工具或执行代码，
任何决策失败都回退到“全不启用”，绝不阻断主流程。
"""
from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from mavea.llm.base import get_llm
from mavea.progress import report_progress

logger = structlog.get_logger(__name__)


class EnhancementDecision(BaseModel):
    """增强导演的结构化决策（等价于一次 function-calling 的入参）。"""
    model_config = {"extra": "ignore"}

    beat_sync: bool = Field(default=False, description="是否按音乐节拍卡点剪辑")
    beauty: bool = Field(default=False, description="是否对人像做轻度美颜")
    add_voiceover: bool = Field(default=False, description="是否需要 AI 旁白")
    voiceover_script: str | None = Field(
        default=None, description="不超过40字的中文旁白文案；不需要旁白则为 null"
    )
    reason: str = Field(default="", description="一句话说明为什么这样选")


# 保守的兜底决策：任何异常都不增强
_FALLBACK = EnhancementDecision(
    beat_sync=False, beauty=False, add_voiceover=False,
    voiceover_script=None, reason="增强决策不可用，回退默认",
)


def decide_enhancements(
    user_prompt: str,
    analysis_summary: str = "",
    num_materials: int = 0,
) -> EnhancementDecision:
    """让 LLM 在受限工具池内决定后期增强方案。失败返回保守兜底。"""
    try:
        llm = get_llm()
        system = (
            "你是短视频后期导演。你只能在以下已实现的增强工具中做选择，"
            "不得选择清单外的能力：\n"
            "1. beat_sync：用户提到卡点/节奏/踩点/动感，或背景音乐节奏感强时启用；\n"
            "2. beauty：素材主要是人像且用户在意好看/美颜/写真时启用；\n"
            "3. voiceover：用户需要旁白/解说/配音，或画面需要一句话串场时启用，"
            "并写出≤40字、贴合内容的中文旁白；纯音乐相册/用户没要求解说时不要加。\n"
            "原则：克制。拿不准就不启用，避免过度处理。"
        )
        user = (
            f"用户需求：{user_prompt}\n"
            f"素材数量：{num_materials}\n"
            f"素材分析摘要：{analysis_summary[:400]}\n"
            "请输出你的增强决策。"
        )
        report_progress("planner", "AI 增强导演正在选择后期工具…", 0.92)
        decision = llm.generate_structured(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=EnhancementDecision,
            temperature=0.2,
        )
        if decision.add_voiceover and decision.voiceover_script:
            decision.voiceover_script = decision.voiceover_script.strip()[:60]
        else:
            decision.voiceover_script = None
            decision.add_voiceover = False
        logger.info(
            "director.decision",
            beat=decision.beat_sync, beauty=decision.beauty,
            voice=decision.add_voiceover, reason=decision.reason,
        )
        return decision
    except Exception as e:  # noqa: BLE001
        logger.warning("director.failed", error=str(e))
        return _FALLBACK
