"""质量评估 Agent。

职责：
1. 无参考画质评估（模糊度、黑帧、响度/爆音）——对成片做程序化检测
2. LLM-as-Judge 评审“剪辑方案 vs 用户需求”的匹配度（只看方案文本，不看画面）
3. 程序化检查（时长准确性）
4. 输出 EvaluationResult，决定是否需要迭代
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, Field

from mavea.llm.base import get_llm
from mavea.models import (
    DimensionScore,
    EvaluationDimension,
    EvaluationResult,
    ExecutionResult,
    TimelinePlan,
)
from mavea.progress import report_progress
from mavea.video.metrics import (
    analyze_audio_quality,
    check_duration_accuracy,
    compute_no_reference_score,
    detect_black_frames,
)

logger = structlog.get_logger(__name__)


class _JudgeResult(BaseModel):
    """LLM-as-Judge 结构化结果（替代正则抠 JSON，避免解析失败退化为默认分）。"""
    model_config = {"extra": "ignore"}
    score: float = Field(ge=1.0, le=5.0, description="内容匹配综合分 1-5")
    reason: str = ""
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


_STAGE = "evaluator"

def _log(msg: str, fraction: float = 0.5):
    print(f"[Evaluator] {msg}", flush=True)
    logger.info("evaluator.progress", msg=msg)
    report_progress(_STAGE, msg, fraction)


def _score_to_5(nr_score: float) -> float:
    """将无参考画质分数映射到 1-5。"""
    return max(1.0, min(5.0, nr_score))


def evaluate_quality(
    execution: ExecutionResult,
    plan: TimelinePlan,
    user_prompt: str,
    iteration: int = 1,
) -> EvaluationResult:
    """评估成片质量。"""
    scores: list[DimensionScore] = []
    issues: list[str] = []
    suggestions: list[str] = []

    if execution is None or not execution.success or not execution.output_path:
        return EvaluationResult(
            scores=[DimensionScore(
                dimension=EvaluationDimension.CONTENT_MATCH,
                score=1.0,
                reason="执行失败，无成片",
            )],
            overall=1.0,
            passed=False,
            issues=["视频执行失败"],
            suggestions=["检查素材格式和FFmpeg执行日志"],
            iteration=iteration,
        )

    output_path = execution.output_path

    # ---- 1. 技术画质 ----
    try:
        nr_score = compute_no_reference_score(output_path)
        score_5 = _score_to_5(nr_score)
        reason = f"无参考画质评分 {nr_score:.1f}（模糊度+亮度综合）"
        if score_5 < 3.0:
            issues.append("成片画质偏低，可能存在模糊或过暗问题")
            suggestions.append("检查源素材质量，避免过度放大低分辨率素材")
        scores.append(DimensionScore(
            dimension=EvaluationDimension.TECHNICAL_QUALITY,
            score=score_5,
            reason=reason,
            metric_value=nr_score,
        ))
    except Exception as e:
        logger.warning("evaluator.technical_failed", error=str(e))
        scores.append(DimensionScore(
            dimension=EvaluationDimension.TECHNICAL_QUALITY,
            score=3.0,
            reason=f"画质评估失败: {e}",
        ))

    # ---- 1b. 音频响度 / 爆音（v8）----
    try:
        audio = analyze_audio_quality(output_path)
        if not audio.get("has_audio"):
            issues.append("成片没有音轨")
            suggestions.append("检查背景音乐是否成功混入")
        else:
            mv = audio.get("mean_volume_db")
            xv = audio.get("max_volume_db")
            if audio.get("too_quiet"):
                issues.append(f"配乐过轻（平均 {mv:.1f}dB），可能听不见")
                suggestions.append("提高 BGM 混音音量，避免 amix 归一化衰减")
            if audio.get("clipping"):
                issues.append(f"存在爆音/削波风险（峰值 {xv:.1f}dB）")
                suggestions.append("降低混音总音量或加 limiter，避免顶到 0dB")
            logger.info("evaluator.audio", mean_db=mv, max_db=xv,
                        quiet=audio.get("too_quiet"), clip=audio.get("clipping"))
    except Exception as e:
        logger.warning("evaluator.audio_failed", error=str(e))

    # ---- 2. 黑帧/流畅度 ----
    try:
        black_frames = detect_black_frames(output_path)
        if black_frames:
            issues.append(f"检测到 {len(black_frames)} 处黑帧，时间点: {[f'{t:.1f}s' for t in black_frames[:5]]}")
            suggestions.append("检查转场参数，转场时长可能超过了片段时长")
            scores.append(DimensionScore(
                dimension=EvaluationDimension.FLUENCY,
                score=2.0,
                reason=f"检测到 {len(black_frames)} 处黑帧",
                metric_value=float(len(black_frames)),
            ))
        else:
            scores.append(DimensionScore(
                dimension=EvaluationDimension.FLUENCY,
                score=4.5,
                reason="未检测到黑帧",
                metric_value=0.0,
            ))
    except Exception as e:
        scores.append(DimensionScore(
            dimension=EvaluationDimension.FLUENCY,
            score=3.0,
            reason=f"流畅度检查失败: {e}",
        ))

    # ---- 3. 时长准确性 ----
    try:
        passed, actual = check_duration_accuracy(
            output_path, plan.target_duration, tolerance=0.15
        )
        if passed:
            scores.append(DimensionScore(
                dimension=EvaluationDimension.DURATION_ACCURACY,
                score=5.0,
                reason=f"成片时长 {actual:.1f}s，目标 {plan.target_duration}s",
                metric_value=actual,
            ))
        else:
            issues.append(f"成片时长 {actual:.1f}s 与目标 {plan.target_duration}s 偏差较大")
            suggestions.append("调整片段数量或每个片段的取段时长")
            scores.append(DimensionScore(
                dimension=EvaluationDimension.DURATION_ACCURACY,
                score=2.5,
                reason=f"时长偏差 {abs(actual - plan.target_duration):.1f}s",
                metric_value=actual,
            ))
    except Exception as e:
        scores.append(DimensionScore(
            dimension=EvaluationDimension.DURATION_ACCURACY,
            score=3.0,
            reason=f"时长检查失败: {e}",
        ))

    # ---- 4. 方案-需求匹配度（LLM-as-Judge：评方案文本，不评成片画面）----
    try:
        llm = get_llm()
        segments_desc = "; ".join(
            f"片段{s.index}: 素材{s.material_id} {s.source_start:.1f}-{s.source_end:.1f}s"
            + (f" 字幕:{s.subtitle}" if s.subtitle else "")
            + (f" 画面大字卡:{s.text_overlay}" if s.text_overlay else "")
            for s in plan.segments
        )
        judge_prompt = f"""作为视频质量评审员，你只能看到剪辑方案的文本描述（看不到成片画面），
请据此评估该方案是否满足用户需求，不要臆造画面内容。
注意：“画面大字卡”是会真实烧录到画面上的卖点文字/行动号召（如结尾“点击购买”），
评估购买引导、卖点呈现时必须把大字卡视为成片中真实存在的内容，不得判为缺失。

用户需求：{user_prompt}
视频类型：{plan.video_type.value}
目标时长：{plan.target_duration}s
剪辑方案：{segments_desc}
规划理由：{plan.rationale}

请从以下维度打分（1-5分）：
1. 内容匹配度：选片和顺序是否符合用户需求
2. 节奏合理性：片段时长和转场是否合适
3. 卖点表达：带货视频是否突出了产品卖点

返回结构化结果：score(1-5)、reason、issues、suggestions。"""

        # 结构化输出：Schema 强约束，不再用正则抠 JSON / 失败退化为默认分
        judge = llm.generate_structured(
            messages=[{"role": "user", "content": judge_prompt}],
            response_model=_JudgeResult,
            temperature=0.1,
        )
        content_score = max(1.0, min(5.0, float(judge.score)))
        scores.append(DimensionScore(
            dimension=EvaluationDimension.CONTENT_MATCH,
            score=content_score,
            reason=judge.reason or "LLM评审",
        ))
        issues.extend([i for i in judge.issues if i])
        suggestions.extend([s for s in judge.suggestions if s])
    except Exception as e:
        logger.warning("evaluator.llm_judge_failed", error=str(e))
        scores.append(DimensionScore(
            dimension=EvaluationDimension.CONTENT_MATCH,
            score=3.5,
            reason=f"LLM评审不可用: {e}",
        ))

    # ---- 5. 字幕质量（程序化检查：底部字幕或卖点大字卡都算文字呈现）----
    has_subtitle = any(s.subtitle or s.text_overlay for s in plan.segments)
    if has_subtitle:
        scores.append(DimensionScore(
            dimension=EvaluationDimension.SUBTITLE_QUALITY,
            score=4.0,
            reason="方案包含字幕，已烧录到成片",
        ))
    else:
        scores.append(DimensionScore(
            dimension=EvaluationDimension.SUBTITLE_QUALITY,
            score=3.0,
            reason="无字幕（带货视频建议加字幕）",
        ))

    # 加权总分
    weights = {
        EvaluationDimension.TECHNICAL_QUALITY: 0.15,
        EvaluationDimension.CONTENT_MATCH: 0.35,
        EvaluationDimension.DURATION_ACCURACY: 0.15,
        EvaluationDimension.FLUENCY: 0.20,
        EvaluationDimension.SUBTITLE_QUALITY: 0.15,
    }
    overall = sum(
        s.score * weights.get(s.dimension, 0.2) for s in scores
    ) / sum(weights.get(s.dimension, 0.2) for s in scores)
    overall = round(overall, 1)
    passed = overall >= 3.5

    return EvaluationResult(
        scores=scores,
        overall=overall,
        passed=passed,
        issues=issues,
        suggestions=suggestions,
        iteration=iteration,
    )


async def run(state) -> dict[str, Any]:
    """LangGraph 节点入口。"""
    from mavea.models import AgentName, AgentStatus

    logger.info("agent.evaluator.start", iteration=state.iteration)
    state.agent_status[AgentName.EVALUATOR.value] = AgentStatus.RUNNING

    try:
        result = evaluate_quality(
            state.execution_result,
            state.edit_plan,
            state.user_prompt,
            state.iteration,
        )
        state.agent_status[AgentName.EVALUATOR.value] = AgentStatus.COMPLETED
        verdict = "通过" if result.passed else "需返工"
        _log(f"评估完成: 总分 {result.overall}/5.0, {verdict}", 1.0)
        logger.info("agent.evaluator.done", overall=result.overall, passed=result.passed)
        # 追加历轮分数，供 graph 做“分数收敛则提前停止”判定
        score_history = list(getattr(state, "score_history", []) or [])
        score_history.append(result.overall)
        return {
            "evaluation_result": result,
            "score_history": score_history,
            "agent_status": state.agent_status,
        }
    except Exception as e:
        state.agent_status[AgentName.EVALUATOR.value] = AgentStatus.FAILED
        state.errors.append(f"评估失败: {e}")
        logger.error("agent.evaluator.failed", error=str(e))
        return {"agent_status": state.agent_status, "errors": state.errors}
