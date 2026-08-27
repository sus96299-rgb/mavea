"""剪辑规划 Agent。

职责：
1. 理解用户意图（视频类型、时长、风格）
2. RAG 检索匹配的剪辑模板
3. 将素材镜头与脚本需求匹配
4. 生成结构化 TimelinePlan
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from mavea.llm.base import get_llm
from mavea.models import (
    MaterialAnalysisReport,
    SubtitlePosition,
    TextStyle,
    TimelinePlan,
    TimelineSegment,
    VideoType,
)
from mavea.progress import report_progress

logger = structlog.get_logger(__name__)


def _detect_video_type(prompt: str) -> VideoType:
    """从用户 prompt 中简单判断视频类型。"""
    if any(kw in prompt for kw in ["带货", "商品", "产品", "种草", "购买", "下单", "价格", "营销"]):
        return VideoType.ECOMMERCE_PROMO
    if any(kw in prompt for kw in ["教程", "教学", "步骤", "怎么做", "讲解"]):
        return VideoType.TUTORIAL
    if any(kw in prompt for kw in ["品牌", "宣传", "形象", "企业"]):
        return VideoType.BRAND_STORY
    return VideoType.VLOG_HIGHLIGHT


def _format_materials_for_prompt(report: MaterialAnalysisReport) -> str:
    """将素材分析报告格式化为 LLM 可读的文本。"""
    lines = []
    for mat in report.materials:
        if mat.is_video:
            lines.append(f"\n素材 {mat.id}（视频，{mat.duration}s，{mat.resolution}）：")
        else:
            lines.append(f"\n素材 {mat.id}（图片，{mat.resolution}）：")
        for i, scene in enumerate(mat.scenes):
            lines.append(
                f"  镜头{i}: [{scene.start:.1f}s-{scene.end:.1f}s] "
                f"{scene.description} [标签: {', '.join(scene.tags)}]"
            )
        if mat.transcript:
            transcript_text = " ".join(t.text for t in mat.transcript[:5])
            lines.append(f"  音频内容: {transcript_text[:200]}")
    return "\n".join(lines)


def _retrieve_templates(video_type: VideoType, prompt: str, top_k: int = 3) -> list[dict]:
    """从 RAG 检索 Top-k 匹配模板（向量+BM25 融合 → BGE Reranker 精排）。"""
    try:
        from mavea.rag.retriever import get_retriever
        retriever = get_retriever()
        retriever.load_templates()
        query = f"{video_type.value} {prompt}"
        results = retriever.retrieve(query, top_k=top_k)
        return results[:top_k] if results else []
    except Exception as e:
        logger.warning("planner.rag_failed", error=str(e))
    return []


def _parse_desired_seconds(user_prompt: str, fallback: float) -> float:
    """从“15秒/15s/15sec”里解析目标时长，解析不到用方案值。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒钟?|sec(?:onds?)?|s)(?![a-z])", user_prompt.lower())
    if m:
        try:
            val = float(m.group(1))
            if 1.0 <= val <= 600.0:
                return val
        except ValueError:
            pass
    return float(fallback)


def _normalize_plan(
    plan: TimelinePlan,
    analysis_report: MaterialAnalysisReport,
    user_prompt: str,
) -> TimelinePlan:
    """强制校正，避免模型擅自少用素材/时长不符：
    1. 用全所有上传素材（图片尤其要全用），漏的补成片段；
    2. 总时长锁定到用户指定秒数（如 15 秒），图片弹性均分，视频不超原长；
    3. 重排 index，统一硬切避免转场吃掉时长。
    """
    info = {m.id: m for m in analysis_report.materials}
    used = {seg.material_id for seg in plan.segments}
    next_idx = len(plan.segments)
    for mid, _mat in info.items():
        if mid not in used:
            plan.segments.append(TimelineSegment(
                index=next_idx, material_id=mid,
                source_start=0.0, source_end=3.0,
            ))
            next_idx += 1

    target = _parse_desired_seconds(user_prompt, plan.target_duration)
    plan.target_duration = target
    n = len(plan.segments)
    if n == 0:
        return plan

    share = target / n
    for seg in plan.segments:
        seg.speed = 1.0
        mat = info.get(seg.material_id)
        is_video = bool(mat and mat.is_video)
        d = share
        if is_video and getattr(mat, "duration", None):
            d = min(share, max(0.5, float(mat.duration)))
        seg.source_start = 0.0
        seg.source_end = round(max(0.4, d), 2)

    # 视频被原长截断产生的缺口，补到弹性图片片段上，迭代两轮收敛
    for _ in range(3):
        total = sum(seg.source_end - seg.source_start for seg in plan.segments)
        diff = target - total
        elastic = [
            seg for seg in plan.segments
            if not (info.get(seg.material_id) and info[seg.material_id].is_video)
        ]
        if not elastic or abs(diff) < 0.05:
            break
        add = diff / len(elastic)
        for seg in elastic:
            cur = seg.source_end - seg.source_start
            seg.source_start = 0.0
            seg.source_end = round(max(0.4, cur + add), 2)

    for i, seg in enumerate(plan.segments):
        seg.index = i
    return plan


def _ensure_bgm(plan: TimelinePlan, user_prompt: str) -> TimelinePlan:
    """保证方案带有可执行的背景音乐风格。

    LLM 经常漏填 bgm_style，导致成片完全无声。这里做确定性兜底：
    - 用户明确说不要音乐/静音 → 不加；
    - 否则根据描述推断风格，推断不出就给默认 ambient，绝不让成片静默。
    """
    muted_keywords = ["不要音乐", "不用音乐", "无音乐", "静音", "不要配乐",
                      "不用配乐", "别配", "不加音乐", "no music", "without music"]
    if any(k in user_prompt.lower() for k in [k.lower() for k in muted_keywords]):
        plan.bgm_style = None
        return plan

    if plan.bgm_style:
        return plan

    if any(k in user_prompt for k in ["轻快", "欢快", "活泼", "动感", "节奏", "活力"]):
        plan.bgm_style = "upbeat"
    elif any(k in user_prompt for k in ["舒缓", "安静", "温馨", "柔和", "治愈"]):
        plan.bgm_style = "calm"
    elif any(k in user_prompt for k in ["大气", "史诗", "震撼", "燃"]):
        plan.bgm_style = "epic"
    elif any(k in user_prompt for k in ["音乐", "配乐", "bgm", "BGM", "背景", "歌"]):
        plan.bgm_style = "ambient"
    else:
        # 用户没提音乐，但营销短视频默认铺一层轻配乐，避免成片无声
        plan.bgm_style = "ambient"
    return plan


# 电商带货节奏模板：主图稍长抓眼、细节图快切、结尾 CTA 行动号召
_COMMERCE_CTA_RULES = (("下单", "立即下单"), ("购买", "点击购买"), ("入手", "立即入手"))
_COMMERCE_CARD_STYLE = TextStyle(
    font_size=64, color="yellow",
    position=SubtitlePosition.BOTTOM,
    stroke_color="black", stroke_width=3,
)
_COMMERCE_CTA_STYLE = TextStyle(
    font_size=88, color="yellow",
    position=SubtitlePosition.CENTER,
    stroke_color="black", stroke_width=3,
)


def _promote_subtitles_to_cards(plan: TimelinePlan) -> None:
    """把底部字幕升级为卖点大字卡（不新增信息，只换呈现样式）。"""
    for seg in plan.segments:
        if seg.subtitle and not seg.text_overlay:
            seg.text_overlay = seg.subtitle.strip()[:12]
            seg.subtitle = None
            seg.text_overlay_style = _COMMERCE_CARD_STYLE


def _apply_commerce_template(
    plan: TimelinePlan,
    video_type: VideoType,
    analysis_report: MaterialAnalysisReport,
    user_prompt: str,
) -> TimelinePlan:
    """电商带货节奏模板（确定性后处理，不依赖 LLM 输出稳定性）。

    - 识别为电商带货时触发；
    - 纯图片素材：结尾复用最后一张商品图追加 CTA 段，主图/细节/CTA
      按 2.5/2.0/2.5 权重分配目标时长，形成长短节奏对比；
    - 混合视频素材：只把字幕升级为大字卡，不改时长（避免视频越界）；
    - 每段卖点字幕升级为黄色大字卡。
    """
    if video_type != VideoType.ECOMMERCE_PROMO:
        return plan
    if not plan.segments:
        return plan

    info = {m.id: m for m in analysis_report.materials}
    has_video = any(
        info.get(s.material_id) and info[s.material_id].is_video
        for s in plan.segments
    )
    _promote_subtitles_to_cards(plan)

    if has_video:
        return plan  # 混合素材保守处理：只升级字卡

    n = len(plan.segments)
    cta_text = "点击购买"
    for kw, txt in _COMMERCE_CTA_RULES:
        if kw in user_prompt:
            cta_text = txt
            break

    # 结尾 CTA：复用最后一张商品图（零新增素材），大字行动号召
    plan.segments.append(TimelineSegment(
        index=n, material_id=plan.segments[-1].material_id,
        source_start=0.0, source_end=2.5,
        text_overlay=cta_text,
        text_overlay_style=_COMMERCE_CTA_STYLE,
    ))

    # 权重时长：主图 2.5、细节 2.0、CTA 2.5，按比例锁定用户目标总时长
    weights = [2.5] + [2.0] * (n - 1) + [2.5]
    scale = plan.target_duration / sum(weights)
    for seg, wgt in zip(plan.segments, weights):
        seg.source_start = 0.0
        seg.source_end = round(max(0.8, wgt * scale), 2)
    for i, seg in enumerate(plan.segments):
        seg.index = i

    logger.info(
        "planner.commerce_template",
        segments=len(plan.segments), cta=cta_text,
        target=plan.target_duration,
    )
    return plan


def plan_editing(
    user_prompt: str,
    analysis_report: MaterialAnalysisReport,
    feedback: str | None = None,
) -> TimelinePlan:
    """剪辑规划主函数。

    Args:
        user_prompt: 用户需求
        analysis_report: 素材分析报告
        feedback: 上一轮评估的改进建议（迭代时传入）
    """
    llm = get_llm()
    video_type = _detect_video_type(user_prompt)
    templates = _retrieve_templates(video_type, user_prompt, top_k=3)
    template = templates[0] if templates else None
    rag_refs = [t.get("name", "") for t in templates if t.get("name")][:3]

    materials_text = _format_materials_for_prompt(analysis_report)

    template_hint = ""
    if template:
        template_hint = (
            f"\n参考剪辑模板《{template['name']}》：\n"
            f"{json.dumps(template.get('editing_params', {}), ensure_ascii=False, indent=2)}\n"
            f"模板结构：{json.dumps(template.get('structure', []), ensure_ascii=False)}\n"
        )

    feedback_hint = ""
    if feedback:
        feedback_hint = f"\n上一轮需要改进的问题：{feedback}\n请针对性修改剪辑方案。\n"

    system_prompt = f"""你是一个专业视频剪辑师。根据用户需求和素材分析结果，制定详细的剪辑方案。

规则：
1. 只能使用提供的素材ID，每个片段必须引用存在的 material_id
2. source_start/source_end 必须在素材时长范围内
3. 片段总时长应接近目标时长
4. 转场优先用 cut 和 fade，其他转场可能不兼容
5. 字幕要简短有力，每句不超过15字
6. 输出严格的 JSON，符合指定 Schema
7. bgm_style 必须是 upbeat/calm/epic/ambient 之一：用户提到配乐/音乐或需要烘托氛围时务必填写；轻快活泼用 upbeat，舒缓温馨用 calm，大气震撼用 epic，其余默认 ambient
{template_hint}{feedback_hint}"""

    user_message = f"""用户需求：{user_prompt}

可用素材：
{materials_text}

请制定剪辑方案。目标视频类型：{video_type.value}。
如果是带货视频，优先展示商品，前3秒用最吸引人的画面。
如果是Vlog集锦，保留有趣的对话和场景，节奏有张有弛。"""

    # 使用结构化输出
    plan = llm.generate_structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_model=TimelinePlan,
        temperature=0.3,
    )

    # 兜底：保证有背景音乐风格（除非用户明确不要音乐）
    plan = _ensure_bgm(plan, user_prompt)
    # 强制：用全部素材 + 总时长锁定到用户指定秒数
    plan = _normalize_plan(plan, analysis_report, user_prompt)
    # 电商带货节奏模板：快切权重 + 卖点大字卡 + 结尾 CTA
    plan = _apply_commerce_template(plan, video_type, analysis_report, user_prompt)
    # 校验素材ID存在
    valid_ids = {m.id for m in analysis_report.materials}
    for seg in plan.segments:
        if seg.material_id not in valid_ids:
            raise ValueError(f"剪辑方案引用了不存在的素材ID: {seg.material_id}")

    # 记录本次 RAG 实际引用的模板（向量+BM25 召回 → BGE 精排 Top3）
    if rag_refs:
        plan.rag_references = rag_refs

    logger.info(
        "planner.done",
        video_type=plan.video_type.value,
        segments=len(plan.segments),
        est_duration=round(plan.estimated_duration(), 1),
        target=plan.target_duration,
    )
    return plan


def run(state) -> dict[str, Any]:
    """LangGraph 节点入口。"""
    from mavea.models import AgentName, AgentStatus

    logger.info("agent.planner.start", iteration=state.iteration)
    state.agent_status[AgentName.PLANNER.value] = AgentStatus.RUNNING

    try:
        feedback = None
        if state.evaluation_result and state.evaluation_result.issues:
            feedback = "; ".join(state.evaluation_result.issues + state.evaluation_result.suggestions)

        plan = plan_editing(state.user_prompt, state.analysis_report, feedback)

        # AI 增强导演：在白名单工具池内自主决定后期增强（卡点/美颜/配音）
        extra: dict[str, Any] = {}
        if getattr(state, "ai_enhance", True):
            try:
                from mavea.agents.director import decide_enhancements
                _summary = getattr(state.analysis_report, "summary", "") or ""
                _n = len(state.analysis_report.materials) if state.analysis_report else 0
                decision = decide_enhancements(state.user_prompt, _summary, _n)
                extra["beat_sync"] = bool(getattr(state, "beat_sync", False) or decision.beat_sync)
                extra["beauty"] = bool(getattr(state, "beauty", False) or decision.beauty)
                if decision.add_voiceover and decision.voiceover_script and not plan.voiceover_text:
                    plan.voiceover_text = decision.voiceover_script
                report_progress(
                    "planner",
                    f"AI增强导演：卡点{'开' if extra['beat_sync'] else '关'}、"
                    f"美颜{'开' if extra['beauty'] else '关'}、"
                    f"配音{'开' if plan.voiceover_text else '关'}",
                    0.97,
                )
            except Exception as de:  # noqa: BLE001
                logger.warning("planner.director_skip", error=str(de))

        state.agent_status[AgentName.PLANNER.value] = AgentStatus.COMPLETED
        return {"edit_plan": plan, "agent_status": state.agent_status, **extra}
    except Exception as e:
        state.agent_status[AgentName.PLANNER.value] = AgentStatus.FAILED
        state.errors.append(f"规划失败: {e}")
        logger.error("agent.planner.failed", error=str(e))
        return {"agent_status": state.agent_status, "errors": state.errors}
