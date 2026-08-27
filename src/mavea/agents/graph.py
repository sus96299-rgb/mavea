"""LangGraph 状态图编排。

定义 4 个 Agent 节点和条件分支：
  analyzer → planner → executor → evaluator
                                  ├── passed=True  → END
                                  └── passed=False → planner（迭代，最多3轮）
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from mavea.agents import analyzer, evaluator, executor, planner
from mavea.models import AgentName, AgentStatus, GraphState

logger = structlog.get_logger(__name__)


def _should_continue(state: GraphState) -> str:
    """条件边：评估通过则结束，否则继续迭代。"""
    if state.errors and not state.execution_result:
        logger.error("graph.abort", errors=state.errors)
        return "end"

    if state.evaluation_result and state.evaluation_result.passed:
        logger.info("graph.passed", iteration=state.iteration, overall=state.evaluation_result.overall)
        return "end"

    if state.iteration >= state.max_iterations:
        logger.warning("graph.max_iterations", iteration=state.iteration)
        return "end"

    # 自适应停止：已有两轮评分时，若本轮相对上轮提升不足 0.3（含不升反降），
    # 说明返工收益已收敛，再迭代只浪费 token 和渲染成本
    history = getattr(state, "score_history", None) or []
    if len(history) >= 2 and (history[-1] - history[-2]) < 0.3:
        logger.info("graph.early_stop", iteration=state.iteration,
                    previous=history[-2], current=history[-1],
                    reason="score_gain_converged")
        return "end"

    logger.info("graph.iterate", iteration=state.iteration)
    return "iterate"


def _increment_iteration(state: GraphState) -> dict[str, Any]:
    """迭代前递增计数器。"""
    return {"iteration": state.iteration + 1}


def build_graph():
    """构建并编译 LangGraph 状态图。"""
    graph = StateGraph(GraphState)

    # 添加节点
    graph.add_node(AgentName.ANALYZER.value, analyzer.run)
    graph.add_node(AgentName.PLANNER.value, planner.run)
    graph.add_node(AgentName.EXECUTOR.value, executor.run)
    graph.add_node(AgentName.EVALUATOR.value, evaluator.run)
    graph.add_node("increment", _increment_iteration)

    # 设置入口
    graph.set_entry_point(AgentName.ANALYZER.value)

    # 线性边
    graph.add_edge(AgentName.ANALYZER.value, AgentName.PLANNER.value)
    graph.add_edge(AgentName.PLANNER.value, AgentName.EXECUTOR.value)
    graph.add_edge(AgentName.EXECUTOR.value, AgentName.EVALUATOR.value)

    # 条件边：评估后决定结束还是迭代
    graph.add_conditional_edges(
        AgentName.EVALUATOR.value,
        _should_continue,
        {
            "end": END,
            "iterate": "increment",
        },
    )

    # 迭代：increment → planner
    graph.add_edge("increment", AgentName.PLANNER.value)

    return graph.compile()


# 全局图实例
_graph = None


def get_graph():
    """获取编译后的图实例（单例）。"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_pipeline(
    material_paths: list[str],
    user_prompt: str,
    max_iterations: int = 3,
    custom_bgm_path: str | None = None,
    custom_bgm_query: str | None = None,
    beat_sync: bool = False,
    lyric_mode: str = "off",
    lrc_path: str | None = None,
    beauty: bool = False,
    ai_enhance: bool = True,
) -> GraphState:
    """运行完整剪辑流水线。

    Args:
        material_paths: 素材文件路径列表
        user_prompt: 用户自然语言需求
        max_iterations: 最大迭代轮数

    Returns:
        最终的 GraphState（包含所有 Agent 产出）
    """
    graph = get_graph()
    initial_state = GraphState(
        user_prompt=user_prompt,
        material_paths=material_paths,
        max_iterations=max_iterations,
        custom_bgm_path=custom_bgm_path,
        custom_bgm_query=custom_bgm_query,
        beat_sync=beat_sync,
        lyric_mode=lyric_mode if lyric_mode in ("off", "lrc", "whisper") else "off",
        lrc_path=lrc_path,
        beauty=bool(beauty),
        ai_enhance=bool(ai_enhance),
        agent_status={
            AgentName.ANALYZER.value: AgentStatus.PENDING,
            AgentName.PLANNER.value: AgentStatus.PENDING,
            AgentName.EXECUTOR.value: AgentStatus.PENDING,
            AgentName.EVALUATOR.value: AgentStatus.PENDING,
        },
    )

    logger.info("graph.pipeline.start", materials=len(material_paths), prompt=user_prompt[:50])
    final_state = await graph.ainvoke(initial_state)

    # LangGraph 可能返回 dict 或 GraphState，统一处理
    if isinstance(final_state, dict):
        exec_res = final_state.get("execution_result")
    else:
        exec_res = final_state.execution_result
    success = bool(exec_res and exec_res.success)
    logger.info("graph.pipeline.done", success=success)
    return final_state
