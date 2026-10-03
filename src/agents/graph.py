"""LangGraph 图：把 3 个 Agent 编排成「检索 → 分析 → 评测 → 打回重做」流程。"""
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    AgentState,
    analyst_agent,
    evaluator_agent,
    retriever_agent,
    router,
)

MAX_RETRY = 3


def route_decision(state: AgentState) -> str:
    """调度器之后的边：根据决策路由到检索或直接分析。"""
    return state.get("route", "retrieve")


def should_retry(state: AgentState) -> str:
    """评测之后的边：通过就结束，不通过且未超上限就打回分析 Agent。"""
    if state.get("eval_pass"):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "end"
    return "retry"


def build_agent_graph():
    """构建并编译 Agent 图。"""
    graph = StateGraph(AgentState)

    graph.add_node("router", router)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("evaluator", evaluator_agent)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retriever", "direct": "analyst"},
    )
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {"end": END, "retry": "analyst"},
    )

    return graph.compile()
