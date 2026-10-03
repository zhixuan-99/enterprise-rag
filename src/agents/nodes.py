"""三个专家 Agent 节点 + 调度器。"""
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agents.tools import search_knowledge_base
from src.models import get_llm


class AgentState(TypedDict):
    """Agent 图的状态（跨节点流转）。"""
    question: str        # 用户问题
    history: list        # 对话历史 [{role, content}]
    route: str           # 调度决策：retrieve / direct
    context: str         # 检索到的上下文
    answer: str          # 生成的答案
    eval_pass: bool      # 评测是否通过
    eval_feedback: str   # 评测反馈
    retry_count: int     # 重做次数


def _format_history(history):
    """把对话历史（[{role, content}]）格式化成文本。"""
    if not history:
        return "（无）"
    lines = []
    for m in history:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def router(state: AgentState) -> dict:
    """调度器：判断问题是否需要检索知识库。"""
    llm = get_llm()
    q = state["question"]
    msg = llm.invoke(
        [
            SystemMessage(
                "判断用户问题是否涉及企业知识库内容（产品、价格、功能、制度、规范、售后等）。"
                "涉及则只回复 retrieve，不涉及（闲聊/常识）则只回复 direct。"
            ),
            HumanMessage(q),
        ]
    )
    decision = msg.content.strip().lower()
    route = "direct" if "direct" in decision else "retrieve"
    return {"route": route}


def retriever_agent(state: AgentState) -> dict:
    """检索 Agent：结合对话历史自主规划检索查询，可多次检索。"""
    llm = get_llm().bind_tools([search_knowledge_base])
    question = state["question"]
    history = _format_history(state.get("history", []))

    system = (
        "你是检索专家。把用户问题拆解成检索查询，调用 search_knowledge_base 获取信息。"
        "复杂问题需要分多个关键词分别检索（例如对比两个版本，就分别检索两个版本的信息）。"
        "信息足够后停止调用工具。"
    )
    if history != "（无）":
        system += (
            f"\n\n对话历史：\n{history}\n"
            "（若当前问题有指代，如'那专业版呢'，请结合历史确定指代对象后再检索。）"
        )

    messages = [
        SystemMessage(system),
        HumanMessage(question),
    ]

    context_parts = []
    for _ in range(5):  # 最多 5 轮工具调用，防止死循环
        response = llm.invoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            if tc["name"] == "search_knowledge_base":
                result = search_knowledge_base.invoke(tc["args"])
                context_parts.append(result)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    context = "\n\n".join(context_parts) if context_parts else ""
    return {"context": context}


def analyst_agent(state: AgentState) -> dict:
    """分析 Agent：结合对话历史与上下文生成答案 / 对比 / 建议。"""
    llm = get_llm()
    question = state["question"]
    context = state.get("context", "")
    feedback = state.get("eval_feedback", "")
    history = _format_history(state.get("history", []))

    if context:
        base = (
            "你是企业知识库分析助手。基于下方【上下文】回答用户问题。\n"
            "要求：完整回答问题的所有方面（对比类要逐项对比，建议类要给出理由）；"
            "每个事实都必须来自【上下文】，不要编造；上下文不足时只回答：无法回答。"
        )
        if feedback:
            base += f"\n\n【上一轮评测反馈，请针对性改进】\n{feedback}"
        prompt = (
            f"{base}\n\n【对话历史】\n{history}\n\n【上下文】\n{context}"
            f"\n\n【当前问题】\n{question}\n\n【答案】"
        )
    else:
        prompt = (
            "你是企业知识库分析助手。当前没有检索到相关知识库内容，请基于对话历史和常识简要回答。\n\n"
            f"【对话历史】\n{history}\n\n【当前问题】\n{question}\n\n【答案】"
        )

    answer = llm.invoke(prompt).content
    return {"answer": answer}


def evaluator_agent(state: AgentState) -> dict:
    """评测 Agent：自评答案质量，决定是否通过。"""
    llm = get_llm()
    question = state["question"]
    context = state.get("context", "")
    answer = state.get("answer", "")
    retry_count = state.get("retry_count", 0)

    prompt = (
        "你是严格的评测员。判断下面答案的质量，只从两个维度看：\n"
        "1. 忠实度：答案事实是否都能在上下文中找到依据，有没有编造。\n"
        "2. 完整性：是否完整回答了问题的所有方面。\n\n"
        f"【问题】{question}\n\n【上下文】{context}\n\n【待评测答案】{answer}\n\n"
        "两个维度都合格只回复：PASS\n"
        "任一不合格回复：FAIL + 一句具体改进建议"
    )

    result = llm.invoke(prompt).content.strip()
    passed = result.upper().startswith("PASS")
    feedback = "" if passed else result.replace("FAIL", "").replace("fail", "").strip("：: ")

    return {
        "eval_pass": passed,
        "eval_feedback": feedback,
        "retry_count": retry_count + 1,
    }
