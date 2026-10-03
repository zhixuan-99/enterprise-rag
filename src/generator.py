"""生成环节：Prompt 模板 + 上下文格式化。"""
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    """你是企业知识库问答助手。请结合【对话历史】和【上下文】回答【当前问题】。

要求：
1. 完整、直接地回答当前问题的所有方面，不要遗漏。
2. 答案中每个事实都必须来自【上下文】，不要编造，也不要加入你自己的知识。
3. 可以用自己的话自然表达，但意思要与上下文一致，不要夸大或曲解。
4. 如果【上下文】不足以回答，只输出四个字：无法回答。
5. 只输出答案本身，不要解释、不要标注来源。

【对话历史】
{history}

【上下文】
{context}

【当前问题】
{question}

【答案】"""
)


def format_docs(docs):
    """把检索到的 chunk 拼成带编号的上下文。"""
    return "\n\n".join(
        f"[来源 {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )


def format_history(history):
    """把对话历史（[{role, content}]）格式化成文本。"""
    if not history:
        return "（无历史对话）"
    lines = []
    for m in history:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)
