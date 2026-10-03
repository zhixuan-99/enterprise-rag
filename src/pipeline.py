"""RAG 管道：retriever + prompt + llm 用 LCEL 串起来。"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

from src.generator import RAG_PROMPT, format_docs, format_history
from src.models import get_llm
from src.retriever import get_retriever


def build_rag_chain(retriever=None):
    """组装 RAG 链。

    invoke 时传入 {"question": str, "history": [{role, content}]}。
    retriever 可选传入，供评测脚本复用同一个检索器（避免重复加载索引）。
    """
    if retriever is None:
        retriever = get_retriever()
    llm = get_llm()

    rag_chain = (
        RunnableParallel(
            context=RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            history=RunnableLambda(lambda x: x.get("history", [])) | format_history,
            question=RunnableLambda(lambda x: x["question"]),
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain
