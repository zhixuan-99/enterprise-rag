"""工具封装：把检索能力封装成 Agent 可调用的工具。"""
from langchain_core.tools import tool

from src.retriever import get_retriever

_retriever = None


def _get_cached_retriever():
    """懒加载 + 缓存检索器，避免多次加载索引和 embedding 模型。"""
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中检索与 query 相关的内容。

    把要查的关键信息写成一个明确的中文查询，例如"专业版的价格和功能"。
    可以针对一个问题多次调用本工具，每次用不同角度查询。
    """
    retriever = _get_cached_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "未检索到相关内容。"
    return "\n\n".join(
        f"[片段 {i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )
