"""检索器：加载 FAISS 索引，提供纯向量检索器和混合检索器。"""
from functools import lru_cache

from langchain_community.vectorstores import FAISS

import config
from src.models import get_embeddings


def get_retriever():
    """返回 top_k 纯向量检索器（保留用于对比）。"""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    """混合检索器（BM25 + 向量 + RRF + Rerank），懒加载 + 缓存。"""
    from src.hybrid_retriever import build_hybrid_retriever

    return build_hybrid_retriever()
