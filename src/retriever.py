"""检索器：加载 FAISS 索引，做向量相似度检索。"""
from langchain_community.vectorstores import FAISS

import config
from src.models import get_embeddings


def get_retriever():
    """返回 top_k 向量检索器（骨架期纯向量，后续加 BM25 混合检索）。"""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
