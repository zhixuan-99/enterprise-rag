"""混合检索器：BM25 关键词检索 + 向量语义检索 + RRF 融合 + Rerank 精排。"""
import pickle

import jieba
from rank_bm25 import BM25Okapi

import config


def _tokenize(text):
    return list(jieba.cut(text))


class HybridRetriever:
    """混合检索器，兼容 LangChain retriever 接口（invoke 返回 List[Document]）。

    流程：query → BM25 top-k*5 + 向量 top-k*5 → RRF 融合 → Rerank 精排 → top-k。
    """

    def __init__(self, vectorstore, docs, reranker):
        self.vectorstore = vectorstore
        self.docs = docs
        self.reranker = reranker
        # 建 BM25 索引（jieba 分词）
        self._corpus = [_tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self._corpus)

    def _bm25_search(self, query, k):
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in ranked]

    def _vector_search(self, query, k):
        return self.vectorstore.similarity_search(query, k=k)

    def _rrf_fusion(self, results_a, results_b, k=60):
        """Reciprocal Rank Fusion：按排名倒数融合两个检索结果。"""
        scores, docs = {}, {}
        for rank, doc in enumerate(results_a):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(results_b):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [docs[k] for k in sorted_keys]

    def _rerank(self, query, docs, top_k):
        pairs = [[query, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in ranked]

    def invoke(self, query, top_k=None):
        top_k = top_k or config.TOP_K
        candidates = self._rrf_fusion(
            self._bm25_search(query, top_k * 5),
            self._vector_search(query, top_k * 5),
        )
        if not candidates:
            return []
        return self._rerank(query, candidates, top_k)


def build_hybrid_retriever():
    """构建混合检索器（加载 FAISS + chunks + reranker）。"""
    from langchain_community.vectorstores import FAISS
    from sentence_transformers import CrossEncoder

    from src.models import get_embeddings

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(config.INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    with open(config.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    reranker = CrossEncoder(config.RERANK_MODEL)
    return HybridRetriever(vectorstore, chunks, reranker)
