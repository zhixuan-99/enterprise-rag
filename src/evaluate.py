"""评测：逐条跑 RAG 管道，用 RAGAS 打 4 个指标，输出基线分数。

指标说明：
- faithfulness       忠实度：答案是否忠于检索到的上下文（防幻觉）
- answer_relevancy   答案相关性：答案是否切题
- context_precision  上下文精确度：检索到的上下文排序是否把相关的排在前面
- context_recall     上下文召回率：标准答案所需信息是否被检索到
"""
import json
import sys
import types

# 绕过 ragas 0.4.3 的 import bug：
# ragas/llms/base.py 无条件导入 langchain_community.chat_models.vertexai，
# 但该模块在 langchain-community 0.4+ 已被移除（迁移到 langchain-google-vertexai），
# 导致 import ragas 直接崩溃。这里先注册一个空 stub 占位（我们显式传了 llm，用不到它）。
import langchain_community.chat_models

_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = None
langchain_community.chat_models.vertexai = _vertexai_stub
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

import config
from src.models import get_embeddings, get_llm
from src.pipeline import build_rag_chain
from src.retriever import get_hybrid_retriever


def run_eval():
    # 1. 加载评测集
    with open(config.EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    # 2. 逐条跑：拿到 answer + contexts（复用同一个 retriever，避免重复加载索引）
    retriever = get_hybrid_retriever()
    chain = build_rag_chain(retriever=retriever)

    questions, answers, contexts_list, ground_truths, references = [], [], [], [], []

    for i, item in enumerate(eval_set, start=1):
        q = item["question"]
        answer = chain.invoke({"question": q, "history": []})
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])
        references.append(item.get("reference", ""))

        print(f"[eval] {i}/{len(eval_set)} 完成: {q[:25]}...")

    # 3. 构造 RAGAS 数据集
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
            "reference": references,
        }
    )

    # 4. 打分
    llm = get_llm()
    embeddings = get_embeddings()
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n========== RAGAS 基线分数 ==========")
    print(result)
    print("=====================================")
    return result


if __name__ == "__main__":
    run_eval()
