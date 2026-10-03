"""模型工厂：统一创建 LLM 和 Embedding，后续替换模型只改这里。"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import config


def get_llm():
    """DeepSeek LLM（OpenAI 兼容接口）。"""
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0,
    )


def get_embeddings():
    """本地 BGE 中文 embedding。

    BGE 系列建议 normalize_embeddings=True，检索效果更好。
    后续可换 BAAI/bge-m3（更强），只改 config.EMBEDDING_MODEL。
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
