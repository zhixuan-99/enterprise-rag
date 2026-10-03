"""文档入库：加载 docs 目录的 .md 文件 -> 切分 -> embedding -> 存 FAISS。"""
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.models import get_embeddings


def load_documents():
    """加载 data/docs 下所有 .md 文件。"""
    docs = []
    for path in sorted(config.DOCS_DIR.glob("*.md")):
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    print(f"[ingest] 共加载 {len(docs)} 个文档")
    return docs


def split_documents(docs):
    """按中文标点友好切分。separators 里加入中文标点，切分更贴合语义。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] 切分后共 {len(chunks)} 个 chunk")
    return chunks


def build_index():
    """完整入库流程，结果存到 config.INDEX_DIR。"""
    docs = load_documents()
    chunks = split_documents(docs)
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.INDEX_DIR))
    print(f"[ingest] 索引已保存到 {config.INDEX_DIR}")
