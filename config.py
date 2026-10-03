"""集中配置：所有可调参数都在这里，方便后续做 A/B 对比。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env
load_dotenv(BASE_DIR / ".env")

# 设置 HuggingFace 镜像（国内加速下载模型，需在加载 embedding 前生效）
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

# ========== 模型配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# 重排序模型（混合检索的精排阶段）
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# ========== 切分配置（后续 A/B 对比点：调 chunk_size 看分数变化）==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# ========== 检索配置 ==========
TOP_K = 4

# ========== 路径 ==========
DOCS_DIR = BASE_DIR / "data" / "docs"
EVAL_SET_PATH = BASE_DIR / "data" / "eval_set.json"

# faiss 在 Windows 下无法读写中文路径，索引改存到用户缓存目录（纯英文路径）。
# 项目移到英文路径后，可改回 BASE_DIR / "data" / "faiss_index"。
INDEX_DIR = Path.home() / ".cache" / "rag_index"

# 切分后的 chunk 列表（供 BM25 混合检索使用）
CHUNKS_PATH = Path.home() / ".cache" / "rag_chunks.pkl"
