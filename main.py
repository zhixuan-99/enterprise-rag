"""命令行入口。

用法：
    python main.py ingest          # 文档入库
    python main.py ask "问题"       # 单条问答（单链 RAG）
    python main.py agent "问题"     # 多 Agent 问答（检索→分析→评测）
    python main.py eval            # RAGAS 评测
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 关键：在任何第三方库 import 之前，先加载 .env 并设置 HF_ENDPOINT。
# huggingface_hub 在 import 时会读取一次 HF_ENDPOINT 作为镜像地址，
# 若等 config.py 再设置就晚了（huggingface_hub 已默认用 huggingface.co，国内连不上）。
load_dotenv(Path(__file__).resolve().parent / ".env")
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

import argparse


def cmd_ingest():
    from src.ingest import build_index
    import config

    build_index()
    print(f"[OK] 文档已入库，向量索引保存在 {config.INDEX_DIR}")


def cmd_ask(question):
    from src.pipeline import build_rag_chain

    chain = build_rag_chain()
    answer = chain.invoke({"question": question, "history": []})
    print(answer)


def cmd_eval():
    from src.evaluate import run_eval

    run_eval()


def cmd_agent(question):
    from src.agents.graph import build_agent_graph

    app = build_agent_graph()
    result = app.invoke(
        {
            "question": question,
            "history": [],
            "route": "",
            "context": "",
            "answer": "",
            "eval_pass": False,
            "eval_feedback": "",
            "retry_count": 0,
        }
    )
    print(result["answer"])


def main():
    parser = argparse.ArgumentParser(description="企业级知识库 RAG 平台")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ingest", help="文档入库（切分 + 向量化 + 存 FAISS）")

    ask = sub.add_parser("ask", help="单条问答")
    ask.add_argument("question", help="要问的问题")

    sub.add_parser("eval", help="RAGAS 评测，输出基线分数")

    agent = sub.add_parser("agent", help="多 Agent 问答（检索 → 分析 → 评测）")
    agent.add_argument("question", help="要问的问题")

    args = parser.parse_args()

    if args.cmd == "ingest":
        cmd_ingest()
    elif args.cmd == "ask":
        cmd_ask(args.question)
    elif args.cmd == "agent":
        cmd_agent(args.question)
    elif args.cmd == "eval":
        cmd_eval()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
