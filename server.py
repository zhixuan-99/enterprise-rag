"""FastAPI Web 服务：桥接前端页面与 RAG / Agent 能力。

启动：python server.py  （默认 http://127.0.0.1:8000）
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# 关键：在任何第三方库 import 之前，先加载 .env 并设置 HF_ENDPOINT（见 README 踩坑 #4）。
load_dotenv(Path(__file__).resolve().parent / ".env")
_hf_endpoint = os.getenv("HF_ENDPOINT")
if _hf_endpoint:
    os.environ["HF_ENDPOINT"] = _hf_endpoint

import json

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import storage

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="知识库 RAG + Agent 平台")


class Question(BaseModel):
    question: str
    history: list = []  # 多轮对话历史 [{role, content}]


@lru_cache(maxsize=1)
def get_chain():
    """懒加载 + 缓存单链 RAG，避免每次请求重复加载索引。"""
    from src.pipeline import build_rag_chain

    return build_rag_chain()


@lru_cache(maxsize=1)
def get_graph():
    """懒加载 + 缓存多 Agent 图。"""
    from src.agents.graph import build_agent_graph

    return build_agent_graph()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask")
def api_ask(body: Question):
    """单链 RAG 问答。"""
    answer = get_chain().invoke({"question": body.question, "history": body.history})
    storage.save_record(body.question, answer, "ask")
    return {"answer": answer}


@app.post("/api/agent")
def api_agent(body: Question):
    """多 Agent 问答。"""
    result = get_graph().invoke(
        {
            "question": body.question,
            "history": body.history,
            "route": "",
            "context": "",
            "answer": "",
            "eval_pass": False,
            "eval_feedback": "",
            "retry_count": 0,
        }
    )
    answer = result["answer"]
    storage.save_record(body.question, answer, "agent")
    return {"answer": answer}


@app.post("/api/ask/stream")
async def api_ask_stream(body: Question):
    """单链 RAG 流式问答（SSE）。"""

    async def gen():
        parts = []
        try:
            async for chunk in get_chain().astream(
                {"question": body.question, "history": body.history}
            ):
                parts.append(chunk)
                yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
            storage.save_record(body.question, "".join(parts), "ask")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: " + json.dumps({"done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/agent/stream")
async def api_agent_stream(body: Question):
    """多 Agent 流式问答（SSE），只流式输出分析 Agent 的答案 token。"""

    async def gen():
        parts = []
        try:
            async for chunk, metadata in get_graph().astream(
                {
                    "question": body.question,
                    "history": body.history,
                    "route": "",
                    "context": "",
                    "answer": "",
                    "eval_pass": False,
                    "eval_feedback": "",
                    "retry_count": 0,
                },
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                if node == "analyst":
                    content = chunk.content
                    if isinstance(content, str) and content:
                        parts.append(content)
                        yield f"data: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
            storage.save_record(body.question, "".join(parts), "agent")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        yield "data: " + json.dumps({"done": True}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/history")
def api_history(keyword: str = "", page: int = 1, page_size: int = 10):
    """搜索 + 分页查询历史记录（倒序）。"""
    return storage.get_records(keyword, page, page_size)


@app.delete("/api/history")
def api_clear_history():
    """清空历史记录。"""
    storage.clear_records()
    return {"ok": True}


# 静态资源（style.css / app.js）
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
