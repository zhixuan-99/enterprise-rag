"""历史会话存储（SQLite，零额外依赖）。"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_record(question: str, answer: str, mode: str):
    """保存一条问答记录。"""
    init_db()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO messages (question, answer, mode, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, mode, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    finally:
        conn.close()


def get_records(keyword: str = "", page: int = 1, page_size: int = 10):
    """搜索 + 分页查询历史记录（倒序）。返回 records/total/page/has_more。"""
    init_db()
    conn = _connect()
    try:
        where = ""
        params = []
        if keyword:
            where = "WHERE question LIKE ? OR answer LIKE ?"
            like = f"%{keyword}%"
            params = [like, like]

        total = conn.execute(
            f"SELECT COUNT(*) FROM messages {where}", params
        ).fetchone()[0]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT id, question, answer, mode, created_at FROM messages {where} "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()
    finally:
        conn.close()

    return {
        "records": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": page * page_size < total,
    }


def clear_records():
    """清空全部历史。"""
    init_db()
    conn = _connect()
    try:
        conn.execute("DELETE FROM messages")
        conn.commit()
    finally:
        conn.close()
