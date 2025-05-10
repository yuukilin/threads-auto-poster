#!/usr/bin/env python3
# manage_posts.py – CSV ➜ SQLite，支援「行內混編碼」，自動建表
# ===============================================================

import csv
import io
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

# ── 常數 ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CSV_PATH   = BASE_DIR / "posts.csv"
DB_PATH    = BASE_DIR / "threads_db.sqlite"
EMBED_MODEL = "all-MiniLM-L6-v2"
ENCODINGS  = ["utf-8-sig", "utf-8", "big5", "cp950"]
# ──────────────────────────────────────────────────────


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS posts (
                 id         INTEGER PRIMARY KEY AUTOINCREMENT,
                 content    TEXT    NOT NULL UNIQUE,
                 embedding  BLOB    NOT NULL,
                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP
               );"""
        )


def decode_line(line_bytes: bytes) -> str | None:
    """嘗試用多種編碼解 1 行；失敗回傳 None"""
    for enc in ENCODINGS:
        try:
            return line_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def load_csv_mixed(path: Path) -> List[str]:
    """回傳解完碼的整行清單（含 header）"""
    decoded_lines = []
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            txt = decode_line(raw)
            if txt is None:
                print(f"[WARN] 第 {lineno} 行解碼失敗，已跳過")
                continue
            decoded_lines.append(txt)
    if not decoded_lines:
        sys.exit("[ERROR] CSV 全部行都解碼失敗")
    return decoded_lines


def main():
    if not CSV_PATH.is_file():
        sys.exit(f"[ERROR] 找不到 {CSV_PATH}")

    init_db()  # ← 保證先建表
    model = SentenceTransformer(EMBED_MODEL)

    print("== 讀取 CSV（混編碼模式） ==")
    decoded_lines = load_csv_mixed(CSV_PATH)

    rdr = csv.DictReader(io.StringIO("".join(decoded_lines)))
    if "content" not in rdr.fieldnames:
        sys.exit("[ERROR] CSV 必須有 content 欄")

    inserted = skipped = 0
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for i, row in enumerate(rdr, 1):
            content = (row.get("content") or "").strip()
            if not content:
                print(f"[WARN] 第 {i} 行空白，跳過")
                continue
            if cur.execute(
                "SELECT 1 FROM posts WHERE content = ?", (content,)
            ).fetchone():
                skipped += 1
                continue

            emb_blob = pickle.dumps(model.encode(content))
            cur.execute(
                "INSERT INTO posts (content, embedding) VALUES (?, ?)",
                (content, emb_blob),
            )
            inserted += 1
        conn.commit()

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, content FROM posts ORDER BY id DESC LIMIT 5"
        ).fetchall()
    print("\n[CHECK] 最後 5 筆貼文：")
    for rid, txt in reversed(rows):
        print(f"{rid:>5} │ {txt.replace(chr(10), ' ')[:60]}")

    print("== 同步完成 ==")


if __name__ == "__main__":
    main()
