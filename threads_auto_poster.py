#!/usr/bin/env python3
# manage_posts.py  –  勇成專用 Threads 文章匯入工具（修正版）

import os
import csv
import pickle
import sqlite3
from typing import Optional

try:
    import chardet  # 建議：pip install chardet
except ImportError:
    chardet = None

from sentence_transformers import SentenceTransformer

# ===== 基本參數 ===== #
DB_PATH     = "threads_db.sqlite"
CSV_PATH    = "posts.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"
FALLBACK_ENCODINGS = ["utf-8-sig", "big5", "cp950", "utf-8"]
# ==================== #


def detect_encoding(path: str) -> Optional[str]:
    if chardet:
        with open(path, "rb") as fh:
            raw = fh.read(8192)
        enc = chardet.detect(raw).get("encoding")
        if enc:
            return enc.lower()
    return None


def initialize_db(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL UNIQUE,
                embedding  BLOB    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def embed_text(model: SentenceTransformer, text: str) -> bytes:
    return pickle.dumps(model.encode(text))


def open_csv(path: str):
    tried = []
    enc = detect_encoding(path)
    if enc:
        tried.append(enc)
        try:
            fh = open(path, newline="", encoding=enc)
            reader = csv.DictReader(fh)
            if reader.fieldnames and "content" in reader.fieldnames:
                print(f"[CSV] 偵測編碼：{enc}")
                return fh, reader
            fh.close()
        except UnicodeDecodeError:
            pass

    for enc in FALLBACK_ENCODINGS:
        if enc in tried:
            continue
        try:
            fh = open(path, newline="", encoding=enc)
            reader = csv.DictReader(fh)
            if reader.fieldnames and "content" in reader.fieldnames:
                print(f"[CSV] 使用備援編碼：{enc}")
                return fh, reader
            fh.close()
        except UnicodeDecodeError:
            continue

    print(f"[ERROR] 無法判斷 {path} 的正確編碼。")
    return None, None


def import_from_csv(
    db_path: str = DB_PATH,
    csv_path: str = CSV_PATH,
    model_name: str = EMBED_MODEL,
) -> None:
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 找不到 {csv_path}，請確認路徑。")
        return

    fh, reader = open_csv(csv_path)
    if reader is None:
        return

    model = SentenceTransformer(model_name)
    inserted = skipped = 0

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for i, row in enumerate(reader, start=1):
            content = (row.get("content") or "").strip()
            if not content:
                print(f"[WARN] 第 {i} 行空白，已跳過。")
                continue
            if cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,)).fetchone():
                skipped += 1
                continue
            try:
                cur.execute(
                    "INSERT INTO posts (content, embedding) VALUES (?, ?)",
                    (content, embed_text(model, content)),
                )
                inserted += 1
            except Exception as e:
                print(f"[ERROR] 第 {i} 行匯入失敗：{e}")

        conn.commit()  # ←←← 額外手動 commit，確保寫盤

    fh.close()
    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆。")


def show_last_five(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, content FROM posts ORDER BY id DESC LIMIT 5"
        ).fetchall()

    print("\n[CHECK] 最後 5 筆貼文：")
    for rid, txt in reversed(rows):
        preview = txt.replace("\n", " ")[:60]
        print(f"{rid:>5} │ {preview}")


def main(db_path: str = DB_PATH, csv_path: str = CSV_PATH) -> None:
    print("== 開始同步 posts.csv → SQLite ==")
    initialize_db(db_path)
    import_from_csv(db_path, csv_path)
    show_last_five(db_path)
    print("== 同步完成 ==")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    db = args[0] if len(args) >= 1 else DB_PATH
    csv_file = args[1] if len(args) >= 2 else CSV_PATH
    main(db, csv_file)
