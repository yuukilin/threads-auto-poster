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
DB_PATH     = "threads_db.sqlite"      # SQLite 資料庫
CSV_PATH    = "posts.csv"              # 貼文 CSV
EMBED_MODEL = "all-MiniLM-L6-v2"       # SBERT
# 手動備援編碼清單（依常見度排序）
FALLBACK_ENCODINGS = ["utf-8-sig", "big5", "cp950", "utf-8"]
# ==================== #


def detect_encoding(path: str) -> Optional[str]:
    """自動偵測檔案編碼，失敗回傳 None"""
    if chardet:
        with open(path, "rb") as fh:
            raw = fh.read(8192)  # 取前 8 KB 夠用
        guess = chardet.detect(raw)
        enc = guess.get("encoding")
        if enc:
            return enc.lower()
    return None


def initialize_db(db_path: str = DB_PATH) -> None:
    """建立資料表（若不存在）"""
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
    """將文字轉成向量並 pickle"""
    vec = model.encode(text)
    return pickle.dumps(vec)


def open_csv(path: str):
    """依偵測／備援編碼嘗試開啟 CSV，成功回傳 (file_handle, dict_reader)"""
    tried = []

    # 1) chardet
    enc = detect_encoding(path)
    if enc:
        tried.append(enc)
        try:
            fh = open(path, newline="", encoding=enc)  # errors=strict
            reader = csv.DictReader(fh)
            if reader.fieldnames and "content" in reader.fieldnames:
                print(f"[CSV] 偵測編碼：{enc}")
                return fh, reader
            fh.close()
        except UnicodeDecodeError:
            pass  # 失敗就往下試

    # 2) 備援列表
    for enc in FALLBACK_ENCODINGS:
        if enc in tried:
            continue
        try:
            fh = open(path, newline="", encoding=enc)  # errors=strict
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
    """將 CSV 內容匯入 SQLite（略過重覆）"""
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
            cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,))
            if cur.fetchone():
                skipped += 1
                continue
            try:
                emb = embed_text(model, content)
                cur.execute(
                    "INSERT INTO posts (content, embedding) VALUES (?, ?)",
                    (content, emb),
                )
                inserted += 1
            except Exception as e:
                print(f"[ERROR] 第 {i} 行匯入失敗：{e}")

    fh.close()
    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆。")


def main(db_path: str = DB_PATH, csv_path: str = CSV_PATH) -> None:
    print("== 開始同步 posts.csv → SQLite ==")
    initialize_db(db_path)
    import_from_csv(db_path, csv_path)
    print("== 同步完成 ==")


if __name__ == "__main__":
    # 若需要自訂路徑，可：
    # python manage_posts.py custom_db.sqlite custom_posts.csv
    import sys

    args = sys.argv[1:]
    db = args[0] if len(args) >= 1 else DB_PATH
    csv_file = args[1] if len(args) >= 2 else CSV_PATH
    main(db, csv_file)
