#!/usr/bin/env python3
# manage_posts.py  –  勇成專用 Threads 貼文匯入（免外掛版）

import os
import csv
import pickle
import sqlite3
import sys
from typing import Optional

from sentence_transformers import SentenceTransformer

# ===== 參數 ===== #
DB_PATH     = "threads_db.sqlite"
CSV_PATH    = "posts.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"

# 依常見度排序的備援編碼（可以再加）
FALLBACK_ENCODINGS = [
    "utf-8-sig", "utf-8",
    "big5", "cp950",
    "utf-16-le", "utf-16-be", "utf-16",
    "windows-1252", "latin1"
]
# ================= #


# ---------- 工具函式 ---------- #
def detect_bom(raw: bytes) -> Optional[str]:
    """簡易 BOM 判斷，回傳對應編碼或 None"""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    return None


def decode_csv_bytes(raw: bytes, enc_list) -> Optional[str]:
    """依編碼清單嘗試解碼 raw，成功回傳字串"""
    for enc in enc_list:
        try:
            return raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue
    # 全部失敗就保底亂碼忽略
    try:
        return raw.decode("latin1", errors="ignore")
    except Exception:
        return None
# ----------------------------- #


def initialize_db(db_path: str = DB_PATH) -> None:
    """建表（若不存在）"""
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
    """文字轉向量 > pickle"""
    return pickle.dumps(model.encode(text))


def load_csv_as_reader(csv_path: str):
    """讀檔 → 自動解碼 → 回傳 csv.DictReader"""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 找不到 {csv_path}")
        return None

    with open(csv_path, "rb") as fh:
        raw = fh.read()

    enc_from_bom = detect_bom(raw[:4])
    enc_list = [enc_from_bom] if enc_from_bom else []
    enc_list += [e for e in FALLBACK_ENCODINGS if e not in enc_list]

    text = decode_csv_bytes(raw, enc_list)
    if text is None:
        print("[ERROR] 全部編碼嘗試失敗，檔案可能毀損？")
        return None

    # 把字串包成 file-like object 給 DictReader
    import io

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "content" not in reader.fieldnames:
        print("[ERROR] CSV 缺少 'content' 欄位，請確認檔案格式。")
        return None
    used_enc = enc_list[0] if enc_from_bom else "fallback"
    print(f"[CSV] 使用編碼：{used_enc}")
    return reader


def import_from_csv(db_path: str, csv_path: str, model_name: str) -> None:
    """讀 CSV → 匯入 SQLite（略過重覆）"""
    reader = load_csv_as_reader(csv_path)
    if reader is None:
        return

    model = SentenceTransformer(model_name)
    inserted = skipped = 0

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for i, row in enumerate(reader, 1):
            content = (row.get("content") or "").strip()
            if not content:
                print(f"[WARN] 第 {i} 行空白，跳過。")
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
                print(f"[ERROR] 第 {i} 行失敗：{e}")

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆。")


def main(db_path: str = DB_PATH, csv_path: str = CSV_PATH) -> None:
    print("== 開始同步 posts.csv → SQLite ==")
    initialize_db(db_path)
    import_from_csv(db_path, csv_path, EMBED_MODEL)
    print("== 同步完成 ==")


if __name__ == "__main__":
    # 自訂路徑：python manage_posts.py my_db.sqlite my_posts.csv
    args = sys.argv[1:]
    db = args[0] if len(args) >= 1 else DB_PATH
    csv_file = args[1] if len(args) >= 2 else CSV_PATH
    main(db, csv_file)
