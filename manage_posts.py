#!/usr/bin/env python3
# manage_posts.py

import sqlite3
import csv
import pickle
import os
from sentence_transformers import SentenceTransformer

# ———— 參數設定 ———— #
DB_PATH     = "threads_db.sqlite"  # SQLite 資料庫檔案
CSV_PATH    = "posts.csv"          # 貼文來源 CSV
EMBED_MODEL = "all-MiniLM-L6-v2"   # SBERT 模型
# 支援的編碼順序
ENCODINGS   = ["utf-8", "big5", "cp950"]
# ———————————————— #

def initialize_db(db_path: str = DB_PATH):
    """建立資料表（若不存在）"""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def embed_text(model, text: str) -> bytes:
    """取得文字向量並 pickle 成 bytes"""
    vec = model.encode(text)
    return pickle.dumps(vec)

def import_from_csv(db_path: str = DB_PATH, csv_path: str = CSV_PATH):
    """從 CSV 讀爆文，支援多種編碼，自動跳過已存在貼文"""
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 找不到 {csv_path}，確認放到專案根目錄。")
        return

    # 嘗試不同編碼讀檔
    reader = None
    f = None
    for enc in ENCODINGS:
        try:
            f = open(csv_path, newline="", encoding=enc, errors="replace")
            tmp = csv.DictReader(f)
            if tmp.fieldnames and "content" in tmp.fieldnames:
                reader = tmp
                print(f"[CSV] 讀取成功，使用編碼：{enc}")
                break
        except Exception:
            pass
    if reader is None:
        print(f"[ERROR] 無法用 {ENCODINGS} 讀取 CSV，檢查檔案格式。")
        return

    # 載入向量模型
    model = SentenceTransformer(EMBED_MODEL)
    conn  = sqlite3.connect(db_path)
    cur   = conn.cursor()
    inserted = skipped = 0

    for i, row in enumerate(reader, start=1):
        content = (row.get("content") or "").strip()
        if not content:
            print(f"[WARN] 第 {i} 行空白，跳過。")
            continue
        cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,))
        if cur.fetchone():
            skipped += 1
        else:
            try:
                emb = embed_text(model, content)
                cur.execute(
                    "INSERT INTO posts (content, embedding) VALUES (?, ?)",
                    (content, emb)
                )
                inserted += 1
            except Exception as e:
                print(f"[ERROR] 第 {i} 行匯入失敗：{e}")

    conn.commit()
    conn.close()
    if f: f.close()
    print(f"[RESULT] 已插入 {inserted} 筆，跳過 {skipped} 筆。")

def main():
    print("== 開始同步 posts.csv 到 SQLite ==")
    initialize_db()
    import_from_csv()
    print("== 同步完成 ==")

if __name__ == "__main__":
    main()
