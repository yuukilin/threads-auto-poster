#!/usr/bin/env python3
# manage_posts.py

import sqlite3
import csv
import pickle
import os
from sentence_transformers import SentenceTransformer

# ———— 參數設定 ———— #
DB_PATH  = "threads_db.sqlite"  # SQLite 資料庫檔案
CSV_PATH = "posts.csv"          # 貼文來源 CSV
# 模型可自行換成你最喜歡的 SBERT 版本
EMBED_MODEL = "all-MiniLM-L6-v2"
# ———————————————— #

def initialize_db(db_path: str = DB_PATH):
    """
    建立資料表（若不存在）
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
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
    """
    取得文字向量並 pickle 成 bytes
    """
    vec = model.encode(text)
    return pickle.dumps(vec)

def import_from_csv(db_path: str = DB_PATH, csv_path: str = CSV_PATH):
    """
    讀取 CSV，將不存在於 posts 表的貼文匯入，已存在者跳過
    """
    if not os.path.isfile(csv_path):
        print(f"[ERROR] 找不到 {csv_path}，請確認檔案路徑正確。")
        return

    # 載入 SBERT 模型
    model = SentenceTransformer(EMBED_MODEL)

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    inserted = 0
    skipped  = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "content" not in reader.fieldnames:
            print("[ERROR] CSV 欄位中找不到 'content'，請確認第一欄標題為 content。")
            return

        for i, row in enumerate(reader, start=1):
            content = row["content"].strip()
            if not content:
                print(f"[WARN] 第 {i} 行為空，跳過。")
                continue

            # 檢查是否已存在
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

    print(f"[RESULT] 已插入 {inserted} 筆，跳過 {skipped} 筆。")

def main():
    print("== 開始同步 posts.csv 到 SQLite ==")
    initialize_db()
    import_from_csv()
    print("== 同步完成 ==")

if __name__ == "__main__":
    main()
