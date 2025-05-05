#!/usr/bin/env python3
# manage_posts.py  –  勇成專用 Threads 文章匯入工具

import os, csv, pickle, sqlite3
from typing import Optional
try:
    import chardet
except ImportError:
    chardet = None
from sentence_transformers import SentenceTransformer

# ===== 參數 ===== #
BASE_DIR = os.path.abspath(os.path.dirname(__file__))       # ← 固定路徑
DB_PATH  = os.path.join(BASE_DIR, "threads_db.sqlite")
CSV_PATH = os.path.join(BASE_DIR, "posts.csv")
EMBED_MODEL = "all-MiniLM-L6-v2"
FALLBACK_ENCODINGS = ["utf-8-sig", "big5", "cp950", "utf-8"]
# ================= #

def detect_encoding(path: str) -> Optional[str]:
    if chardet:
        with open(path, "rb") as fh:
            raw = fh.read(8192)
        enc = chardet.detect(raw).get("encoding")
        if enc:
            return enc.lower()

def initialize_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

def embed_text(model, text):  # -> bytes
    return pickle.dumps(model.encode(text))

def open_csv(path: str):
    tried = []
    enc = detect_encoding(path)
    if enc:
        tried.append(enc)
        try:
            fh = open(path, newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 偵測編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            pass
    for enc in FALLBACK_ENCODINGS:
        if enc in tried: continue
        try:
            fh = open(path, newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 使用備援編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            continue
    print(f"[ERROR] 無法判斷 {path} 的正確編碼。")
    return None, None

def import_from_csv():
    if not os.path.isfile(CSV_PATH):
        print(f"[ERROR] 找不到 {CSV_PATH}")
        return
    fh, rdr = open_csv(CSV_PATH)
    if rdr is None: return

    model = SentenceTransformer(EMBED_MODEL)
    inserted = skipped = 0

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for i, row in enumerate(rdr, 1):
            content = (row.get("content") or "").strip()
            if not content:
                print(f"[WARN] 第 {i} 行空白，跳過。"); continue
            if cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,)).fetchone():
                skipped += 1; continue
            try:
                cur.execute("INSERT INTO posts (content, embedding) VALUES (?, ?)",
                            (content, embed_text(model, content)))
                inserted += 1
            except Exception as e:
                print(f"[ERROR] 第 {i} 行匯入失敗：{e}")
        conn.commit()                               # ← 手動保險 commit

    fh.close()
    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆。")

def show_last_five():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, content FROM posts ORDER BY id DESC LIMIT 5").fetchall()
    print("\n[CHECK] 最後 5 筆貼文：")
    for rid, txt in reversed(rows):
        print(f"{rid:>5} │ {txt.replace(chr(10), ' ')[:60]}")

def main():
    print("== 開始同步 posts.csv → SQLite ==")
    initialize_db()
    import_from_csv()
    show_last_five()
    print("== 同步完成 ==")

if __name__ == "__main__":
    main()
