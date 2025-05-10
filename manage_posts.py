#!/usr/bin/env python3
# manage_posts.py – 勇成專用 Threads 文章匯入工具（CSV ➜ SQLite）
# ===============================================================

import csv
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import chardet
except ImportError:
    chardet = None

from sentence_transformers import SentenceTransformer

# ─── 檔案路徑 ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "posts.csv"
DB_PATH  = BASE_DIR / "threads_db.sqlite"
EMBED_MODEL = "all-MiniLM-L6-v2"

FALLBACK_ENC = ["utf-8-sig", "utf-8", "big5", "cp950"]
# ──────────────────────────────────────────────────────────


# ---------- 工具 ---------- #
def detect_encoding(path: Path) -> Optional[str]:
    if not chardet:
        return None
    with path.open("rb") as fh:
        raw = fh.read(8192)
    enc = chardet.detect(raw).get("encoding")
    return enc.lower() if enc else None


def open_csv(path: Path):
    tried = []
    enc = detect_encoding(path)
    if enc:
        tried.append(enc)
        try:
            fh = path.open(newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 偵測編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            pass
    for enc in FALLBACK_ENC:
        if enc in tried:
            continue
        try:
            fh = path.open(newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 使用備援編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            continue
    print(f"[ERROR] 無法判斷 {path} 編碼")
    return None, None


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


# ---------- 主流程 ---------- #
def main():
    if not CSV_PATH.is_file():
        sys.exit(f"[ERROR] 找不到 {CSV_PATH}")

    fh, rdr = open_csv(CSV_PATH)
    if rdr is None:
        sys.exit(1)

    print("== 讀取 CSV，匯入 DB ==")
    model = SentenceTransformer(EMBED_MODEL)

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

            emb = pickle.dumps(model.encode(content))
            cur.execute(
                "INSERT INTO posts (content, embedding) VALUES (?, ?)",
                (content, emb),
            )
            inserted += 1
        conn.commit()

    fh.close()
    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")

    # 顯示最後 5 筆
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
