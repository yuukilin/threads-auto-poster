#!/usr/bin/env python3
# manage_posts.py – CSV ➜ SQLite，自動 git push（最簡版）
# =======================================================

import csv, io, os, pickle, sqlite3, subprocess, sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "posts.csv"
DB_PATH  = BASE_DIR / "threads_db.sqlite"
MODEL    = "all-MiniLM-L6-v2"
ENCODING_TRY = ["utf-8-sig", "utf-8", "big5", "cp950"]

# ---------- 基礎工具 ---------- #
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content   TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );""")

def try_decode(b: bytes) -> Optional[str]:
    for enc in ENCODING_TRY:
        try: return b.decode(enc)
        except UnicodeDecodeError: pass
    return None

def read_csv_lines(path: Path) -> List[str]:
    lines = []
    with path.open("rb") as fh:
        for i, raw in enumerate(fh, 1):
            txt = try_decode(raw)
            if txt is None:
                print(f"[WARN] 第 {i} 行解碼失敗，跳過")
                continue
            lines.append(txt)
    if not lines:
        sys.exit("[ERROR] 整份 CSV 解碼失敗")
    return lines

# ---------- 自動 git push ---------- #
def git_push_if_new(inserted: int):
    if inserted == 0:
        return
    pat = os.getenv("GH_PAT")
    if not pat:
        print("[INFO] GH_PAT 未設，DB 只存在 Action 環境，不 push")
        return
    subprocess.run(["git", "config", "--global", "user.email", "bot@threads"])
    subprocess.run(["git", "config", "--global", "user.name",  "threads-bot"])
    # 重新寫 remote URL，嵌入 PAT
    remote = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True
    ).strip()
    if remote.startswith("https://"):
        remote = remote.replace("https://", f"https://{pat}@")
    subprocess.run(["git", "add", str(DB_PATH)], check=True)
    subprocess.run(["git", "commit", "-m", f"chore(db): +{inserted} posts [skip ci]"], check=True)
    subprocess.run(["git", "push", remote, "HEAD:main"], check=True)
    print("[GIT] DB 已 push 回 repo")

# ---------- 主程式 ---------- #
def main():
    if not CSV_PATH.is_file():
        sys.exit("[ERROR] 找不到 posts.csv")

    init_db()
    model = SentenceTransformer(MODEL)

    rdr = csv.DictReader(io.StringIO("".join(read_csv_lines(CSV_PATH))))
    if "content" not in rdr.fieldnames:
        sys.exit("[ERROR] CSV 缺少 content 欄")

    inserted = skipped = 0
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for row in rdr:
            content = (row.get("content") or "").strip()
            if not content:
                continue  # 空白行略過
            if cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,)).fetchone():
                skipped += 1
                continue
            emb = pickle.dumps(model.encode(content))
            cur.execute("INSERT INTO posts (content, embedding) VALUES (?, ?)", (content, emb))
            inserted += 1
        conn.commit()

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")
    git_push_if_new(inserted)

if __name__ == "__main__":
    main()
