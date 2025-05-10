#!/usr/bin/env python3
# manage_posts.py – CSV ➜ SQLite，自動 git commit & push（v2.0）
# ===============================================================

import csv, io, os, pickle, sqlite3, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

BASE_DIR   = Path(__file__).resolve().parent
CSV_PATH   = BASE_DIR / "posts.csv"
DB_PATH    = BASE_DIR / "threads_db.sqlite"
EMBED_MODEL = "all-MiniLM-L6-v2"
ENCODINGS  = ["utf-8-sig", "utf-8", "big5", "cp950"]

# ---------- DB ---------- #
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS posts (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 content   TEXT NOT NULL UNIQUE,
                 embedding BLOB NOT NULL,
                 created_at DATETIME DEFAULT CURRENT_TIMESTAMP
               );"""
        )

# ---------- CSV decode ---------- #
def try_decode(raw: bytes) -> str | None:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None

def load_csv(path: Path) -> List[str]:
    lines = []
    with path.open("rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            txt = try_decode(raw)
            if txt is None:
                print(f"[WARN] 第 {lineno} 行編碼炸裂，跳過")
                continue
            lines.append(txt)
    if not lines:
        sys.exit("[ERROR] 整份 CSV 都無法解碼")
    return lines

# ---------- Git auto commit ---------- #
def auto_commit(inserted: int):
    if inserted == 0:
        return
    pat = os.getenv("GH_PAT")
    if not pat:
        print("[INFO] GH_PAT 未設定，僅本地更新 DB，不 push")
        return

    repo_url = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True
    ).strip()
    # 改用 PAT 認證
    if repo_url.startswith("https://"):
        repo_url = repo_url.replace(
            "https://", f"https://{pat}@"
        )

    subprocess.run(["git", "add", str(DB_PATH)], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore(db): update posts ({inserted} new) [skip ci]"],
        check=True,
    )
    subprocess.run(["git", "push", repo_url, "HEAD:main"], check=True)
    print("[GIT] 已自動 push 最新 DB")

# ---------- 主程式 ---------- #
def main():
    if not CSV_PATH.is_file():
        sys.exit(f"[ERROR] 找不到 {CSV_PATH}")

    init_db()
    model = SentenceTransformer(EMBED_MODEL)

    print("== 讀取 CSV（混編碼）==")
    csv_lines = load_csv(CSV_PATH)
    rdr = csv.DictReader(io.StringIO("".join(csv_lines)))
    if "content" not in rdr.fieldnames:
        sys.exit("[ERROR] CSV 需要 content 欄")

    inserted = skipped = 0
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for i, row in enumerate(rdr, 1):
            content = (row.get("content") or "").strip()
            if not content:
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

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")

    # 最新五筆
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, content FROM posts ORDER BY id DESC LIMIT 5"
        ).fetchall()
    print("最後 5 筆：")
    for rid, txt in reversed(rows):
        print(f"{rid:>5} │ {txt.replace(chr(10), ' ')[:60]}")

    # git commit & push
    auto_commit(inserted)

    print("== 同步完成 ==")


if __name__ == "__main__":
    main()
