#!/usr/bin/env python3
# manage_posts.py – CSV ➜ SQLite，自動 git push（偵測未追蹤檔案）
# ==================================================================

import csv, io, os, pickle, sqlite3, subprocess, sys
from pathlib import Path
from typing import List, Optional
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "posts.csv"
DB_PATH  = BASE_DIR / "threads_db.sqlite"
MODEL    = "all-MiniLM-L6-v2"
ENCODING_TRY = ["utf-8-sig", "utf-8", "big5", "cp950"]

# ---------- 基礎 ----------
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

# ---------- Git push ----------
def file_tracked(filepath: Path) -> bool:
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(filepath)],
        capture_output=True, text=True
    )
    return res.returncode == 0

def git_push_db(force_push: bool):
    """force_push=True 代表 DB 檔未追蹤，需要先 add 再推"""
    pat = os.getenv("GH_PAT")
    if not pat:
        print("[INFO] GH_PAT 未設，DB 只留在 Action，不 push")
        return

    subprocess.run(["git", "config", "--global", "user.email", "bot@threads"])
    subprocess.run(["git", "config", "--global", "user.name",  "threads-bot"])
    remote = subprocess.check_output(
        ["git", "config", "--get", "remote.origin.url"], text=True
    ).strip()
    if remote.startswith("https://"):
        remote = remote.replace("https://", f"https://{pat}@")

    if force_push:
        subprocess.run(["git", "add", str(DB_PATH)], check=True)

    # --allow-empty：就算沒變動也能產生一顆 commit
    subprocess.run(
        ["git", "commit", "--allow-empty",
         "-m", "chore(db): sync threads_db.sqlite [skip ci]"], check=True)
    subprocess.run(["git", "push", remote, "HEAD:main"], check=True)
    print("[GIT] DB 已 push 至 repo")

# ---------- Main ----------
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
                continue
            if cur.execute("SELECT 1 FROM posts WHERE content = ?", (content,)).fetchone():
                skipped += 1
                continue
            emb = pickle.dumps(model.encode(content))
            cur.execute("INSERT INTO posts (content, embedding) VALUES (?, ?)", (content, emb))
            inserted += 1
        conn.commit()

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")

    # ── 關鍵：如果檔案未被追蹤 OR 有新增貼文，就 push ──
    must_push = (not file_tracked(DB_PATH)) or (inserted > 0)
    git_push_db(must_push)

if __name__ == "__main__":
    main()
