#!/usr/bin/env python3
# threads_auto_poster.py – DeepSeek 生成 + Threads API 自動發文（v1.2）
# --------------------------------------------------------------
# 變更：
# 1. refresh 端點改為 graph.instagram.com/refresh_access_token
# 2. 400 錯誤（<24h 或已過期）視為「不必續期」而非報錯
# 3. 續期成功才更新 .env 與 .token_stamp

import os
import pickle
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv, set_key
from sentence_transformers import SentenceTransformer

# ─── 基本參數 ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DB_PATH   = BASE_DIR / "threads_db.sqlite"
RECENT_PATH  = BASE_DIR / "recent.pkl"
ENDINGS_PATH = BASE_DIR / "recent_endings.pkl"
TOKEN_STAMP  = BASE_DIR / ".token_stamp"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

THREADS_USER_ID  = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN = os.getenv("LONG_LIVED_TOKEN")

CHECK_QUOTA = os.getenv("CHECK_THREADS_QUOTA", "0") == "1"

for v in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[v]:
        sys.exit(f"[ERROR] 缺少環境變數 {v}")

# ─── 向量工具 ─────────────────────────────────────────────
sbert = SentenceTransformer("all-MiniLM-L6-v2")
def cosine(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def fetch_posts():
    if not DB_PATH.is_file():
        sys.exit("[ERROR] 找不到 threads_db.sqlite，先跑 manage_posts.py")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, content, embedding FROM posts").fetchall()
    return [{"id": rid, "content": cnt, "emb": pickle.loads(emb)} for rid, cnt, emb in rows]

# ---- 模板抽取 ----
def pick_template():
    posts  = fetch_posts()
    recent = pickle.load(open(RECENT_PATH, "rb")) if RECENT_PATH.exists() else []
    pool   = [p for p in posts if p["id"] not in recent] or posts
    chosen = random.choice(pool)
    recent.append(chosen["id"])
    pickle.dump(recent[-20:], open(RECENT_PATH, "wb"))
    return chosen["content"].strip()

# ---- 問號管制 ----
def ends_q(t): return t.rstrip().endswith(("?", "？"))
def record_q(is_q):
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    data.append(is_q); pickle.dump(data[-10:], open(ENDINGS_PATH, "wb"))
def too_many_q():
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    return len(data) >= 3 and all(data[-3:])

# ---- DeepSeek 生成 ----
PROMPT_TEMPLATE = """# Threads 爆文生成器 v5
你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{tpl}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. **行數必須為 {lines} 行**；總字數 20–200。
   - 行與行直接換行，禁止任何列表符號或數字編號。
3. 可使用2 個以內 Emoji；禁止使用破折號，包含「——」「--」皆不得出現。
4. 主題隨機選：腥羶色／爭議／親情／搞笑（單篇僅 1 類）。
5. 三秒鈎子：首句需高情緒或反差；首詞不得重複最近 10 篇的首 3 字。
6. 結尾可用 punchline、反轉句點，或問句誘餌（不必每篇都問）；禁 hashtag、括號附註。
7. 至少 80 % 詞彙必須與 TEMPLATE 不同；標點節奏可致敬但不可照抄。
8. **僅輸出貼文本體**，不得附加任何說明、標籤、序號。
（思考過程勿輸出）
"""

def generate_post():
    tpl = pick_template()
    base_lines = tpl.count("\n") + 1
    target = 1 if (base_lines >= 2 and random.random() < 0.4) else base_lines

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是 Threads 文案助手，只回傳貼文本體。"},
            {"role": "user",   "content": PROMPT_TEMPLATE.format(tpl=tpl, lines=target)}
        ],
        "max_tokens": 200,
        "temperature": 0.9,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type":  "application/json"}

    for _ in range(6):
        r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=25)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        if (text.count("\n")+1) != target: continue
        if not (20 <= len(text) <= 120):   continue
        if too_many_q() and ends_q(text):  continue
        record_q(ends_q(text))
        return text

    record_q(ends_q(text))
    return text

# ---- Token 續命 ----
REFRESH_URL = "https://graph.instagram.com/refresh_access_token"

def need_refresh() -> bool:
    if not TOKEN_STAMP.exists():
        return True
    ts = datetime.fromisoformat(TOKEN_STAMP.read_text().strip())
    return (datetime.now() - ts) > timedelta(days=53)  # 60 - 7

def refresh_token(token: str) -> str:
    if not need_refresh():
        return token
    try:
        r = requests.get(REFRESH_URL,
                         params={"grant_type": "ig_refresh_token",
                                 "access_token": token},
                         timeout=10)
        if r.status_code == 400:
            # <24h 或已過期；跳過但不終止
            print("[INFO] token 尚未到可刷新時間或已失效，跳過續期")
            return token
        r.raise_for_status()
        new_tok = r.json().get("access_token")
        if new_tok and new_tok != token:
            set_key(ENV_FILE, "LONG_LIVED_TOKEN", new_tok)
            TOKEN_STAMP.write_text(datetime.now().isoformat(timespec="seconds"))
            print("[TOKEN] 已自動續期 +60 天")
            return new_tok
    except Exception as e:
        print(f"[WARN] token 續期失敗：{e}")
    return token

# ---- 可選 Threads quota 檢查 ----
def quota_ok(token: str) -> bool:
    if not CHECK_QUOTA:
        return True
    try:
        url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publishing_limit"
        r = requests.get(url, params={"access_token": token}, timeout=8)
        r.raise_for_status()
        used = r.json().get("quota_usage", 0)
        return used < 250
    except Exception as e:
        print(f"[WARN] quota 查詢失敗：{e}")
        return True

# ---- 發文 ----
def post_thread(text: str, token: str):
    url_c = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    r1 = requests.post(url_c, data={
        "media_type": "TEXT",
        "text":       text,
        "access_token": token,
    }, timeout=25)
    r1.raise_for_status()
    cid = r1.json().get("id") or sys.exit(f"[ERROR] container 失敗：{r1.text}")

    url_p = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    r2 = requests.post(url_p, data={
        "creation_id":  cid,
        "access_token": token,
    }, timeout=25)
    r2.raise_for_status()
    print("[API] 發文成功，Post id =", r2.json().get("id"))

# ---- Main ----
if __name__ == "__main__":
    LONG_LIVED_TOKEN = refresh_token(LONG_LIVED_TOKEN)

    if not quota_ok(LONG_LIVED_TOKEN):
        sys.exit("[INFO] 今日 quota 用盡，跳過發文")

    print("=== 生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n")

    print("=== 發布貼文 ===")
    post_thread(post_text, LONG_LIVED_TOKEN)

    print("== Done ==")
