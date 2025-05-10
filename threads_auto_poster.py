#!/usr/bin/env python3
# threads_auto_poster.py – DeepSeek 生成 + Threads API 自動發文（快速安全版）
# ===============================================================

import os
import pickle
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv, set_key
from sentence_transformers import SentenceTransformer

# ─── 路徑 & 參數 ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "threads_db.sqlite"
RECENT_PATH  = BASE_DIR / "recent.pkl"          # 最近 20 筆 id
ENDINGS_PATH = BASE_DIR / "recent_endings.pkl"  # 最近 10 篇是否問句
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

THREADS_USER_ID  = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN = os.getenv("LONG_LIVED_TOKEN")

# ---- 可選安全：開或關 Quota 檢查 ----
CHECK_QUOTA = os.getenv("CHECK_THREADS_QUOTA", "0") == "1"
# ─────────────────────────────────────────────────────────

for var in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[var]:
        sys.exit(f"[ERROR] 缺少環境變數 {var}")

# ---- 向量工具 ----
sbert = SentenceTransformer("all-MiniLM-L6-v2")
def cosine(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def fetch_posts():
    if not DB_PATH.is_file():
        sys.exit("[ERROR] 找不到 threads_db.sqlite，先跑 manage_posts.py")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, content, embedding FROM posts").fetchall()
    return [{"id": rid, "content": cnt, "emb": pickle.loads(emb)} for rid, cnt, emb in rows]

# ---- 範例取樣（保留功能） ----
def get_examples(k=3):
    rows = fetch_posts()
    sample = random.sample(rows, k=min(10, len(rows)))
    q_vec  = sbert.encode("隨機取樣查詢")
    scored = [(cosine(q_vec, row["emb"]), row["content"]) for row in sample]
    return [txt for _, txt in sorted(scored, reverse=True)[:k]]

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
def ends_q(text): return text.rstrip().endswith(("?", "？"))
def record_q(is_q):
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    data.append(is_q)
    pickle.dump(data[-10:], open(ENDINGS_PATH, "wb"))
def too_many_q():
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    return len(data) >= 3 and all(data[-3:])

# ---- DeepSeek 生成 ----
PROMPT_TEMPLATE = """# Threads 爆文生成器 v5
你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{template}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. **行數必須為 {lines} 行**；總字數 20–120。
   - 行與行直接換行，禁止任何列表符號或數字編號。
3. 可用破折號「—」或 2 個以內 Emoji；句尾不得留「——」「--」。
4. 主題隨機選：腥羶色／爭議／親情／搞笑（單篇僅 1 類）。
5. 三秒鈎子：首句需高情緒或反差；首詞不得重複最近 10 篇的首 3 字。
6. 結尾可用 punchline、反轉句點，或問句誘餌（不必每篇都問）；禁 hashtag、括號附註。
7. 至少 50 % 詞彙必須與 TEMPLATE 不同；標點節奏可致敬但不可照抄。
8. **僅輸出貼文本體**，不得附加任何說明、標籤、序號。
（思考過程勿輸出）
"""

def generate_post():
    tpl = pick_template()
    base_lines = tpl.count("\n") + 1
    target = 1 if (base_lines_
