#!/usr/bin/env python3
# threads_auto_poster.py
# DeepSeek 生成 + Threads Graph API 自動發文（可隨機 1 行或多行）

import os
import sys
import sqlite3
import pickle
import random
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== 參數設定 =====
DB_PATH = "threads_db.sqlite"

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

THREADS_USER_ID  = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN = os.getenv("LONG_LIVED_TOKEN")
# ====================

for var in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[var]:
        sys.exit(f"[ERROR] 缺少環境變數 {var}")

# ──────────────────── 舊功能：取樣範例留著 ───────────────────
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_examples(top_k: int = 3):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT content, embedding FROM posts")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < top_k:
        sys.exit(f"[ERROR] posts 表少於 {top_k} 筆，請先跑 manage_posts.py")

    sample = random.sample(rows, k=min(10, len(rows)))
    q_vec  = sbert.encode("隨機取樣查詢")
    scored = [(_cosine(q_vec, pickle.loads(emb)), txt) for txt, emb in sample]
    scored.sort(reverse=True)
    return [txt for _, txt in scored[:top_k]]
# ────────────────────────────────────────────────────────────

RECENT_PATH   = "recent.pkl"          # 最近 20 條 template id
ENDINGS_PATH  = "recent_endings.pkl"  # 最近 10 篇是否問句

PROMPT_TEMPLATE = """# Threads 爆文生成器 v5

你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{template_line}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. **行數必須為 {target_lines} 行**；總字數 20–120。
   ‑ 行與行直接換行，禁止任何列表符號或數字編號。
3. 可用破折號「—」或 2 個以內 Emoji；句尾不得留「——」「--」。
4. 主題隨機選：腥羶色／爭議／親情／搞笑（單篇僅 1 類）。
5. 三秒鈎子：首句需高情緒或反差；首詞不得重複最近 10 篇的首 3 字。
6. 結尾可用 punchline、反轉句點，或問句誘餌（不必每篇都問）；禁 hashtag、括號附註。
7. 至少 50 % 詞彙必須與 TEMPLATE 不同；標點節奏可致敬但不可照抄。
8. **僅輸出貼文本體**，不得附加任何說明、標籤、序號。

（思考過程勿輸出）
"""

def pick_template() -> str:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, content FROM posts")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        sys.exit("[ERROR] 資料庫沒有貼文")

    recent = pickle.load(open(RECENT_PATH, "rb")) if os.path.exists(RECENT_PATH) else []
    pool   = [r for r in rows if r[0] not in recent] or rows
    tid, tline = random.choice(pool)

    recent.append(tid)
    recent = recent[-20:]
    pickle.dump(recent, open(RECENT_PATH, "wb"))
    return tline.strip()

def ends_with_q(text):      # 問句偵測
    return text.rstrip().endswith(("?", "？"))

def record_ending(is_q):
    data = pickle.load(open(ENDINGS_PATH, "rb")) if os.path.exists(ENDINGS_PATH) else []
    data.append(is_q)
    pickle.dump(data[-10:], open(ENDINGS_PATH, "wb"))

def too_many_q():
    data = pickle.load(open(ENDINGS_PATH, "rb")) if os.path.exists(ENDINGS_PATH) else []
    return len(data) >= 3 and all(data[-3:])

# ────────────────── 生成 ──────────────────
def generate_post() -> str:
    template_line = pick_template()
    tpl_lines     = template_line.count("\n") + 1

    # 決定目標行數：若模板 ≥2 行，40% 機率壓為 1 行
    target_lines = 1 if (tpl_lines >= 2 and random.random() < 0.4) else tpl_lines

    prompt = PROMPT_TEMPLATE.format(template_line=template_line,
                                    target_lines=target_lines)

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是 Threads 文案助手，只回傳貼文本體。"},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.9,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type":  "application/json"}

    for _ in range(6):  # 最多重試 6 次
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()

        # 基本驗證：行數 & 字數
        if (text.count("\n") + 1) != target_lines:
            continue
        if not (20 <= len(text) <= 120):
            continue
        # 問句機率控制
        is_q = ends_with_q(text)
        if too_many_q() and is_q:
            continue

        record_ending(is_q)
        return text

    # 若重試失敗，最後結果強行回傳
    record_ending(ends_with_q(text))
    return text
# ───────────────────────────────────────────

# === Threads 發布流程（完全未動） ===
def post_with_api(text: str):
    url_container = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    r1 = requests.post(
        url_container,
        data={
            "media_type":  "TEXT",
            "text":        text,
            "access_token": LONG_LIVED_TOKEN,
        }, timeout=30
    )
    r1.raise_for_status()
    container_id = r1.json().get("id") or sys.exit(f"[ERROR] container 失敗：{r1.text}")

    url_publish = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    r2 = requests.post(
        url_publish,
        data={
            "creation_id":  container_id,
            "access_token": LONG_LIVED_TOKEN,
        }, timeout=30
    )
    r2.raise_for_status()
    print("[API] 發文成功，Post id =", r2.json().get("id"))

# === 主程式 ===
if __name__ == "__main__":
    print("=== 生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n")

    print("=== 發布貼文 ===")
    post_with_api(post_text)
