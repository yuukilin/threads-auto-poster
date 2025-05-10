#!/usr/bin/env python3
# threads_auto_poster.py – DeepSeek 生成 + Threads Graph API 自動發文（JSON 版）

import os, sys, json, pickle, random, requests, base64, time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, set_key

# ===== 參數設定 =====
BASE_DIR        = Path(__file__).resolve().parent
JSON_PATH       = BASE_DIR / "posts.json"
RECENT_PATH     = BASE_DIR / "recent.pkl"           # 最近 20 筆 id
ENDINGS_PATH    = BASE_DIR / "recent_endings.pkl"   # 最近 10 篇是否問句

load_dotenv()                                       # 讀 .env
DEEPSEEK_KEY       = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL       = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL     = "deepseek-chat"

THREADS_USER_ID    = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN   = os.getenv("LONG_LIVED_TOKEN")   # 會被更新，別寫死
TOKEN_FILE         = BASE_DIR / ".env"               # 直接改 .env

# === 檢查 env ===
for var in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[var]:
        sys.exit(f"[ERROR] 缺少環境變數 {var}")

# ────────── 向量工具 ──────────
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_posts():
    if not JSON_PATH.is_file():
        sys.exit(f"[ERROR] 找不到 {JSON_PATH}，請先跑 manage_posts.py")
    with JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)

# ────────── 取樣範例（保留原功能） ──────────
def get_examples(top_k=3):
    rows = load_posts()
    if len(rows) < top_k:
        sys.exit(f"[ERROR] posts 少於 {top_k} 筆")
    sample = random.sample(rows, k=min(10, len(rows)))
    q_vec  = sbert.encode("隨機取樣查詢")
    scored = [(_cosine(q_vec, item["embedding"]), item["content"]) for item in sample]
    scored.sort(reverse=True)
    return [txt for _, txt in scored[:top_k]]

# ────────── 模板挑選 ──────────
def pick_template():
    posts = load_posts()
    recent = pickle.load(open(RECENT_PATH, "rb")) if RECENT_PATH.exists() else []
    pool   = [p for p in posts if p["id"] not in recent] or posts
    chosen = random.choice(pool)

    recent.append(chosen["id"])
    pickle.dump(recent[-20:], open(RECENT_PATH, "wb"))
    return chosen["content"].strip()

# ────────── 行數 / 問句控制 ──────────
def ends_with_q(text): return text.rstrip().endswith(("?", "？"))
def record_ending(is_q):
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    data.append(is_q); pickle.dump(data[-10:], open(ENDINGS_PATH, "wb"))
def too_many_q():
    data = pickle.load(open(ENDINGS_PATH, "rb")) if ENDINGS_PATH.exists() else []
    return len(data) >= 3 and all(data[-3:])

# ────────── DeepSeek 生成 ──────────
PROMPT_TEMPLATE = """# Threads 爆文生成器 v5
你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{template_line}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. **行數必須為 {target_lines} 行**；總字數 20–120。
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
    template_line = pick_template()
    tpl_lines     = template_line.count("\n") + 1
    target_lines  = 1 if (tpl_lines >= 2 and random.random() < 0.4) else tpl_lines

    prompt = PROMPT_TEMPLATE.format(template_line=template_line, target_lines=target_lines)
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type":  "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是 Threads 文案助手，只回傳貼文本體。"},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.9,
    }

    for _ in range(6):  # 重試最多 6 次
        r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        if (text.count("\n")+1) != target_lines:  continue
        if not (20 <= len(text) <= 120):          continue
        is_q = ends_with_q(text)
        if too_many_q() and is_q:                continue
        record_ending(is_q)
        return text

    record_ending(ends_with_q(text))
    return text  # 就算全掛還是給最後一版

# ────────── Token 續命 ──────────
REFRESH_URL = "https://graph.threads.net/v1.0/refresh_access_token"

def refresh_token(token: str) -> str:
    try:
        r = requests.get(REFRESH_URL,
                         params={"grant_type": "ig_refresh_token",
                                 "access_token": token},
                         timeout=15)
        r.raise_for_status()
        data = r.json()
        new_token = data.get("access_token")
        if new_token and new_token != token:
            # 寫回 .env
            set_key(TOKEN_FILE, "LONG_LIVED_TOKEN", new_token)
            print(f"[TOKEN] 已自動續期（{data.get('expires_in')} 秒）")
            return new_token
    except Exception as e:
        print(f"[WARN] token 續期失敗：{e}")
    return token

# ────────── 發文 ──────────
def post_with_api(text: str, token: str):
    url_container = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    r1 = requests.post(url_container, data={
        "media_type":  "TEXT",
        "text":        text,
        "access_token": token,
    }, timeout=30)
    r1.raise_for_status()
    container_id = r1.json().get("id") or sys.exit(f"[ERROR] container 失敗：{r1.text}")

    url_publish = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    r2 = requests.post(url_publish, data={
        "creation_id":  container_id,
        "access_token": token,
    }, timeout=30)
    r2.raise_for_status()
    print("[API] 發文成功，Post id =", r2.json().get("id"))

# ────────── 主程式 ──────────
if __name__ == "__main__":
    # 1. 續命 token
    LONG_LIVED_TOKEN = refresh_token(LONG_LIVED_TOKEN)

    # 2. 生成貼文
    print("=== 生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n")

    # 3. 發布
    print("=== 發布貼文 ===")
    post_with_api(post_text, LONG_LIVED_TOKEN)
