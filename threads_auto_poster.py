#!/usr/bin/env python3
# threads_auto_poster.py – DeepSeek 生成 + Threads API 自動發文（安全節流版）
# -------------------------------------------------------------
import os, sys, json, random, time, requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv, set_key

import numpy as np
from sentence_transformers import SentenceTransformer

# ==== 路徑 & 檔案 ====
BASE_DIR      = Path(__file__).resolve().parent
JSON_PATH     = BASE_DIR / "posts.json"
RECENT_PATH   = BASE_DIR / "recent.pkl"
ENDINGS_PATH  = BASE_DIR / "recent_endings.pkl"
ENV_FILE      = BASE_DIR / ".env"

# ==== 載入環境 ====
load_dotenv(ENV_FILE)
DEEPSEEK_KEY     = os.getenv("DEEPSEEK_API_KEY")
THREADS_USER_ID  = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN = os.getenv("LONG_LIVED_TOKEN")
DEEPSEEK_URL     = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"

for var in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[var]:
        sys.exit(f"[ERROR] 缺少環境變數 {var}")

# ==== 文字向量工具 ====
sbert = SentenceTransformer("all-MiniLM-L6-v2")
def cosine(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def load_posts():
    if not JSON_PATH.is_file():
        sys.exit("[ERROR] 找不到 posts.json，先跑 manage_posts.py")
    with JSON_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)

# ==== 範例取樣 ====
def get_examples(k=3):
    rows = load_posts()
    sample = random.sample(rows, k=min(10, len(rows)))
    q_vec  = sbert.encode("隨機取樣查詢")
    scored = [(cosine(q_vec, row["embedding"]), row["content"]) for row in sample]
    return [txt for _, txt in sorted(scored, reverse=True)[:k]]

# ==== 模板抽取 ====
def pick_template():
    posts = load_posts()
    recent = json.loads(RECENT_PATH.read_text()) if RECENT_PATH.exists() else []
    pool   = [p for p in posts if p["id"] not in recent] or posts
    chosen = random.choice(pool)
    recent.append(chosen["id"])
    RECENT_PATH.write_text(json.dumps(recent[-20:]))
    return chosen["content"].strip()

# ==== 問號管制 ====
def ends_q(t): return t.rstrip().endswith(("?", "？"))
def record_q(is_q):
    data = json.loads(ENDINGS_PATH.read_text()) if ENDINGS_PATH.exists() else []
    data.append(is_q); ENDINGS_PATH.write_text(json.dumps(data[-10:]))
def too_many_q():
    data = json.loads(ENDINGS_PATH.read_text()) if ENDINGS_PATH.exists() else []
    return len(data) >= 3 and all(data[-3:])

# ==== DeepSeek 生成 ====
PROMPT = """# Threads 爆文生成器 v5
你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{tpl}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. **行數必須為 {lines} 行**；總字數 20–120。
   - 行與行直接換行，禁止任何列表符號或數字編號。
3. 可用或 2 個以內 Emoji；句尾不得留「——」「--」。
4. 主題隨機選：腥羶色／爭議／親情／搞笑/感人（單篇僅 1 類）。
5. 三秒鈎子：首句需高情緒或反差；首詞不得重複最近 10 篇的首 3 字。
6. 結尾可用 punchline、反轉句點，或問句誘餌（不必每篇都問）；禁 hashtag、括號附註。
7. 至少 50 % 詞彙必須與 TEMPLATE 不同；標點節奏可致敬但不可照抄。
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
            {"role": "user",   "content": PROMPT.format(tpl=tpl, lines=target)}
        ],
        "max_tokens": 200,
        "temperature": 0.9,
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type":  "application/json"}

    for _ in range(6):
        r = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
        if (txt.count("\n")+1) != target: continue
        if not (20 <= len(txt) <= 120):  continue
        q = ends_q(txt)
        if too_many_q() and q:           continue
        record_q(q)
        return txt
    record_q(ends_q(txt))
    return txt

# ==== Token 續命（只剩 ≤7 天才刷） ====
REFRESH_URL = "https://graph.threads.net/v1.0/refresh_access_token"

def refresh_token(token):
    # 查詢剩餘天數
    expiry = datetime.now() + timedelta(days=60)  # Meta 沒提供 decode，用保守法
    days_left = (expiry - datetime.now()).days
    if days_left > 7:
        return token  # 還早

    try:
        r = requests.get(REFRESH_URL,
                         params={"grant_type": "ig_refresh_token",
                                 "access_token": token},
                         timeout=15)
        r.raise_for_status()
        new_token = r.json().get("access_token")
        if new_token and new_token != token:
            set_key(ENV_FILE, "LONG_LIVED_TOKEN", new_token)
            print("[TOKEN] 已自動續期 +60 天")
            return new_token
    except Exception as e:
        print(f"[WARN] token 續期失敗：{e}")
    return token

# ==== 查 quota（250/天） ====
def check_quota(token):
    url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publishing_limit"
    r = requests.get(url, params={"access_token": token}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("quota_usage", 0), data.get("config", {}).get("quota", 250)

# ==== 發文 ====
def post_thread(text, token):
    # 1. container
    url_c = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    r1 = requests.post(url_c, data={
        "media_type": "TEXT",
        "text":       text,
        "access_token": token,
    }, timeout=30)
    r1.raise_for_status()
    cid = r1.json().get("id") or sys.exit(f"[ERROR] container 失敗：{r1.text}")

    # 2. publish
    url_p = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    r2 = requests.post(url_p, data={
        "creation_id": cid,
        "access_token": token,
    }, timeout=30)
    r2.raise_for_status()
    print("[API] 發文成功，Post id =", r2.json().get("id"))

# ==== Main ====
if __name__ == "__main__":
    # 0. 續命
    LONG_LIVED_TOKEN = refresh_token(LONG_LIVED_TOKEN)

    # 1. 檢查 quota
    used, limit_ = check_quota(LONG_LIVED_TOKEN)
    if used >= 230:
        sys.exit(f"[INFO] 今日已用 {used}/{limit_}，停筆！")

    # 2. 生成內容
    print("=== 生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n")

    # 3. 發文
    print("=== 發布貼文 ===")
    post_thread(post_text, LONG_LIVED_TOKEN)

    # 4. 人性化間隔
    time.sleep(random.uniform(30, 180))
