#!/usr/bin/env python3
# threads_auto_poster.py
# DeepSeek 生成 + Threads Graph API 自動發文（多樣化版‧問句機率調整）

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

DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")    # DeepSeek Long-Lived Key
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Threads Graph API
THREADS_USER_ID  = os.getenv("THREADS_USER_ID")   # 你的 Threads user id
LONG_LIVED_TOKEN = os.getenv("LONG_LIVED_TOKEN")  # th_exchange_token 換到的 token
# ====================

for var in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[var]:
        sys.exit(f"[ERROR] 缺少環境變數 {var}")

# ─────────────────────────────────────────────────────────
# 嵌入相關（舊功能保留，給其他流程用）
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a, b) -> float:
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

    sample_rows = random.sample(rows, k=min(len(rows), 10))
    q_vec       = sbert.encode("隨機取樣查詢")
    scored      = [(_cosine(q_vec, pickle.loads(blob)), txt) for txt, blob in sample_rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
# ─────────────────────────────────────────────────────────

# ===== 新增：模板抽樣、Prompt 與問句機率控制 =====
RECENT_PATH   = "recent.pkl"           # 最近 20 筆 template id
ENDINGS_PATH  = "recent_endings.pkl"   # 最近 10 筆是否為問句

PROMPT_TEMPLATE = """# Threads 爆文生成器 v4.1

你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
你的任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{template_line}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. 行數必須與 TEMPLATE 相同；總字數 20–120。  
   ‑ 行與行直接換行，禁止任何列表符號、數字編號。  
3. 可使用2個以內 Emoji 點綴；**句尾不得留「——」「--」「-」**。  
4. 主題隨機選：腥羶色／爭議／親情／搞笑（單篇只用一類）。  
5. **三秒鈎子**：首句必須帶強情緒或反差；**首詞不得重複最近 10 篇的首 3 個字**（如：今天、對不起、我是）。  
6. 結尾可用 punchline、反轉句號，或開放式問題／挑釁（機率自行評估，**不必每篇都用問句**）；禁止 hashtag、括號附註。  
7. 至少 50 % 詞彙必須與 TEMPLATE 不同；語感、標點節奏可致敬但不得照抄。  
8. 只輸出貼文本體，不得多任何說明、標籤、序號。

（內部思考：拆解 TEMPLATE 行數與節奏，執行以上規則 → 產出新貼文。思考過程勿輸出）
"""

def pick_template() -> str:
    """隨機選 1 行當 Template，避開最近 20 次"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, content FROM posts")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        sys.exit("[ERROR] 資料庫沒有貼文可供抽樣")

    recent = pickle.load(open(RECENT_PATH, "rb")) if os.path.exists(RECENT_PATH) else []

    pool = [r for r in rows if r[0] not in recent] or rows
    tid, tline = random.choice(pool)

    # 更新 recent
    recent.append(tid)
    recent = recent[-20:]
    pickle.dump(recent, open(RECENT_PATH, "wb"))
    return tline.strip()

def ends_with_question(text: str) -> bool:
    return text.rstrip().endswith("?") or text.rstrip().endswith("？")

def update_endings(is_q: bool):
    lst = pickle.load(open(ENDINGS_PATH, "rb")) if os.path.exists(ENDINGS_PATH) else []
    lst.append(is_q)
    lst = lst[-10:]
    pickle.dump(lst, open(ENDINGS_PATH, "wb"))

def too_many_questions() -> bool:
    """檢查最近 3 篇是否全為問句"""
    lst = pickle.load(open(ENDINGS_PATH, "rb")) if os.path.exists(ENDINGS_PATH) else []
    return len(lst) >= 3 and all(lst[-3:])

# ========================================================


def generate_post() -> str:
    template_line = pick_template()
    prompt = PROMPT_TEMPLATE.format(template_line=template_line)

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

    for _ in range(5):               # 最多嘗試 5 次
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # 字數 / 行數 防呆
        if (content.count("\n") + 1) != (template_line.count("\n") + 1):
            continue
        if not (20 <= len(content) <= 120):
            continue

        # 問句機率控制：若最近 3 篇皆為問句，強制本篇不能再問
        is_q = ends_with_question(content)
        if too_many_questions() and is_q:
            continue

        update_endings(is_q)
        return content

    # 若多次都不符，只好回傳最後一次結果
    update_endings(ends_with_question(content))
    return content


# ─────────────────────────────────────────────────────────
# 以下 Threads API 與主程式區域「完全未動」
def post_with_api(text: str):
    url_container = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    r1 = requests.post(
        url_container,
        data={
            "media_type":  "TEXT",
            "text":        text,
            "access_token": LONG_LIVED_TOKEN,
        },
        timeout=30
    )
    r1.raise_for_status()
    container_id = r1.json().get("id")
    if not container_id:
        sys.exit(f"[ERROR] 取得 container id 失敗：{r1.text}")

    url_publish = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    r2 = requests.post(
        url_publish,
        data={
            "creation_id":  container_id,
            "access_token": LONG_LIVED_TOKEN,
        },
        timeout=30
    )
    r2.raise_for_status()
    print("[API] 發文成功，Post id =", r2.json().get("id"))


if __name__ == "__main__":
    print("=== 生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n")

    print("=== 發布貼文 ===")
    post_with_api(post_text)
