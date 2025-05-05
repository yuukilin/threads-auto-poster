#!/usr/bin/env python3
# threads_auto_poster.py
# DeepSeek 生成 + Threads Graph API 自動發文（格式隨模板多樣化版）

import os
import sys
import sqlite3
import pickle
import random
import json
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
# 嵌入相關：保留舊功能（選填，可用於其他場景）
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def _cosine(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_examples(top_k: int = 3):
    """舊版保留：從 DB 抽樣貼文作參考，不影響新生成邏輯"""
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

# ===== 新增：模板抽樣與 Prompt =====
RECENT_PATH = "recent.pkl"            # 紀錄最近使用過的 template id 列表 (最多 20)

PROMPT_TEMPLATE = """# Threads 爆文生成器 v4

你是一位 24 歲、嘴砲又樂觀的女性數據分析師，天天在 Threads 衝流量。
你的任務：**重寫下方 TEMPLATE，生成 1 則全新 Threads 貼文**。

【TEMPLATE】
{template_line}
【/TEMPLATE】

── 規格 ──────────────────────────
1. 純繁體中文，可少量中英混雜；嚴禁簡體字。
2. 行數必須與 TEMPLATE 相同；總字數 20–120。  
   ‑ 行與行直接換行，禁止任何列表符號、數字編號。  
3. 可使用破折號「—」或 2 個以內 Emoji 點綴；**句尾不得留「——」「--」**。  
4. 主題隨機選：腥羶色／爭議／親情／搞笑（單篇只用一類）。  
5. **三秒鈎子**：首句必須帶強情緒或反差；**首詞不得重複最近 10 篇的首 3 個字**（如：今天、對不起、我是）。  
6. 結尾用開放式問題或挑釁收尾；**禁止 hashtag、括號附註**。  
7. 至少 50 % 詞彙必須與 TEMPLATE 不同；語感、標點節奏可致敬但不得照抄。  
8. 只輸出貼文本體，不得多任何說明、標籤、序號。

（內部思考：拆解 TEMPLATE 行數與節奏，執行以上規則 → 產出新貼文。思考過程勿輸出）
"""

def pick_template() -> str:
    """隨機挑 1 行當模板，避開最近 20 次使用過的"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, content FROM posts")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        sys.exit("[ERROR] 資料庫沒有貼文可供抽樣")

    # 讀取最近使用過的 id 列表
    if os.path.exists(RECENT_PATH):
        recent = pickle.load(open(RECENT_PATH, "rb"))
    else:
        recent = []

    pool = [r for r in rows if r[0] not in recent]
    if not pool:            # 若全部都用過，刷新 recent
        recent = []
        pool   = rows

    tid, tline = random.choice(pool)

    # 更新 recent 列表
    recent.append(tid)
    recent = recent[-20:]   # 只保留最近 20 筆
    pickle.dump(recent, open(RECENT_PATH, "wb"))

    return tline.strip()
# ========================================================


def generate_post() -> str:
    """生成 1 則符合規格的 Threads 貼文"""
    template_line = pick_template()             # 先由程式決定要模仿哪一行
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

    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # 簡易防呆：若 DeepSeek 偷偷多行或超字數，就重生一次（最多 3 次）
    for _ in range(2):
        line_cnt = content.count("\n") + 1
        if line_cnt != template_line.count("\n") + 1 or not (20 <= len(content) <= 120):
            resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            break

    return content


# ─────────────────────────────────────────────────────────
# 以下「發布 Threads」與主程式區域【完全未動】
def post_with_api(text: str):
    # Step 1：Create container
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

    # Step 2：Publish container
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
