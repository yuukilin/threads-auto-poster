#!/usr/bin/env python3
# threads_auto_poster_api.py
# 結合 DeepSeek 生成 + 官方 Threads API 自動發文

import os
import sys
import sqlite3
import pickle
import requests
from sentence_transformers import SentenceTransformer

# ———— 設定區 ———— #
DB_PATH           = "threads_db.sqlite"
DEEPSEEK_KEY      = os.getenv("DEEPSEEK_API_KEY")      # DeepSeek Long Lived Key
DEEPSEEK_URL      = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL    = "deepseek-chat"

# 來自 Meta Graph API 的參數（文章所述官方 API）&#8203;:contentReference[oaicite:0]{index=0}
THREADS_USER_ID   = os.getenv("THREADS_USER_ID")       # 你的 Threads User ID
LONG_LIVED_TOKEN  = os.getenv("LONG_LIVED_TOKEN")      # 透過 grant_type=th_exchange_token 換取的長效 Token :contentReference[oaicite:1]{index=1}
# ———————————————— #

for v in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[v]:
        sys.exit(f"[ERROR] 環境變數 {v} 未設定！")

# 初始化向量模型
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def get_examples(top_k=3):
    """從 SQLite 讀入示例貼文，挑出 top_k 條最相似範例"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT content, embedding FROM posts")
    rows = cur.fetchall()
    conn.close()
    if len(rows) < top_k:
        sys.exit(f"[ERROR] posts 表少於 {top_k} 條示例，請先匯入！")
    # 簡單隨機 query
    q_vec = sbert.encode("示例查詢")
    sims  = []
    for content, blob in rows:
        emb = pickle.loads(blob)
        sim = (q_vec @ emb) / ((q_vec @ q_vec)**0.5 * (emb @ emb)**0.5)
        sims.append((sim, content))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in sims[:top_k]]

def generate_post() -> str:
    """呼叫 DeepSeek 生成爭議性貼文"""
    examples = get_examples()
    # 完整 Prompt (照文章要求)&#8203;:contentReference[oaicite:2]{index=2}
    prompt = """
請先完整閱讀下方所有示例貼文內容，根據它們的風格、長度與語氣，生成一條新的爭議性或情感性 Threads 貼文。生成文案必須與示例貼文保持相似的長度與節奏，且參照它們的氛圍。

請嚴格遵守以下規則：
1. 語言：純繁體中文，可偶爾中英夾雜，禁止簡體字。
2. 長度與形式：最多500字；可為一句 punchline、分行碎念、段落敘事，或使用 Emoji、換行；格式隨機多變；結尾禁止任何 hashtag；禁止任何括號內的附加說明或動作描述。
3. 主題：從「腥羶色／爭議／親情／搞笑」中隨機擇一，每次僅用一種主題。
4. 踩雷規則：可大膽挑戰性別、兩性、社會議題；禁止宗教歧視、露骨色情、暴力威脅、仇恨言論。
5. 內容創新：學習示例節奏與氛圍，嚴禁抄襲示例文字；開頭不得一成不變。
6. 輸出格式：僅回傳貼文本體，不附任何額外說明、分析或標籤。

示例貼文：
"""
    for i, ex in enumerate(examples, 1):
        prompt += f"{i}. {ex}\n"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是爭議性貼文助手，只回傳貼文內容本體。"},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.9
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type":  "application/json"
    }
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def post_with_api(text: str):
    """使用官方 Threads API 自動發文（兩步：create container + publish）"""
    # 1) Create Container :contentReference[oaicite:3]{index=3}
    url1 = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    data1 = {
        "media_type": "TEXT",
        "text":       text,
        "access_token": LONG_LIVED_TOKEN
    }
    r1 = requests.post(url1, data=data1)
    r1.raise_for_status()
    container_id = r1.json().get("id")
    if not container_id:
        sys.exit(f"[ERROR] 取得 container ID 失敗：{r1.text}")

    # 2) Publish Container :contentReference[oaicite:4]{index=4}
    url2 = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    data2 = {
        "creation_id":  container_id,
        "access_token": LONG_LIVED_TOKEN
    }
    r2 = requests.post(url2, data=data2)
    r2.raise_for_status()
    print("[API] 發文成功，Post ID：", r2.json().get("id"))

if __name__ == "__main__":
    print("=== 測試：先生成貼文 ===")
    post_text = generate_post()
    print(post_text, "\n\n")

    print("=== 測試：官方 API 發文 ===")
    post_with_api(post_text)
    print("=== 完成 ===")
