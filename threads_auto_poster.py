#!/usr/bin/env python3
# threads_auto_poster.py
# DeepSeek 生成 + 官方 Threads API 自動發文

import os
import sys
import sqlite3
import pickle
import requests
from pathlib import Path
from huggingface_hub import login as hf_login
from sentence_transformers import SentenceTransformer

# ---------- 基本設定 ---------- #
DB_PATH        = "threads_db.sqlite"
DEEPSEEK_KEY   = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL   = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

THREADS_USER_ID   = os.getenv("THREADS_USER_ID")
LONG_LIVED_TOKEN  = os.getenv("LONG_LIVED_TOKEN")

EMBED_MODEL  = "all-MiniLM-L6-v2"
MODEL_CACHE  = str(Path.home() / ".cache" / "huggingface")
# -------------------------------- #

# ---- Hugging Face 認證 ---- #
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    hf_login(token=HF_TOKEN)

# 檢查必填環境變數
for v in ("DEEPSEEK_KEY", "THREADS_USER_ID", "LONG_LIVED_TOKEN"):
    if not globals()[v]:
        sys.exit(f"[ERROR] 環境變數 {v} 未設定！")

# 初始化向量模型（走本地快取）
sbert = SentenceTransformer(EMBED_MODEL, cache_folder=MODEL_CACHE)

def get_examples(top_k=3):
    """從 SQLite 抓示例貼文並算餘弦相似度"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT content, embedding FROM posts")
    rows = cur.fetchall()
    conn.close()

    if len(rows) < top_k:
        sys.exit(f"[ERROR] posts 表少於 {top_k} 條示例，請先匯入！")

    q_vec = sbert.encode("示例查詢")
    sims  = []
    for content, blob in rows:
        emb  = pickle.loads(blob)
        sim  = (q_vec @ emb) / ((q_vec @ q_vec)**0.5 * (emb @ emb)**0.5)
        sims.append((sim, content))

    sims.sort(reverse=True)
    return [c for _, c in sims[:top_k]]

def generate_post() -> str:
    """呼叫 DeepSeek 生成新貼文"""
    examples = get_examples()
    prompt = """
請先完整閱讀下方所有示例貼文內容，根據它們的風格、長度與語氣，生成一條新的爭議性或情感性 Threads 貼文。
生成文案必須與示例貼文保持相似的長度與節奏，且參照它們的氛圍。

請嚴格遵守以下規則：
1. 語言：純繁體中文，可偶爾中英夾雜，禁止簡體字。
2. 長度：最多 500 字；可使用換行、Emoji；結尾禁止任何 hashtag。
3. 主題：從「腥羶色／爭議／親情／搞笑」隨機擇一。
4. 踩雷規則：不可宗教歧視、不可暴力威脅或仇恨言論。
5. 內容創新：嚴禁抄襲示例文字；開頭要多樣。
6. 輸出：僅回傳貼文本體，不附任何額外說明。

示例貼文：
"""
    for i, ex in enumerate(examples, 1):
        prompt += f"{i}. {ex}\n"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是爭議性貼文助手，只回傳貼文內容本體。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.9
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}",
               "Content-Type": "application/json"}

    resp = requests.post(DEEPSEEK_URL, json=payload,
                         headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def post_with_api(text: str):
    """兩步驟：create container → publish"""
    # 1) 建 container
    url1 = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    data1 = {
        "media_type": "TEXT",
        "text": text,
        "access_token": LONG_LIVED_TOKEN
    }
    r1 = requests.post(url1, data=data1)
    r1.raise_for_status()
    container_id = r1.json().get("id")
    if not container_id:
        sys.exit(f"[ERROR] 取得 container ID 失敗：{r1.text}")

    # 2) 發佈
    url2 = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
    data2 = {
        "creation_id": container_id,
        "access_token": LONG_LIVED_TOKEN
    }
    r2 = requests.post(url2, data=data2)
    r2.raise_for_status()
    print("[API] 發文成功，Post ID：", r2.json().get("id"))

if __name__ == "__main__":
    post_text = generate_post()
    print("=== 預覽貼文 ===\n", post_text, "\n")
    post_with_api(post_text)
    print("=== 完成 ===")
