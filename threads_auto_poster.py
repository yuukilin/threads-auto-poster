#!/usr/bin/env python3
# threads_auto_poster.py
# DeepSeek 生成 + Threads Graph API 自動發文

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

sbert = SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_examples(top_k: int = 3):
    """隨機抽樣 + 餘弦排序，避免生成貼文千篇一律"""
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


def generate_post() -> str:
    examples = get_examples()
    prompt = """
請先完整閱讀下方示例，依其風格與長度生成全新 Threads 貼文。
規則：
⚠️ 生成規格  
1. **語言**：純繁體中文，可少量中英夾雜，禁止任何簡體字。  
2. **長度與排版**：  
   - 字數 ≤ 500，但 **優先 30-120 字／3 行以內** 的短貼：Threads 目前演算法偏好短句 :contentReference[oaicite:0]{index=0}。  
   - 允許自由換行、破折號「—」、Emoji（≤ 2 個），**結尾不得出現 hashtag、數字列表、括號動作描述**。  
3. **主題**：在「腥羶色／爭議／親情／搞笑」四大類中 **隨機擇一**，且單篇只用一種。  
4. **踩雷限制**：可挑戰性別、兩性、社會議題與「rage-bait 式反差」；**嚴禁宗教歧視、露骨色情、暴力威脅、仇恨言論** :contentReference[oaicite:1]{index=1}。  
5. **創新要求**：  
   - 嚴禁複製資料庫範例文字，但 **必須擷取其中一行的「節奏與格式」**（行數、標點、語感），不可混合多行。  
   - 開頭句式必須變化，不能每次同一開場白。  
6. **爆文寫作技巧**（務必全部具備）：  
   • **3-秒鈎子**：首句給強烈情緒／反常對比／犀利疑問，拒絕平鋪直敘。  
   • **可轉傳性**：使用 punchline、反轉或自嘲，讓讀者「想 tag 朋友」 :contentReference[oaicite:2]{index=2}。  
   • **互動誘餌**：最後一句拋出開放式問題或挑釁句，刺激留言，但**不可附 hashtag**。  
   • **時事潑辣**：可暗示當週熱門梗或新聞，增加共鳴；勿直接寫日期或貼新聞連結。  
   • **視覺節奏**：適度 Emoji / 破折號製造停頓感，但不可濫用。  
7. **輸出格式**：只回傳貼文本體，不附任何多餘說明、標籤或序號。  

示例貼文：

""".lstrip() + "\n".join(f"{i}. {txt}" for i, txt in enumerate(examples, 1))

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
    return resp.json()["choices"][0]["message"]["content"].strip()


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
