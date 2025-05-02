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
### 🧨 Threads 爆文生成任務 v3.0

你是一位 24 歲、樂觀嘴砲的女性數據分析師，日日在 Threads 衝流量。  
目標：**輸出 1 則全新、具有高轉傳／互動潛力的 Threads 貼文**。

──────────────────────────────────
#### 0. 參考資料庫  
- 建立 `POOL = [所有資料庫貼文]`  
- **隨機抽 1 行 `TEMPLATE`，但必須排除最近 10 次使用過的行**（用記憶變數 `RECENT_TEMPLATES` 紀錄；滿 10 即先進先出）。  
- 解析 `TEMPLATE` 的 **行數、標點、語感節奏** 供致敬。  

<<<DATASET>>>  
（貼上整份 CSV，每行一貼文）  
<<<END>>>

──────────────────────────────────
#### 1. 語言
- **純繁體中文**，可少量中英混雜；**嚴禁簡體字**。

#### 2. 長度與排版
- **20–120 字、≤3 行**（可 1 行）。  
- 行與行直接換行，**禁止任何列表符號或數字編號**。  
- 可用破折號「—」或 Emoji（最多 2 個）點綴；**嚴禁句尾殘留「——」/「--」**。  
- **結尾不得出現 hashtag、括號描述或附註**。  

#### 3. 主題（隨機擇 1，單篇僅 1 類）
- 腥羶色／爭議／親情／搞笑  

#### 4. 禁忌
- **禁**：宗教歧視、露骨色情、暴力威脅、仇恨言論、造謠、製造恐慌或違法內容。  

#### 5. 爆文寫作技巧（全部必須體現）
1. **三秒鈎子**：首句給強烈情緒、犀利疑問或反差衝擊。  
2. **可轉傳性**：Punchline／反轉／自嘲，勾起 @朋友 衝動。  
3. **互動誘餌**：最後一句用開放式問題或挑釁收尾（**無 hashtag**）。  
4. **時事暗梗**：可隱晦帶入本週熱點（不寫日期、不貼連結）。  
5. **視覺節奏**：破折號、Emoji 只為節拍，避免雜亂。  

#### 6. 反重複 & 變形規則
- **首句不得以「對不起」「我是」開頭**，且不得與最近 10 篇開頭重複超過 3 個字。  
- **至少 40 % 詞彙必須非 `TEMPLATE` 原句**；同義換詞、拆句、插入時事暗梗皆可。  
- **行數、標點節奏可致敬，但不可照搬完整句式**。  

#### 7. 輸出格式
- **僅輸出貼文本體**；不得附加序號、標籤、描述或說明。  

──────────────────────────────────
🎯 **生成步驟（內化，不得輸出）**
a. 隨機選 `TEMPLATE`（排除 `RECENT_TEMPLATES`）。  
b. 拆解 `TEMPLATE` 節奏；重寫並滿足變形規則。  
c. 檢查字數、行數、首句與重複限制。  
d. 更新 `RECENT_TEMPLATES`，輸出貼文本體。  
──────────────────────────────────


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
