#!/usr/bin/env python3
# test_deepseek_chat.py

import os
import sys
import json
import requests
import logging

# 設定日誌輸出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# 讀取 API Key
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    logging.error("環境變數 DEEPSEEK_API_KEY 未設定！")
    sys.exit(1)

# DeepSeek Chat Completion Endpoint
URL = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 構造聊天訊息：隨便跟勇成聊天
messages = [
    {"role": "system", "content": "你是一個友好的聊天助手。"},
    {"role": "user",   "content": "勇成，隨便跟我聊幾句。"}
]

payload = {
    "model": "deepseek-chat",
    "messages": messages,
    "max_tokens": 50,
    "temperature": 0.9
}

# 顯示請求內容
logging.info(f"發送 POST {URL}")
logging.debug(f"Request Headers: {headers}")
logging.debug(f"Request Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

# 發送請求
resp = requests.post(URL, headers=headers, json=payload)
logging.info(f"HTTP 狀態碼：{resp.status_code}")
logging.debug(f"Response Content: {resp.text}")

# 處理回應
if resp.status_code == 200:
    data = resp.json()
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("\nDeepSeek 回覆：", reply)
else:
    logging.error(f"API 呼叫失敗：{resp.status_code}\n{resp.text}")
