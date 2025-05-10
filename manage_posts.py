#!/usr/bin/env python3
# manage_posts.py – 勇成專用 Threads 文章匯入工具（CSV ➜ posts.json）
# -------------------------------------------------------------
import os, csv, json, sys
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

try:
    import chardet
except ImportError:
    chardet = None

from sentence_transformers import SentenceTransformer

# ==== 參數 ====
BASE_DIR    = Path(__file__).resolve().parent
CSV_PATH    = BASE_DIR / "posts.csv"
JSON_PATH   = BASE_DIR / "posts.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
FALLBACK_ENC = ["utf-8-sig", "utf-8", "big5", "cp950"]
# ==============

def detect_encoding(path: Path) -> Optional[str]:
    if not chardet:
        return None
    with path.open("rb") as fh:
        raw = fh.read(8192)
    enc = chardet.detect(raw).get("encoding")
    return enc.lower() if enc else None

def open_csv(path: Path):
    tried = []
    enc = detect_encoding(path)
    if enc:
        tried.append(enc)
        try:
            fh = path.open(newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 偵測編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            pass
    for enc in FALLBACK_ENC:
        if enc in tried: continue
        try:
            fh = path.open(newline="", encoding=enc)
            rdr = csv.DictReader(fh)
            if rdr.fieldnames and "content" in rdr.fieldnames:
                print(f"[CSV] 使用備援編碼：{enc}")
                return fh, rdr
            fh.close()
        except UnicodeDecodeError:
            continue
    print(f"[ERROR] 無法判斷 {path} 編碼")
    return None, None

def load_json(path: Path) -> List[Dict]:
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return []

def save_json(path: Path, data: List[Dict]):
    try:
        # 確保檔案存在（避免權限問題）
        if not path.exists():
            path.touch(mode=0o644, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except PermissionError as e:
        sys.exit(f"[ERROR] 無法寫入 {path}：{e}\n"
                 f"請確認目錄權限，或執行：chmod -R u+rw {path.parent}")

def main():
    print("== 開始同步 posts.csv ➜ posts.json ==")
    if not CSV_PATH.is_file():
        sys.exit(f"[ERROR] 找不到 {CSV_PATH}")

    fh, rdr = open_csv(CSV_PATH)
    if rdr is None:
        sys.exit(1)

    model = SentenceTransformer(EMBED_MODEL)
    json_data = load_json(JSON_PATH)
    existing  = {item["content"]: item for item in json_data}
    next_id   = max((item["id"] for item in json_data), default=0) + 1
    inserted = skipped = 0

    for i, row in enumerate(rdr, 1):
        content = (row.get("content") or "").strip()
        if not content:
            print(f"[WARN] 第 {i} 行空白，跳過"); continue
        if content in existing:
            skipped += 1; continue
        emb = model.encode(content).tolist()
        json_data.append({
            "id":         next_id,
            "content":    content,
            "embedding":  emb,
            "created_at": datetime.now().isoformat(timespec="seconds")
        })
        next_id  += 1
        inserted += 1

    save_json(JSON_PATH, json_data)
    fh.close()

    print(f"[RESULT] 新增 {inserted} 筆，跳過 {skipped} 筆")
    if inserted:
        print("最新五筆：")
        for item in json_data[-5:]:
            print(f"{item['id']:>5} │ {item['content'].replace(chr(10), ' ')[:60]}")

if __name__ == "__main__":
    main()
