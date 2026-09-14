#!/usr/bin/env python3
"""Minimal, approval-gated client for the Meta Threads API.

The access token is read from macOS Keychain at runtime. It is never read from
an environment file and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


KEYCHAIN_ACCOUNT = "lin.yusei"
KEYCHAIN_SERVICE = "com.yuukilin.threads-auto-poster.access-token"
EXPECTED_USERNAME = "lin.yusei"
GRAPH_BASE = os.environ.get(
    "THREADS_GRAPH_BASE", "https://graph.threads.com/v1.0"
).rstrip("/")
GRAPH_ORIGIN = os.environ.get(
    "THREADS_GRAPH_ORIGIN", "https://graph.threads.com"
).rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30
REFRESH_STATE_PATH = (
    Path.home()
    / ".codex"
    / "automations"
    / "threads-token-refresh"
    / "last-run.json"
)
KEYCHAIN_UPDATE_HELPER = (
    Path(__file__).resolve().parent / ".local" / "bin" / "keychain_update"
)


class ThreadsApiError(RuntimeError):
    """A safe-to-display Threads API error with secrets removed."""


def load_access_token() -> str:
    if not KEYCHAIN_UPDATE_HELPER.is_file():
        raise ThreadsApiError("缺少 Mac 鑰匙圈安全讀取程式。")
    try:
        result = subprocess.run(
            [
                str(KEYCHAIN_UPDATE_HELPER),
                "read",
                KEYCHAIN_ACCOUNT,
                KEYCHAIN_SERVICE,
            ],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise ThreadsApiError("讀取 Mac 鑰匙圈逾時。") from None
    token = result.stdout.decode("utf-8").strip()
    if result.returncode != 0 or not token:
        raise ThreadsApiError("找不到 Threads 權杖，請先將它存入 Mac 鑰匙圈。")
    return token


def save_access_token(token: str) -> None:
    """Update Keychain through Security.framework without exposing the token.

    The token travels only over stdin to a local native helper. It is never put
    in argv, a shell command, an environment variable, stdout/stderr, or a file.
    This also avoids the 128-character limit of ``security -w`` prompt mode.
    """
    if not KEYCHAIN_UPDATE_HELPER.is_file():
        raise ThreadsApiError("缺少 Mac 鑰匙圈安全更新程式。")
    try:
        result = subprocess.run(
            [
                str(KEYCHAIN_UPDATE_HELPER),
                "update",
                KEYCHAIN_ACCOUNT,
                KEYCHAIN_SERVICE,
                "Threads API token - lin.yusei",
            ],
            input=token.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise ThreadsApiError("更新 Mac 鑰匙圈逾時。") from None
    if result.returncode != 0:
        raise ThreadsApiError("無法將新權杖寫入 Mac 鑰匙圈。")
    if load_access_token() != token:
        raise ThreadsApiError("Mac 鑰匙圈寫入後比對失敗，未記錄續期成功。")


def _safe_api_error(error: urllib.error.HTTPError, token: str) -> ThreadsApiError:
    message = "Meta Threads API 呼叫失敗"
    code: Any = error.code
    try:
        payload = json.loads(error.read().decode("utf-8"))
        details = payload.get("error", payload)
        code = details.get("code", code)
        message = details.get("message", message)
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        pass
    return ThreadsApiError(f"{message.replace(token, '[REDACTED]')}（代碼 {code}）")


def graph_request(
    path: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    token: str,
) -> dict[str, Any]:
    payload = dict(params or {})
    payload["access_token"] = token
    encoded = urllib.parse.urlencode(payload).encode("utf-8")
    url = f"{GRAPH_BASE}/{path.lstrip('/')}"
    if method == "GET":
        request = urllib.request.Request(
            f"{url}?{encoded.decode('utf-8')}", method="GET"
        )
    else:
        request = urllib.request.Request(url, data=encoded, method=method)

    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise _safe_api_error(error, token) from None
    except urllib.error.URLError as error:
        raise ThreadsApiError(f"無法連線至 Meta Threads API：{error.reason}") from None


def _request_refreshed_token(token: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {"grant_type": "th_refresh_token", "access_token": token}
    )
    request = urllib.request.Request(
        f"{GRAPH_ORIGIN}/refresh_access_token?{params}", method="GET"
    )
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            body = response.read().decode("utf-8").strip()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        raise _safe_api_error(error, token) from None
    except urllib.error.URLError as error:
        raise ThreadsApiError(f"無法連線至 Meta Threads API：{error.reason}") from None


def _write_refresh_state(state: dict[str, Any]) -> None:
    REFRESH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = REFRESH_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(REFRESH_STATE_PATH)


def refresh_access_token() -> dict[str, Any]:
    current_token = load_access_token()
    refreshed = _request_refreshed_token(current_token)
    new_token = refreshed.get("access_token") or current_token

    profile = graph_request(
        "me", params={"fields": "id,username"}, token=new_token
    )
    if profile.get("username") != EXPECTED_USERNAME:
        raise ThreadsApiError("續期後帳號檢查失敗；原權杖已保留。")

    if new_token != current_token:
        save_access_token(new_token)

    now = datetime.now(timezone.utc)
    expires_in = refreshed.get("expires_in")
    state: dict[str, Any] = {
        "ok": True,
        "refreshed_at": now.isoformat(),
        "username": EXPECTED_USERNAME,
        "token_rotated": new_token != current_token,
    }
    if isinstance(expires_in, int) and expires_in > 0:
        state["expires_in_seconds"] = expires_in
        state["estimated_expires_at"] = (
            now + timedelta(seconds=expires_in)
        ).isoformat()
    _write_refresh_state(state)
    return state


def refresh_status() -> dict[str, Any]:
    if not REFRESH_STATE_PATH.exists():
        return {"ok": False, "status": "尚無自動續期紀錄"}
    try:
        state = json.loads(REFRESH_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "status": "續期紀錄無法讀取"}
    return state


def verify_profile() -> dict[str, Any]:
    token = load_access_token()
    profile = graph_request(
        "me",
        params={"fields": "id,username,name"},
        token=token,
    )
    username = profile.get("username")
    if username != EXPECTED_USERNAME:
        raise ThreadsApiError(
            f"權杖屬於 @{username or '未知帳號'}，不是預期的 @{EXPECTED_USERNAME}。"
        )
    return {
        "ok": True,
        "id": profile.get("id"),
        "username": username,
        "name": profile.get("name"),
    }


def validate_text(text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if not clean_text:
        raise ThreadsApiError("貼文內容不可為空白。")
    if len(clean_text) > 500:
        raise ThreadsApiError(f"貼文共有 {len(clean_text)} 字元，超過 Threads 的 500 字元上限。")
    return {"ok": True, "characters": len(clean_text), "limit": 500}


def publish_text(text: str, approval: str) -> dict[str, Any]:
    if approval != "發送":
        raise ThreadsApiError("缺少明確核准；只有 approval=發送 才允許發布。")

    clean_text = text.strip()
    validate_text(clean_text)

    token = load_access_token()
    profile = graph_request(
        "me", params={"fields": "id,username"}, token=token
    )
    if profile.get("username") != EXPECTED_USERNAME:
        raise ThreadsApiError("發布前帳號檢查失敗，已停止發文。")

    container = graph_request(
        "me/threads",
        method="POST",
        params={"media_type": "TEXT", "text": clean_text},
        token=token,
    )
    creation_id = container.get("id")
    if not creation_id:
        raise ThreadsApiError("Meta 未回傳貼文容器 ID，已停止發文。")

    last_error: ThreadsApiError | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(attempt)
        try:
            published = graph_request(
                "me/threads_publish",
                method="POST",
                params={"creation_id": creation_id},
                token=token,
            )
            post_id = published.get("id")
            if not post_id:
                raise ThreadsApiError("Meta 未回傳貼文 ID，發布結果不明。")
            return {"ok": True, "id": post_id, "username": EXPECTED_USERNAME}
        except ThreadsApiError as error:
            last_error = error

    raise last_error or ThreadsApiError("Threads 發布失敗。")


def inspect_post(post_id: str) -> dict[str, Any]:
    token = load_access_token()
    post = graph_request(
        post_id,
        params={"fields": "id,text,timestamp,permalink,username"},
        token=token,
    )
    if post.get("username") != EXPECTED_USERNAME:
        raise ThreadsApiError("回查到的貼文不屬於預期帳號。")
    return {
        "ok": True,
        "id": post.get("id"),
        "username": post.get("username"),
        "text": post.get("text"),
        "timestamp": post.get("timestamp"),
        "permalink": post.get("permalink"),
    }


def _read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="安全操作 lin.yusei 的 Threads API")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="唯讀驗證權杖所屬帳號")
    subparsers.add_parser("refresh", help="刷新長效權杖並更新 Mac 鑰匙圈")
    subparsers.add_parser("status", help="查看最近一次自動續期狀態")
    validate_parser = subparsers.add_parser("validate", help="檢查草稿字數與格式")
    validate_parser.add_argument("--file", required=True, help="UTF-8 草稿檔案")
    inspect_parser = subparsers.add_parser("inspect", help="唯讀回查已發布貼文")
    inspect_parser.add_argument("--id", required=True, help="Threads 貼文 ID")

    publish_parser = subparsers.add_parser("publish", help="發布已核准的文字草稿")
    publish_parser.add_argument("--file", required=True, help="UTF-8 草稿檔案")
    publish_parser.add_argument(
        "--approval",
        required=True,
        help="必須精確填入「發送」",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "verify":
            result = verify_profile()
        elif args.command == "refresh":
            result = refresh_access_token()
        elif args.command == "status":
            result = refresh_status()
        elif args.command == "validate":
            result = validate_text(_read_text_file(args.file))
        elif args.command == "inspect":
            result = inspect_post(args.id)
        else:
            result = publish_text(_read_text_file(args.file), args.approval)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ThreadsApiError, OSError) as error:
        if args.command == "refresh":
            try:
                _write_refresh_state(
                    {
                        "ok": False,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(error),
                    }
                )
            except OSError:
                pass
        print(f"錯誤：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
