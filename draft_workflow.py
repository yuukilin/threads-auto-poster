#!/usr/bin/env python3
"""Daily draft/recovery check. Local files only; never accesses Threads or Keychain."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
ROOT = Path(__file__).resolve().parent
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def report_date(text: str) -> str | None:
    front = re.search(r"\A---\s*\n(.*?)\n---", text, re.S)
    field = re.search(r"^date:\s*[\"']?(\d{4}-\d{2}-\d{2})", front[1], re.M) if front else None
    heading = re.search(r"^#\s+(\d{4}-\d{2}-\d{2})", text, re.M)
    if field and heading and field[1] != heading[1]:
        return None
    return field[1] if field else heading[1] if heading else None


def find_delivery(codex_home: Path, thread_id: str, body: str, day: date) -> dict:
    """Require an exact body in a completed final answer in the requested history."""
    state_path = codex_home / "state_5.sqlite"
    history_path = codex_home / "thread_history_1.sqlite"
    if not state_path.is_file() or not history_path.is_file():
        return {"verified": False, "reason": "history_unavailable"}
    try:
        with closing(sqlite3.connect(f"file:{state_path}?mode=ro", uri=True, timeout=3)) as state:
            row = state.execute("SELECT rollout_path FROM threads WHERE id=?", (thread_id,)).fetchone()
        if not row:
            return {"verified": False, "reason": "thread_not_found"}
        ids = {thread_id}
        segments = UUID.findall(Path(row[0]).stem)
        if segments:
            ids.add(segments[-1])
        cutoff = int(datetime.combine(day, datetime.min.time(), TAIPEI).timestamp())
        placeholders = ",".join("?" for _ in ids)
        with closing(sqlite3.connect(f"file:{history_path}?mode=ro", uri=True, timeout=3)) as history:
            rows = history.execute(
                f"SELECT i.item_json,t.turn_id,t.completed_at FROM thread_items i "
                "JOIN thread_turns t ON i.thread_id=t.thread_id AND i.turn_id=t.turn_id "
                f"WHERE i.thread_id IN ({placeholders}) AND i.item_type='agentMessage' "
                "AND t.status='completed' AND t.completed_at>=? "
                "AND i.item_id=t.final_agent_item_id ORDER BY t.completed_at DESC",
                (*sorted(ids), cutoff),
            )
            for raw, turn_id, completed in rows:
                item = json.loads(raw)
                text = item.get("text", "").replace("\r\n", "\n")
                if body in text:
                    return {"verified": True, "turn_id": turn_id, "item_id": item["id"],
                            "completed_at": datetime.fromtimestamp(completed, TAIPEI).isoformat()}
        return {"verified": False, "reason": "full_draft_not_in_completed_final"}
    except (sqlite3.Error, ValueError, OSError) as error:
        return {"verified": False, "reason": "history_read_error", "error_type": type(error).__name__}


def check(repo: Path, codex_home: Path, thread_id: str, day: date, canonical_root: Path | None = None) -> dict:
    result = {"date": day.isoformat(), "thread_id": thread_id}
    if day.weekday() >= 5:
        return {**result, "action": "skip_weekend"}
    metadata_files = sorted((repo / "drafts").glob(f"{day.isoformat()}-*.json"))
    drafts = []
    for path in metadata_files:
        try:
            metadata = json.loads(path.read_text())
            if metadata.get("date") != day.isoformat():
                raise ValueError("draft_date_mismatch")
            if metadata.get("status") in ("superseded", "rejected", "cancelled"):
                continue
            body = path.with_suffix(".txt").read_text().replace("\r\n", "\n").rstrip("\n")
            if not body.strip() or len(body) > 500:
                raise ValueError("invalid_draft_length")
            drafts.append((path, metadata, body))
        except (OSError, ValueError) as error:
            return {**result, "action": "repair_draft", "file": str(path), "reason": str(error)}
    if len(drafts) > 1:
        return {**result, "action": "review_multiple_drafts", "files": [str(d[0]) for d in drafts]}
    if drafts:
        path, metadata, body = drafts[0]
        if metadata.get("status") == "published" and metadata.get("published_post_id"):
            return {**result, "action": "skip_published", "draft": str(path.with_suffix('.txt'))}
        evidence = find_delivery(codex_home, thread_id, body, day)
        return {**result, "action": "skip_delivered" if evidence["verified"] else "deliver_saved_draft",
                "draft": str(path.with_suffix('.txt')), "revision": metadata.get("revision"),
                "characters": len(body), "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "delivery": evidence}
    orphaned = sorted((repo / "drafts").glob(f"{day.isoformat()}-*.txt"))
    if orphaned:
        return {**result, "action": "repair_draft", "reason": "text_exists_without_metadata",
                "files": [str(p) for p in orphaned]}
    reports = [codex_home / "automations/daily-agri-check/last-run.md"]
    if canonical_root:
        reports.append(canonical_root / f"{day.isoformat()}-農產品日報.md")
    for report in reports:
        try:
            text = report.read_text()
        except OSError:
            continue
        # This is only a freshness precheck. Full source/price/topic validation
        # remains in AUTOMATION_PROMPT.md and cannot be bypassed by this result.
        if report_date(text) == day.isoformat() and "|" in text and "價格" in text:
            return {**result, "action": "generate_draft", "report": str(report)}
    return {**result, "action": "await_today_report", "checked": [str(p) for p in reports]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--record", action="store_true", help="Save local per-day check receipt")
    args = parser.parse_args()
    day = args.date or datetime.now(TAIPEI).date()
    canonical = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/卡片筆記盒模板/農產品追蹤/daily-report"
    result = check(ROOT, Path.home() / ".codex", args.thread_id, day, canonical)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    if args.record:
        destination = ROOT / ".local/draft-runs" / f"{day.isoformat()}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
