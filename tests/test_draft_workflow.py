import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("draft_workflow", Path(__file__).resolve().parents[1] / "draft_workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
THREAD = "01a060db-205e-7650-94dc-c1da579e036c"
SEGMENT = "01a0610d-9444-7a20-8984-bd5d947aff0c"
DAY = date(2026, 9, 16)


class DailyRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.home = self.root / "codex"
        (self.repo / "drafts").mkdir(parents=True)
        (self.home / "automations/daily-agri-check").mkdir(parents=True)
        self.report("2026-09-16")

    def report(self, day):
        (self.home / "automations/daily-agri-check/last-run.md").write_text(
            f"---\ndate: {day}\n---\n# {day} 農產品日報\n## 價格確認\n|咖啡|1.6%|\n")

    def draft(self, day="2026-09-16", body="當天完整草稿。", status="awaiting_approval"):
        p = self.repo / "drafts" / f"{day}-coffee.json"
        p.write_text(json.dumps({"date": day, "status": status, "revision": 1}))
        p.with_suffix(".txt").write_text(body + "\n")
        return p

    def delivery(self, body, status="completed", final=True):
        with closing(sqlite3.connect(self.home / "state_5.sqlite")) as c:
            c.execute("CREATE TABLE threads(id TEXT,rollout_path TEXT)")
            c.execute("INSERT INTO threads VALUES(?,?)", (THREAD, f"/tmp/rollout-{THREAD}_{SEGMENT}.jsonl"))
            c.commit()
        with closing(sqlite3.connect(self.home / "thread_history_1.sqlite")) as c:
            c.execute("CREATE TABLE thread_items(thread_id TEXT,turn_id TEXT,item_id TEXT,item_type TEXT,item_json TEXT)")
            c.execute("CREATE TABLE thread_turns(thread_id TEXT,turn_id TEXT,status TEXT,completed_at INT,final_agent_item_id TEXT)")
            # A resumed thread stores recent messages in its latest segment.
            c.execute("INSERT INTO thread_items VALUES(?,?,?,?,?)", (SEGMENT, "turn", "message", "agentMessage", json.dumps({"id":"message","text":body})))
            c.execute("INSERT INTO thread_turns VALUES(?,?,?,?,?)", (SEGMENT, "turn", status, 1789531200, "message" if final else "other"))
            c.commit()

    def check(self):
        return workflow.check(self.repo, self.home, THREAD, DAY)

    def test_old_awaiting_approval_does_not_block_new_day(self):
        self.draft(day="2026-09-15")
        self.assertEqual(self.check()["action"], "generate_draft")

    def test_saved_but_undelivered_draft_is_resumed(self):
        self.draft()
        self.assertEqual(self.check()["action"], "deliver_saved_draft")

    def test_exact_final_delivery_prevents_duplicate_notification(self):
        self.draft()
        self.delivery("說明\n當天完整草稿。\n尚未發布")
        self.assertEqual(self.check()["action"], "skip_delivered")

    def test_partial_message_does_not_claim_delivery(self):
        self.draft()
        self.delivery("草稿已完成")
        self.assertEqual(self.check()["action"], "deliver_saved_draft")

    def test_running_turn_is_not_completed_delivery(self):
        self.draft()
        self.delivery("當天完整草稿。", status="inProgress")
        self.assertEqual(self.check()["action"], "deliver_saved_draft")

    def test_commentary_is_not_final_delivery(self):
        self.draft()
        self.delivery("當天完整草稿。", final=False)
        self.assertEqual(self.check()["action"], "deliver_saved_draft")

    def test_revision_requires_its_own_full_delivery(self):
        self.draft(body="新版完整草稿。")
        self.delivery("當天完整草稿。")
        self.assertEqual(self.check()["action"], "deliver_saved_draft")

    def test_stale_report_waits_for_retry(self):
        self.report("2026-09-15")
        self.assertEqual(self.check()["action"], "await_today_report")

    def test_failed_token_state_does_not_affect_generation(self):
        p = self.home / "automations/threads-token-refresh"
        p.mkdir()
        (p / "last-run.json").write_text('{"ok":false}')
        self.assertEqual(self.check()["action"], "generate_draft")

    def test_missing_metadata_recovers_saved_text(self):
        self.draft().unlink()
        self.assertEqual(self.check()["action"], "repair_draft")

    def test_weekend_stays_quiet(self):
        result = workflow.check(self.repo, self.home, THREAD, date(2026, 9, 19))
        self.assertEqual(result["action"], "skip_weekend")


if __name__ == "__main__":
    unittest.main()
