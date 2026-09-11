from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PromptContractTests(unittest.TestCase):
    def test_daily_report_format_variants_are_supported(self):
        prompt = (ROOT / "AUTOMATION_PROMPT.md").read_text(encoding="utf-8")

        self.assertIn("## 今日價格", prompt)
        self.assertIn("## 價格確認", prompt)
        self.assertIn("H1 標題日期", prompt)
        self.assertIn("不綁死 Markdown 標題層級", prompt)
        self.assertIn("以品種為中心", prompt)

    def test_old_draft_cannot_block_a_new_workday(self):
        prompt = (ROOT / "AUTOMATION_PROMPT.md").read_text(encoding="utf-8")

        self.assertIn("每個工作日都是獨立批次", prompt)
        self.assertIn("不能阻擋、取代或延後今天的新草稿", prompt)

    def test_token_health_cannot_block_draft_generation(self):
        prompt = (ROOT / "AUTOMATION_PROMPT.md").read_text(encoding="utf-8")

        self.assertIn("權杖健康檢查與每日草稿是兩條獨立流程", prompt)
        self.assertIn("絕對不能阻擋、延後或取代當日草稿", prompt)


if __name__ == "__main__":
    unittest.main()
