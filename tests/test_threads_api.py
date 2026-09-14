import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import threads_api


class ThreadsApiTests(unittest.TestCase):
    def test_verify_rejects_token_for_another_account(self):
        with (
            patch.object(threads_api, "load_access_token", return_value="secret"),
            patch.object(
                threads_api,
                "graph_request",
                return_value={"id": "1", "username": "wrong.account"},
            ),
        ):
            with self.assertRaisesRegex(threads_api.ThreadsApiError, "不是預期"):
                threads_api.verify_profile()

    def test_publish_requires_exact_approval_before_reading_token(self):
        with patch.object(threads_api, "load_access_token") as load_token:
            with self.assertRaisesRegex(threads_api.ThreadsApiError, "缺少明確核准"):
                threads_api.publish_text("草稿", "同意")
            load_token.assert_not_called()

    def test_validate_text_enforces_threads_limit(self):
        self.assertEqual(threads_api.validate_text("草稿")["characters"], 2)
        with self.assertRaisesRegex(threads_api.ThreadsApiError, "500 字元上限"):
            threads_api.validate_text("字" * 501)

    def test_keychain_read_uses_only_fixed_native_helper(self):
        token = "sensitive-token-value"
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                threads_api.subprocess,
                "run",
                return_value=CompletedProcess([], 0, stdout=token.encode("utf-8")),
            ) as run,
        ):
            self.assertEqual(threads_api.load_access_token(), token)

        command = run.call_args.args[0]
        self.assertEqual(command[0], str(threads_api.KEYCHAIN_UPDATE_HELPER))
        self.assertNotIn("/usr/bin/security", command)

    def test_keychain_read_does_not_fallback_to_security(self):
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                threads_api.subprocess,
                "run",
                return_value=CompletedProcess([], 1, stdout=b""),
            ) as run,
        ):
            with self.assertRaisesRegex(threads_api.ThreadsApiError, "找不到"):
                threads_api.load_access_token()

        self.assertEqual(run.call_count, 1)
        self.assertNotIn("/usr/bin/security", run.call_args.args[0])

    def test_publish_checks_account_and_uses_two_step_api(self):
        responses = [
            {"id": "user-1", "username": "lin.yusei"},
            {"id": "container-1"},
            {"id": "post-1"},
        ]
        with (
            patch.object(threads_api, "load_access_token", return_value="secret"),
            patch.object(threads_api, "graph_request", side_effect=responses) as request,
        ):
            result = threads_api.publish_text("已核准草稿", "發送")

        self.assertEqual(result["id"], "post-1")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[1].args[0], "me/threads")
        self.assertEqual(request.call_args_list[2].args[0], "me/threads_publish")

    def test_refresh_verifies_then_rotates_keychain_token(self):
        with (
            patch.object(threads_api, "load_access_token", return_value="old-token"),
            patch.object(
                threads_api,
                "_request_refreshed_token",
                return_value={"access_token": "new-token", "expires_in": 5184000},
            ),
            patch.object(
                threads_api,
                "graph_request",
                return_value={"id": "user-1", "username": "lin.yusei"},
            ),
            patch.object(threads_api, "save_access_token") as save_token,
            patch.object(threads_api, "_write_refresh_state") as write_state,
        ):
            result = threads_api.refresh_access_token()

        save_token.assert_called_once_with("new-token")
        self.assertEqual(result["expires_in_seconds"], 5184000)
        self.assertTrue(result["token_rotated"])
        write_state.assert_called_once()

    def test_refresh_keeps_old_token_if_account_check_fails(self):
        with (
            patch.object(threads_api, "load_access_token", return_value="old-token"),
            patch.object(
                threads_api,
                "_request_refreshed_token",
                return_value={"access_token": "new-token"},
            ),
            patch.object(
                threads_api,
                "graph_request",
                return_value={"id": "user-2", "username": "wrong.account"},
            ),
            patch.object(threads_api, "save_access_token") as save_token,
        ):
            with self.assertRaisesRegex(threads_api.ThreadsApiError, "原權杖已保留"):
                threads_api.refresh_access_token()
        save_token.assert_not_called()

    def test_keychain_update_uses_stdin_and_round_trip_verification(self):
        token = "sensitive-token-value"
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(
                threads_api.subprocess,
                "run",
                return_value=CompletedProcess([], 0),
            ) as run,
            patch.object(threads_api, "load_access_token", return_value=token),
        ):
            threads_api.save_access_token(token)

        command = run.call_args.args[0]
        self.assertNotIn(token, command)
        self.assertEqual(run.call_args.kwargs["input"], token.encode("utf-8"))

    def test_refresh_failure_is_written_to_status_file(self):
        with (
            patch.object(sys, "argv", ["threads_api.py", "refresh"]),
            patch.object(
                threads_api,
                "refresh_access_token",
                side_effect=threads_api.ThreadsApiError("invalid token"),
            ),
            patch.object(threads_api, "_write_refresh_state") as write_state,
            redirect_stderr(StringIO()),
        ):
            self.assertEqual(threads_api.main(), 1)

        state = write_state.call_args.args[0]
        self.assertFalse(state["ok"])
        self.assertEqual(state["error"], "invalid token")
        self.assertIn("failed_at", state)


if __name__ == "__main__":
    unittest.main()
