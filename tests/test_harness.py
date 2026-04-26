"""End-to-end tests for the LLM-as-harness P0 pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from autorl.observability.harness import (
    HarnessStore,
    Turn,
    TurnRole,
    detect_drift,
    parse_hook_payload,
    read_transcript_turns,
    summarize_turns,
    tick,
)
from autorl.observability.harness.__main__ import main as cli_main
from autorl.observability.harness.schema import DriftType, EventKind


class SummarizerRulesTest(unittest.TestCase):
    def test_first_user_turn_becomes_task(self) -> None:
        events = summarize_turns(
            [Turn(index=0, role=TurnRole.USER, content="find the root cause of API latency")],
            prior_events=[],
            next_event_id=0,
        )
        self.assertEqual(len(events), 1)
        self.assertIs(events[0].kind, EventKind.TASK)
        self.assertEqual(events[0].id, 0)

    def test_assistant_tool_call_emits_action(self) -> None:
        events = summarize_turns(
            [
                Turn(
                    index=1,
                    role=TurnRole.ASSISTANT,
                    tool_name="bash",
                    tool_args={"cmd": "kubectl get pods"},
                )
            ],
            prior_events=[],
            next_event_id=5,
        )
        self.assertEqual(len(events), 1)
        self.assertIs(events[0].kind, EventKind.ACTION)
        self.assertEqual(events[0].id, 5)
        self.assertIn("bash", events[0].summary)

    def test_assistant_text_classification(self) -> None:
        turns = [
            Turn(index=0, role=TurnRole.ASSISTANT, content="I think the root cause is service A"),
            Turn(index=1, role=TurnRole.ASSISTANT, content="因此根因是 service B 的网络问题"),
            Turn(index=2, role=TurnRole.ASSISTANT, content="actually, let me reconsider"),
            Turn(index=3, role=TurnRole.ASSISTANT, content="Let me check the logs."),
        ]
        events = summarize_turns(turns, prior_events=[], next_event_id=0)
        kinds = [e.kind for e in events]
        self.assertEqual(
            kinds,
            [EventKind.HYPOTHESIS, EventKind.CONCLUSION, EventKind.REFLECTION],
            "plain narration without markers should be skipped",
        )


class DetectorTest(unittest.TestCase):
    def test_silent_when_no_pattern(self) -> None:
        events = summarize_turns(
            [
                Turn(index=0, role=TurnRole.USER, content="diagnose pod restart loop"),
                Turn(index=1, role=TurnRole.ASSISTANT, tool_name="kubectl", tool_args={"a": "1"}),
                Turn(index=2, role=TurnRole.TOOL, tool_name="kubectl", content="restart count: 5"),
            ],
            prior_events=[],
            next_event_id=0,
        )
        self.assertFalse(detect_drift(events).drift)

    def test_stuck_loop_detected(self) -> None:
        turns = [Turn(index=0, role=TurnRole.USER, content="diagnose service-a 502 errors")]
        for i in range(4):
            turns.append(
                Turn(
                    index=1 + i,
                    role=TurnRole.ASSISTANT,
                    tool_name="kubectl",
                    tool_args={"cmd": "logs service-a"},
                )
            )
        events = summarize_turns(turns, prior_events=[], next_event_id=0)
        verdict = detect_drift(events)
        self.assertTrue(verdict.drift)
        self.assertIs(verdict.type, DriftType.STUCK_LOOP)

    def test_premature_conclusion_detected(self) -> None:
        events = summarize_turns(
            [
                Turn(index=0, role=TurnRole.USER, content="root cause of order pipeline failure"),
                Turn(index=1, role=TurnRole.ASSISTANT, content="所以根因是 db 连接池耗尽"),
            ],
            prior_events=[],
            next_event_id=0,
        )
        verdict = detect_drift(events)
        self.assertTrue(verdict.drift)
        self.assertIs(verdict.type, DriftType.PREMATURE_CONCLUSION)


class StoreAndWorkerTest(unittest.TestCase):
    def test_tick_is_incremental_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessStore(tmp)
            sid = "session-x"
            store.append_inbox(sid, [Turn(index=0, role=TurnRole.USER, content="diagnose X")])
            r1 = tick(store, sid)
            self.assertEqual(r1.new_event_count, 1)

            r_again = tick(store, sid)
            self.assertEqual(r_again.new_event_count, 0, "second tick with no new turns is a no-op")

            store.append_inbox(
                sid,
                [
                    Turn(
                        index=1,
                        role=TurnRole.ASSISTANT,
                        tool_name="bash",
                        tool_args={"cmd": "ls"},
                    )
                ],
            )
            r2 = tick(store, sid)
            self.assertEqual(r2.new_event_count, 1)
            events = store.read_events(sid)
            self.assertEqual([e.id for e in events], [0, 1], "event ids stay contiguous")

    def test_stuck_loop_writes_pending_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessStore(tmp)
            sid = "loop-session"
            turns = [Turn(index=0, role=TurnRole.USER, content="diagnose service-a 502 errors")]
            for i in range(4):
                turns.append(
                    Turn(
                        index=1 + i,
                        role=TurnRole.ASSISTANT,
                        tool_name="kubectl",
                        tool_args={"cmd": "logs service-a"},
                    )
                )
            store.append_inbox(sid, turns)

            result = tick(store, sid)
            self.assertTrue(result.reminder_written)

            popped = store.pop_reminder(sid)
            self.assertIsNotNone(popped)
            assert popped is not None
            self.assertIs(popped.type, DriftType.STUCK_LOOP)
            self.assertFalse(store.reminder_path(sid).exists())


class CliTest(unittest.TestCase):
    def test_full_loop_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sid = "cli-session"
            payload_path = Path(tmp) / "delta.json"
            turns_payload = {
                "turns": [
                    {"index": 0, "role": "user", "content": "diagnose service-a 502 errors"},
                    *[
                        {
                            "index": i + 1,
                            "role": "assistant",
                            "tool_name": "kubectl",
                            "tool_args": {"cmd": "logs service-a"},
                        }
                        for i in range(4)
                    ],
                ]
            }
            payload_path.write_text(json.dumps(turns_payload), encoding="utf-8")

            self.assertEqual(
                cli_main(
                    [
                        "--root",
                        tmp,
                        "ingest",
                        "--session",
                        sid,
                        "--input",
                        str(payload_path),
                    ]
                ),
                0,
            )
            self.assertEqual(
                cli_main(["--root", tmp, "tick", "--session", sid]),
                0,
            )

            reminder_file = Path(tmp) / "pending_reminders" / f"{sid}.json"
            self.assertTrue(reminder_file.exists())

            self.assertEqual(
                cli_main(["--root", tmp, "inject", "--session", sid]),
                0,
            )
            self.assertFalse(reminder_file.exists(), "inject must consume the reminder")


class HookPayloadTest(unittest.TestCase):
    def test_parse_happy_path(self) -> None:
        payload = parse_hook_payload(
            json.dumps(
                {
                    "session_id": "sid-abc",
                    "transcript_path": "/tmp/transcript.jsonl",
                    "hook_event_name": "PostToolUse",
                    "tool_name": "Bash",
                }
            )
        )
        assert payload is not None
        self.assertEqual(payload.session_id, "sid-abc")
        self.assertEqual(payload.transcript_path, "/tmp/transcript.jsonl")
        self.assertEqual(payload.hook_event_name, "PostToolUse")

    def test_parse_returns_none_for_garbage(self) -> None:
        for bad in ["", "   ", "not-json", "[]", json.dumps({"foo": "bar"})]:
            self.assertIsNone(parse_hook_payload(bad), bad)


class TranscriptAdapterTest(unittest.TestCase):
    @staticmethod
    def _write_transcript(path: Path, messages: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in messages) + "\n",
            encoding="utf-8",
        )

    def test_decodes_text_tooluse_toolresult_skips_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp) / "transcript.jsonl"
            self._write_transcript(
                tp,
                [
                    {"type": "permission-mode", "permissionMode": "default"},
                    {"type": "user", "message": {"role": "user", "content": "diagnose foo"}},
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "text": "should not appear"},
                                {"type": "text", "text": "Let me check the logs."},
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": "kubectl logs"},
                                },
                            ],
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "content": [{"type": "text", "text": "no output"}],
                                }
                            ],
                        },
                    },
                ],
            )
            turns = read_transcript_turns(tp)
            roles = [(t.role, t.tool_name, t.content[:30]) for t in turns]
            self.assertEqual(
                roles,
                [
                    (TurnRole.USER, None, "diagnose foo"),
                    (TurnRole.ASSISTANT, None, "Let me check the logs."),
                    (TurnRole.ASSISTANT, "Bash", ""),
                    (TurnRole.TOOL, None, "no output"),
                ],
            )
            # Indices are sequential
            self.assertEqual([t.index for t in turns], [0, 1, 2, 3])

    def test_returns_empty_for_missing_file(self) -> None:
        self.assertEqual(read_transcript_turns("/no/such/file.jsonl"), [])


class RateLimitTest(unittest.TestCase):
    def test_second_drift_is_suppressed_within_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessStore(tmp)
            sid = "rl-session"
            turns: list[Turn] = [
                Turn(index=0, role=TurnRole.USER, content="diagnose service-a 502 errors")
            ]
            for i in range(4):
                turns.append(
                    Turn(
                        index=1 + i,
                        role=TurnRole.ASSISTANT,
                        tool_name="kubectl",
                        tool_args={"cmd": "logs service-a"},
                    )
                )
            store.append_inbox(sid, turns)
            r1 = tick(store, sid, min_reminder_gap=5)
            self.assertTrue(r1.reminder_written)

            # Drain the reminder so the pending-file check no longer blocks.
            store.pop_reminder(sid)

            # Add 2 more identical actions — gap < 5, should suppress.
            store.append_inbox(
                sid,
                [
                    Turn(
                        index=5 + i,
                        role=TurnRole.ASSISTANT,
                        tool_name="kubectl",
                        tool_args={"cmd": "logs service-a"},
                    )
                    for i in range(2)
                ],
            )
            r2 = tick(store, sid, min_reminder_gap=5)
            self.assertFalse(r2.reminder_written)
            self.assertTrue(r2.suppressed_by_rate_limit)
            self.assertTrue(r2.verdict.drift)

            # Add enough turns to exceed the gap; reminder may fire again.
            store.append_inbox(
                sid,
                [
                    Turn(
                        index=7 + i,
                        role=TurnRole.ASSISTANT,
                        tool_name="kubectl",
                        tool_args={"cmd": "logs service-a"},
                    )
                    for i in range(4)
                ],
            )
            r3 = tick(store, sid, min_reminder_gap=5)
            self.assertTrue(r3.reminder_written)


class CliFromHookTest(unittest.TestCase):
    def _make_transcript(self, path: Path) -> None:
        msgs: list[dict] = [
            {"type": "user", "message": {"role": "user", "content": "diagnose service-a 502 errors"}},
        ]
        for _ in range(4):
            msgs.append(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "kubectl",
                                "input": {"cmd": "logs service-a"},
                            }
                        ],
                    },
                }
            )
        path.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs) + "\n",
            encoding="utf-8",
        )

    def test_from_hook_end_to_end(self) -> None:
        import io

        with tempfile.TemporaryDirectory() as tmp:
            sid = "from-hook-session"
            transcript = Path(tmp) / "transcript.jsonl"
            self._make_transcript(transcript)
            payload = json.dumps(
                {
                    "session_id": sid,
                    "transcript_path": str(transcript),
                    "hook_event_name": "PostToolUse",
                }
            )
            saved_stdin = sys.stdin
            sys.stdin = io.StringIO(payload)
            try:
                self.assertEqual(
                    cli_main(["--root", tmp, "ingest", "--from-hook"]),
                    0,
                )
            finally:
                sys.stdin = saved_stdin

            self.assertEqual(
                cli_main(["--root", tmp, "tick", "--session", sid]),
                0,
            )
            reminder_file = Path(tmp) / "pending_reminders" / f"{sid}.json"
            self.assertTrue(reminder_file.exists())

            # inject --from-hook should consume it
            sys.stdin = io.StringIO(payload)
            try:
                self.assertEqual(
                    cli_main(["--root", tmp, "inject", "--from-hook"]),
                    0,
                )
            finally:
                sys.stdin = saved_stdin
            self.assertFalse(reminder_file.exists())

    def test_from_hook_silent_on_garbage(self) -> None:
        import io

        with tempfile.TemporaryDirectory() as tmp:
            saved_stdin = sys.stdin
            sys.stdin = io.StringIO("not json at all")
            try:
                self.assertEqual(
                    cli_main(["--root", tmp, "ingest", "--from-hook"]),
                    0,
                    "garbage stdin must not raise",
                )
            finally:
                sys.stdin = saved_stdin


if __name__ == "__main__":
    unittest.main()
