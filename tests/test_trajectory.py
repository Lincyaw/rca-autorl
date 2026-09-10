"""A session log projects to the episode a reviewer reads."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from autorl.trajectory import case_of, load_episode

SEQ = 0


def ev(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    global SEQ
    SEQ += 1
    return {"seq": SEQ, "type": kind, "data": data}


def step(index: int, call_id: str, *, tokens: int, tool: str = "sql") -> list[dict[str, Any]]:
    """One whole agent step: start, the assistant turn, the call, its result."""
    return [
        ev("step/start", {"step": index, "turn": 1}),
        ev(
            "assistant/message",
            {
                "step": index,
                "usage": {"inputTokens": tokens, "outputTokens": 40},
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": f"think {index}"},
                        {"type": "text", "text": f"say {index}"},
                    ],
                },
            },
        ),
        ev("tool/call", {"step": index, "callId": call_id, "name": tool, "arguments": "{}"}),
        ev(
            "tool/result",
            {
                "step": index,
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": call_id,
                            "content": [{"type": "text", "text": f"rows {index}"}],
                        }
                    ],
                },
            },
        ),
    ]


def write(events: list[dict[str, Any]], case: str = "batch-01KQABC") -> Path:
    """The events on disk under the directory layout `find_sessions` walks."""
    root = Path(tempfile.mkdtemp())
    home = root / "sessions" / f"--home-fay-cases-{case}--" / "rca-abc123"
    home.mkdir(parents=True)
    with (home / "session.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return home / "session.jsonl"


class TrajectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        global SEQ
        SEQ = 0

    def test_steps_carry_reasoning_call_and_context_size(self) -> None:
        path = write([*step(1, "c1", tokens=2200), *step(2, "c2", tokens=5100)])
        episode = load_episode(path)

        self.assertEqual([s.index for s in episode.steps], [1, 2])
        self.assertEqual(episode.steps[0].reasoning, "think 1")
        self.assertEqual(episode.steps[0].text, "say 1")
        self.assertEqual(episode.steps[0].tool, "sql")
        self.assertEqual(episode.steps[0].result_head, "rows 1")
        self.assertFalse(episode.steps[0].result_error)
        # The curve the window bounds: the peak is what a review compares to it.
        self.assertEqual([s.input_tokens for s in episode.steps], [2200, 5100])
        self.assertEqual(episode.peak_input_tokens, 5100)
        self.assertEqual(episode.tool_counts, {"sql": 2})

    def test_a_step_is_keyed_by_its_number_not_arrival_order(self) -> None:
        """A compaction pass writes events between a step's start and its message."""
        events = [
            ev("step/start", {"step": 1}),
            ev("compaction/start", {}),
            ev("compaction/end", {"error": "400 status code (no body)"}),
            ev(
                "assistant/message",
                {
                    "step": 1,
                    "usage": {"inputTokens": 24222},
                    "message": {"role": "assistant", "content": []},
                },
            ),
        ]
        episode = load_episode(write(events))
        self.assertEqual(len(episode.steps), 1)
        self.assertEqual(episode.steps[0].input_tokens, 24222)

    def test_a_failed_compaction_keeps_its_error_and_lands_nothing(self) -> None:
        events = [
            *step(1, "c1", tokens=19700),
            ev("compaction/start", {}),
            ev("compaction/end", {"error": "RCA checkpoint produced no text content"}),
            ev("compaction/start", {}),
            ev(
                "compaction/summary",
                {
                    "compactionId": "k1",
                    "shadowedTokenCount": 14300,
                    "summary": [{"type": "text", "text": "## Incident\n- x"}],
                },
            ),
            ev("compaction/end", {}),
        ]
        episode = load_episode(write(events))

        self.assertEqual(len(episode.compactions), 2)
        failed, landed = episode.compactions
        self.assertFalse(failed.landed)
        self.assertIn("no text content", failed.error)
        # The pass is attributed to the step it interrupted, which is what makes
        # it plottable against the context curve.
        self.assertEqual(failed.at_step, 1)
        self.assertTrue(landed.landed)
        self.assertEqual(landed.shadowed_tokens, 14300)
        self.assertIn("## Incident", landed.summary)

    def test_an_error_result_is_marked_so_a_retry_is_visible(self) -> None:
        events = [
            ev("step/start", {"step": 1}),
            ev("tool/call", {"step": 1, "callId": "c1", "name": "submit_result", "arguments": "{}"}),
            ev(
                "tool/result",
                {
                    "step": 1,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "c1",
                                "isError": True,
                                "content": [
                                    {"type": "text", "text": 'Error: missing property "edges"'}
                                ],
                            }
                        ],
                    },
                },
            ),
        ]
        episode = load_episode(write(events))
        self.assertTrue(episode.steps[0].result_error)
        self.assertIn("missing property", episode.steps[0].result_head)
        # A rejected submission is not a submission.
        self.assertIsNone(episode.submission)

    def test_finish_reason_reads_a_dict_and_a_repr_alike(self) -> None:
        as_dict = load_episode(
            write(
                [
                    *step(1, "c1", tokens=100),
                    ev(
                        "turn/end",
                        {"reason": {"kind": "error", "error": {"message": "400 status code"}}},
                    ),
                ]
            )
        )
        self.assertEqual(as_dict.finish, "error")
        self.assertIn("400", as_dict.error)

        # Some harness versions write the reason already stringified.
        as_text = load_episode(
            write(
                [
                    *step(1, "c1", tokens=100),
                    ev("turn/end", {"reason": "{'kind': 'error', 'error': {'message': '500'}}"}),
                ]
            )
        )
        self.assertEqual(as_text.finish, "error")

        completed = load_episode(
            write([*step(1, "c1", tokens=100), ev("turn/end", {"reason": {"kind": "completed"}})])
        )
        self.assertEqual(completed.finish, "completed")
        self.assertEqual(completed.error, "")

    def test_a_half_written_line_does_not_lose_the_episode(self) -> None:
        """A run still in flight leaves a truncated last line."""
        path = write(step(1, "c1", tokens=100))
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"seq": 99, "type": "assistant/mess')
        self.assertEqual(len(load_episode(path).steps), 1)

    def test_an_episode_that_never_submitted_still_carries_the_truth(self) -> None:
        """"Found none of these" is the finding; no axes at all would hide it."""
        path = write(step(1, "c1", tokens=100))
        truth = mock.Mock()
        truth.graph.nodes = [mock.Mock(id="n1", subject="svc-a")]
        truth.graph.edges = []
        truth.graph.root_causes = [mock.Mock(id="n1")]
        with mock.patch("autorl.trajectory.truth_for_case", return_value=truth):
            episode = load_episode(path, case_dir=Path("/nonexistent"))

        self.assertIsNone(episode.submission)
        self.assertEqual(episode.score, 0.0)
        self.assertIsNotNone(episode.answer)
        assert episode.answer is not None
        self.assertEqual(episode.answer.truth["subjects"], frozenset({"svc-a"}))
        self.assertEqual(episode.answer.found["subjects"], frozenset())

    def test_case_comes_from_the_mangled_snapshot_directory(self) -> None:
        """The case id itself contains a `-`, so the name cannot just be split on it."""
        path = write([], case="batch-01KQHDBB5Y8K69Z13G7VSWG9FS")
        self.assertEqual(case_of(path), "batch-01KQHDBB5Y8K69Z13G7VSWG9FS")
        self.assertEqual(load_episode(path).session_id, "rca-abc123")


if __name__ == "__main__":
    unittest.main()
