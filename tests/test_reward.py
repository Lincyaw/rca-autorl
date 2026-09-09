"""A value reaches a row by the completion id the session log carries."""

from __future__ import annotations

import unittest
from typing import Any

from autorl.reward import step_ids


def message(step: int, response_id: str | None) -> dict[str, Any]:
    source: dict[str, Any] = {"kind": "model", "provider": "gateway"}
    if response_id is not None:
        source["replayState"] = {"response": {"responseId": response_id}}
    return {"type": "assistant/message", "data": {"step": step, "message": {"source": source}}}


class StepIdsTest(unittest.TestCase):
    def test_each_step_maps_to_the_completion_that_produced_it(self) -> None:
        events = [message(1, "chatcmpl-a"), {"type": "tool/call"}, message(2, "chatcmpl-b")]
        self.assertEqual(step_ids(events), {1: "chatcmpl-a", 2: "chatcmpl-b"})

    def test_a_route_that_keeps_no_replay_state_maps_nothing(self) -> None:
        """`llm-deepseek` sessions carry no id; nothing is guessed for them."""
        self.assertEqual(step_ids([message(1, None)]), {})


if __name__ == "__main__":
    unittest.main()
