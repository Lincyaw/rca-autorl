"""`submitted_result` reads the submission the tool accepted, not the first one.

`submit_result` is terminal, but only once `execute` runs: a call the argument
schema rejects never reaches it, so the turn does not end, the guard is not
armed, and the model retries inside the same turn. Both attempts leave a
`tool/call` event — that event is written before execution — and the rejected
one comes first. Three of the first ten collected episodes took that path.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from autorl.agent import submitted_result  # noqa: E402


def _call(call_id: str, arguments: str) -> dict[str, object]:
    return {
        "type": "tool/call",
        "data": {"name": "submit_result", "callId": call_id, "arguments": arguments},
    }


def _result(call_id: str, *, is_error: bool) -> dict[str, object]:
    block: dict[str, object] = {"type": "tool-result", "toolCallId": call_id}
    if is_error:
        block["isError"] = True
    return {"type": "tool/result", "data": {"message": {"content": [block]}}}


REJECTED = '{"nodes": [{"id": "n1"}]}'
ACCEPTED = '{"nodes": [{"id": "n1"}], "edges": [], "root_causes": ["n1"]}'


class SubmittedResultTests(unittest.TestCase):
    def _run(self, events: list[dict[str, object]]) -> dict[str, object] | None:
        return submitted_result(types.SimpleNamespace(events=events))

    def test_rejected_attempt_is_skipped_for_the_retry(self) -> None:
        got = self._run(
            [
                _call("c1", REJECTED),
                _result("c1", is_error=True),
                _call("c2", ACCEPTED),
                _result("c2", is_error=False),
            ]
        )
        assert got is not None
        self.assertEqual(got["root_causes"], ["n1"])

    def test_single_accepted_call_is_returned(self) -> None:
        got = self._run([_call("c1", ACCEPTED), _result("c1", is_error=False)])
        assert got is not None
        self.assertIn("edges", got)

    def test_only_rejected_calls_read_as_no_submission(self) -> None:
        """An episode that never landed a valid submission must not look like one."""
        self.assertIsNone(self._run([_call("c1", REJECTED), _result("c1", is_error=True)]))

    def test_missing_result_reads_as_no_submission(self) -> None:
        """A call cut off before its result never executed."""
        self.assertIsNone(self._run([_call("c1", ACCEPTED)]))


if __name__ == "__main__":
    unittest.main()
