"""The tool and the verifier must accept the same submissions.

`agent/rca-harness/src/submit-result.js` gates what becomes a training example;
`autorl.fpg.parse_submission` gates what the verifier can read back. They are
written in different languages against the same profile, which is exactly the
arrangement that drifted before. These fixtures run through both.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from pydantic import ValidationError

from autorl.fpg import VOCABULARY_JS, entity_ref_pattern, parse_submission, schema

CONTRACT_JS = VOCABULARY_JS.parent / "contract.js"

WINDOW = {"start": "2025-07-21T14:47:09+01:00", "end": "2025-07-21T14:49:04+01:00"}
EVIDENCE = [
    {
        "query": {"language": "sql", "statement": "SELECT count(*) FROM abnormal_logs"},
        "explanation": "1 042 error lines in the abnormal window against 0 in the normal one",
    }
]


def node(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "n1",
        "subject": "svc:ts-auth-service",
        "predicate": "process_killed",
        "time": WINDOW,
        "evidence": EVIDENCE,
    }
    return {**base, **overrides}


def submission(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "nodes": [node()],
        "edges": [],
        "root_causes": ["n1"],
    }
    return {**base, **overrides}


# (name, payload, accepted) — every case must get the same verdict on both sides.
CASES: list[tuple[str, dict[str, object], bool]] = [
    ("minimal", submission(), True),
    (
        "chain",
        submission(
            nodes=[
                node(),
                node(id="n2", subject="svc:ts-ui-dashboard", predicate="latency_degraded"),
            ],
            edges=[{"src": "n1", "dst": "n2"}],
        ),
        True,
    ),
    ("hypothesis without evidence", submission(nodes=[node(evidence=[], hypothesis=True)]), True),
    (
        "instant window",
        submission(nodes=[node(time={"start": WINDOW["start"], "end": WINDOW["start"]})]),
        True,
    ),
    (
        "utc offset",
        submission(
            nodes=[node(time={"start": "2025-07-21T13:47:09Z", "end": "2025-07-21T13:49:04Z"})]
        ),
        True,
    ),
    # The four shapes every one of the first ten collected episodes emitted.
    ("bare service name", submission(nodes=[node(subject="ts-auth-service")]), False),
    (
        "subject carrying a component",
        submission(nodes=[node(subject="svc:ts-auth-service jvm")]),
        False,
    ),
    ("prose predicate", submission(nodes=[node(predicate="container is killed at 14:47")]), False),
    (
        "truncated offset",
        submission(
            nodes=[node(time={"start": "2025-07-21T14:47:09+01", "end": "2025-07-21T14:49:04+01"})]
        ),
        False,
    ),
    # Structural rules the old tool did not enforce.
    (
        "entity kind the profile does not declare",
        submission(nodes=[node(subject="pod:ts-auth-service-77d85c69dd-rk2h5")]),
        False,
    ),
    ("evidence missing", submission(nodes=[node(evidence=[])]), False),
    (
        "duplicate node id",
        submission(nodes=[node(), node(subject="svc:mysql")], root_causes=["n1"]),
        False,
    ),
    ("self loop", submission(edges=[{"src": "n1", "dst": "n1"}]), False),
    ("dangling edge", submission(edges=[{"src": "n1", "dst": "n9"}]), False),
    ("dangling root cause", submission(root_causes=["n9"]), False),
    ("no nodes", submission(nodes=[], root_causes=[]), False),
    ("no root causes", submission(root_causes=[]), False),
    (
        "unknown query language",
        submission(
            nodes=[
                node(
                    evidence=[
                        {"query": {"language": "duckdb", "statement": "x"}, "explanation": "y"}
                    ]
                )
            ]
        ),
        False,
    ),
    (
        "reversed window",
        submission(nodes=[node(time={"start": WINDOW["end"], "end": WINDOW["start"]})]),
        False,
    ),
]

# The JSON Schema subset the tool's parameters compile to enforces types,
# required keys, and the two `enum`s before `execute` runs; `validate` covers
# the rest. The Node side below drives `validate` directly, so the cases the
# registry would have rejected are the ones it cannot see.
SCHEMA_ENFORCED = {"prose predicate", "unknown query language"}

NODE_HARNESS = """
import { validate } from %s
const cases = JSON.parse(process.env.FIXTURES)
const verdicts = cases.map(payload => {
  try {
    validate(payload)
    return { accepted: true }
  } catch (error) {
    return { accepted: false, reason: String(error.message) }
  }
})
process.stdout.write(JSON.stringify(verdicts))
"""


class SubmitResultContractTest(unittest.TestCase):
    def test_the_verifier_agrees_with_the_fixtures(self) -> None:
        for name, payload, accepted in CASES:
            with self.subTest(name):
                if accepted:
                    parse_submission(payload)
                    continue
                with self.assertRaises(ValidationError):
                    parse_submission(payload)

    def test_the_tool_agrees_with_the_verifier(self) -> None:
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not on PATH")
        driven = [case for case in CASES if case[0] not in SCHEMA_ENFORCED]
        script = NODE_HARNESS % json.dumps(str(CONTRACT_JS))
        completed = subprocess.run(
            [node_bin, "--input-type=module", "-e", script],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "FIXTURES": json.dumps([case[1] for case in driven])},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        verdicts = json.loads(completed.stdout)
        for (name, _payload, accepted), verdict in zip(driven, verdicts, strict=True):
            with self.subTest(name):
                self.assertEqual(verdict["accepted"], accepted, verdict.get("reason", ""))

    def test_the_generated_vocabulary_matches_the_profile(self) -> None:
        from autorl.fpg import render_vocabulary_js

        self.assertEqual(
            VOCABULARY_JS.read_text(encoding="utf-8"),
            render_vocabulary_js(),
            "agent/rca-harness/src/vocabulary.js is stale; run `python -m autorl.fpg`",
        )

    def test_the_two_sides_hold_one_entity_pattern(self) -> None:
        rendered = VOCABULARY_JS.read_text(encoding="utf-8")
        self.assertIn(json.dumps(entity_ref_pattern()), rendered)

    def test_the_profile_declares_the_testbed_it_names(self) -> None:
        profile = schema().profile
        self.assertEqual(profile.vocab_version, "core-0.1.0+microservices-0.5.0")
        self.assertIn("svc", profile.entity_types)


if __name__ == "__main__":
    unittest.main()
