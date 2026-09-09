"""A submission and its truth reduce to the same three sets of subjects."""

from __future__ import annotations

import unittest

from autorl.agent import _answer_of
from autorl.fpg import schema

WINDOW = {"start": "2026-05-01T17:19:07+08:00", "end": "2026-05-01T17:24:07+08:00"}
EVIDENCE = [
    {
        "query": {"language": "sql", "statement": "SELECT 1"},
        "explanation": "the abnormal window differs from the normal one",
    }
]


def node(node_id: str, subject: str, predicate: str = "latency_degraded") -> dict[str, object]:
    return {
        "id": node_id,
        "subject": subject,
        "predicate": predicate,
        "time": WINDOW,
        "evidence": EVIDENCE,
    }


def truth_node(
    node_id: str, subject: str, predicate: str = "latency_degraded"
) -> dict[str, object]:
    return {**node(node_id, subject, predicate), "kind": "event", "grounding": "observed"}


def truth_edge(src: str, dst: str) -> dict[str, object]:
    return {
        "src": src,
        "dst": dst,
        "mechanism": "sync_call_blocking",
        "verification": "consistency-checked",
    }


TRUTH = schema().Scenario.model_validate(
    {
        "schema_version": "0.1.0",
        "scenario_id": "test",
        "testbed": "hs",
        "vocab_version": schema().profile.vocab_version,
        "injections": [
            {
                "node_id": "geo",
                "fault_type": "PodFailure",
                "target_entity": "svc:geo",
                "time": WINDOW,
            }
        ],
        "graph": {
            "nodes": [
                truth_node("geo", "svc:geo", "process_killed"),
                truth_node("profile", "svc:profile"),
                truth_node("frontend", "svc:frontend"),
            ],
            "edges": [truth_edge("geo", "profile"), truth_edge("profile", "frontend")],
        },
    }
)


class AnswerAxesTest(unittest.TestCase):
    def test_truth_reduces_to_roots_subjects_and_subject_edges(self) -> None:
        answer = _answer_of(None, TRUTH)
        self.assertEqual(answer.truth["roots"], {"svc:geo"})
        self.assertEqual(answer.truth["subjects"], {"svc:geo", "svc:profile", "svc:frontend"})
        self.assertEqual(
            answer.truth["edges"], {"svc:geo->svc:profile", "svc:profile->svc:frontend"}
        )
        self.assertEqual(answer.found["edges"], frozenset())

    def test_an_edge_counts_by_its_endpoints_not_its_node_ids(self) -> None:
        submission = {
            "nodes": [node("a", "svc:profile"), node("b", "svc:frontend"), node("c", "svc:rate")],
            "edges": [{"src": "a", "dst": "b"}, {"src": "c", "dst": "b"}],
            "root_causes": ["c"],
        }
        answer = _answer_of(submission, TRUTH)
        self.assertEqual(answer.found["edges"], {"svc:profile->svc:frontend"})
        self.assertEqual(
            answer.claimed["edges"], {"svc:profile->svc:frontend", "svc:rate->svc:frontend"}
        )
        self.assertEqual(answer.found["roots"], frozenset())
        self.assertEqual(answer.claimed["roots"], {"svc:rate"})


if __name__ == "__main__":
    unittest.main()
