"""The reward has to land on the turns that earned it, or not land at all."""

from __future__ import annotations

import unittest
from typing import Any

from autorl.reward import Block, blocks_of, episode_reward, step_completions, true_entities


def call(step: int, name: str, **arguments: Any) -> dict[str, Any]:
    import json

    return {
        "type": "tool/call",
        "data": {"step": step, "name": name, "arguments": json.dumps(arguments)},
    }


def message(step: int) -> dict[str, Any]:
    return {"type": "assistant/message", "data": {"step": step}}


def completion(ordinal: int, purpose: str = "agent") -> dict[str, Any]:
    return {"ordinal": ordinal, "responseId": f"chatcmpl-{ordinal}", "purpose": purpose}


class Node:
    def __init__(self, subject: str) -> None:
        self.subject = subject


class Truth:
    """The parts of an `fpg.Scenario` the reward reads."""

    def __init__(self, subjects: list[str]) -> None:
        self.graph = type("G", (), {"nodes": [Node(s) for s in subjects]})()


TRUTH = Truth(["svc:geo", "svc:profile", "svc:frontend"])

# Two blocks: three queries then a note, two queries then a note.
EVENTS = [
    message(1),
    call(1, "sql", statement="SELECT * FROM abnormal_traces WHERE service_name = 'geo'"),
    message(2),
    call(2, "sql", statement="SELECT * FROM abnormal_logs"),
    message(3),
    call(3, "sql", statement="SELECT 1 WHERE service_name = 'profile'"),
    message(4),
    call(4, "take_note", content="geo is down"),
    message(5),
    call(5, "sql", statement="SELECT * FROM abnormal_metrics WHERE service_name = 'frontend'"),
    message(6),
    call(6, "sql", statement="SELECT * FROM normal_metrics WHERE service_name = 'frontend'"),
    message(7),
    call(7, "take_note", content="frontend degraded"),
]
COMPLETIONS = [completion(i) for i in range(7)]


class BlockTest(unittest.TestCase):
    def test_a_note_closes_a_block(self) -> None:
        blocks = blocks_of(EVENTS)
        self.assertEqual([b.steps for b in blocks], [(1, 2, 3), (5, 6)])
        self.assertEqual([b.closing_step for b in blocks], [4, 7])

    def test_a_trailing_run_without_a_note_is_still_a_block(self) -> None:
        blocks = blocks_of([*EVENTS, message(8), call(8, "sql", statement="SELECT 2")])
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[-1].closing_step, 8)

    def test_the_hit_rate_counts_queries_that_name_a_true_entity(self) -> None:
        entities = true_entities(TRUTH)
        self.assertEqual(entities, frozenset({"geo", "profile", "frontend"}))
        first, second = blocks_of(EVENTS)
        self.assertAlmostEqual(first.hit_rate(entities), 2 / 3)
        self.assertEqual(second.hit_rate(entities), 1.0)

    def test_a_query_that_only_prints_an_entity_does_not_count(self) -> None:
        """Naming a service in the output is not interrogating it."""
        block = Block(
            steps=(1,), statements=("SELECT service_name FROM abnormal_traces",), closing_step=1
        )
        self.assertEqual(block.hit_rate(true_entities(TRUTH)), 0.0)


class MappingTest(unittest.TestCase):
    def test_agent_turns_map_to_agent_completions_in_order(self) -> None:
        mapping = step_completions(EVENTS, COMPLETIONS)
        self.assertEqual(mapping[1], "chatcmpl-0")
        self.assertEqual(mapping[7], "chatcmpl-6")

    def test_a_compaction_completion_is_not_an_agent_turn(self) -> None:
        with_compaction = [*COMPLETIONS, completion(7, purpose="compaction")]
        self.assertEqual(
            step_completions(EVENTS, with_compaction), step_completions(EVENTS, COMPLETIONS)
        )

    def test_a_count_mismatch_refuses_to_guess(self) -> None:
        """Off by one is worse than nothing: it credits the wrong turn."""
        self.assertEqual(step_completions(EVENTS, COMPLETIONS[:-1]), {})


class EpisodeRewardTest(unittest.TestCase):
    def reward(self, **overrides: Any) -> Any:
        kwargs: dict[str, Any] = {
            "events": EVENTS,
            "completions": COMPLETIONS,
            "submission": None,
            "truth": TRUTH,
            "shaping": 0.2,
        }
        kwargs.update(overrides)
        return episode_reward(**kwargs)

    def test_block_reward_lands_on_the_turn_that_closed_the_block(self) -> None:
        episode = self.reward()
        self.assertAlmostEqual(episode.shaped["chatcmpl-3"], 0.2 * (2 / 3))  # step 4
        self.assertAlmostEqual(episode.shaped["chatcmpl-6"], 0.2 * 1.0)  # step 7

    def test_no_submission_scores_the_outcome_zero(self) -> None:
        episode = self.reward()
        self.assertEqual(episode.outcome, 0.0)
        self.assertEqual(episode.blocks, 2)

    def test_without_the_sidecar_the_outcome_still_lands_somewhere(self) -> None:
        episode = self.reward(completions=[])
        self.assertEqual(episode.shaped, {})
        self.assertEqual(episode.unmapped, 0)

    def test_a_broken_mapping_degrades_to_the_last_turn(self) -> None:
        episode = self.reward(completions=COMPLETIONS[:-1])
        self.assertEqual(list(episode.shaped), ["chatcmpl-5"])
        self.assertEqual(episode.unmapped, len(COMPLETIONS) - 1)

    def test_compaction_is_never_rewarded(self) -> None:
        episode = self.reward(completions=[*COMPLETIONS, completion(7, purpose="compaction")])
        self.assertNotIn("chatcmpl-7", episode.shaped)


if __name__ == "__main__":
    unittest.main()
