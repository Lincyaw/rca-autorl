"""The reward has to land on the turns that earned it, or not land at all."""

from __future__ import annotations

import unittest
from typing import Any

from autorl.reward import episode_reward, step_completions


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

# What real logs do: one response holding a note and then the next block's
# queries, a submit turn holding no query, and a compaction request in between.
SHARED_STEP_EVENTS = [
    message(1),
    call(1, "sql", statement="SELECT 1 WHERE service_name = 'geo'"),
    message(2),
    call(2, "take_note", content="geo"),
    call(2, "sql", statement="SELECT 2 FROM abnormal_logs"),
    message(3),
    call(3, "take_note", content="nothing"),
    message(4),
    call(4, "submit_result", nodes=[]),
]
SHARED_STEP_COMPLETIONS = [
    completion(0),
    completion(1),
    completion(2, purpose="compaction"),
    completion(3),
    completion(4),
]


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
        }
        kwargs.update(overrides)
        return episode_reward(**kwargs)

    @staticmethod
    def accumulate(shaped: dict[str, float], discount: float = 1.0) -> list[float]:
        """What AReaL makes of the own-rewards: `value[i] = own[i] + value[i+1] * discount`."""
        values: list[float] = []
        running = 0.0
        for own in reversed(list(shaped.values())):
            running = own + running * discount
            values.append(running)
        return list(reversed(values))

    def test_every_turn_carries_the_episode_outcome(self) -> None:
        """There is no turn-level credit; the value is the same everywhere."""
        episode = self.reward()
        for value in self.accumulate(episode.shaped):
            self.assertAlmostEqual(value, episode.outcome)

    def test_no_submission_scores_the_outcome_zero(self) -> None:
        episode = self.reward()
        self.assertEqual(episode.outcome, 0.0)
        for value in self.accumulate(episode.shaped):
            self.assertEqual(value, 0.0)

    def test_without_the_sidecar_the_outcome_still_lands_somewhere(self) -> None:
        episode = self.reward(completions=[])
        self.assertEqual(episode.shaped, {})
        self.assertEqual(episode.unmapped, 0)

    def test_a_broken_mapping_degrades_to_the_last_agent_turn(self) -> None:
        """Not simply the last completion: that can be the compaction summarizer."""
        broken = [*COMPLETIONS[:-1], completion(6, purpose="compaction")]
        episode = self.reward(completions=broken)
        self.assertEqual(list(episode.shaped), ["chatcmpl-5"])
        self.assertEqual(episode.unmapped, len(broken))

    def test_a_compaction_row_carries_the_same_value_as_the_turns(self) -> None:
        """It is exported and trained on, so it cannot be left to inherit one."""
        episode = episode_reward(
            events=SHARED_STEP_EVENTS,
            completions=SHARED_STEP_COMPLETIONS,
            submission=None,
            truth=TRUTH,
        )
        for value in self.accumulate(episode.shaped):
            self.assertAlmostEqual(value, episode.outcome)

    def test_the_discount_the_caller_declares_is_the_one_inverted(self) -> None:
        for discount in (1.0, 0.9):
            with self.subTest(discount=discount):
                episode = self.reward(turn_discount=discount)
                values = self.accumulate(episode.shaped, discount)
                self.assertAlmostEqual(sum(values) / len(values), episode.outcome, places=6)

    def test_a_compaction_row_carries_no_credit_but_is_still_addressed(self) -> None:
        """Omitting it does not exclude it — `individual` exports and trains on it.

        An unaddressed row takes reward 0.0 and then accumulates its neighbour's
        value, which is a trained row carrying credit no policy earned. It is
        given a value instead, with the shaping term zero.
        """
        episode = self.reward(completions=[*COMPLETIONS, completion(7, purpose="compaction")])
        self.assertIn("chatcmpl-7", episode.shaped)
        for value in self.accumulate(episode.shaped):
            self.assertAlmostEqual(value, episode.outcome)


if __name__ == "__main__":
    unittest.main()
