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

    @staticmethod
    def accumulate(shaped: dict[str, float], discount: float = 1.0) -> list[float]:
        """What AReaL makes of the own-rewards: `value[i] = own[i] + value[i+1] * discount`."""
        values: list[float] = []
        running = 0.0
        for own in reversed(list(shaped.values())):
            running = own + running * discount
            values.append(running)
        return list(reversed(values))

    def test_each_turn_accumulates_to_its_outcome_plus_its_block_credit(self) -> None:
        """The own-rewards are differences; the contract is what they sum back to."""
        episode = self.reward()
        values = self.accumulate(episode.shaped)
        # Four turns in the first block (three queries and the note), three in
        # the second; centring is per turn, so that is the weighting.
        level = (4 * 0.2 * (2 / 3) + 3 * 0.2 * 1.0) / 7
        first_block = 0.2 * (2 / 3) - level
        second_block = 0.2 * 1.0 - level
        self.assertEqual(len(values), 7)
        for value in values[:4]:
            self.assertAlmostEqual(value, first_block)
        for value in values[4:]:
            self.assertAlmostEqual(value, second_block)

    def test_the_shaping_is_a_redistribution(self) -> None:
        """A trajectory's mean turn value is its outcome, whatever the blocks did."""
        episode = self.reward()
        values = self.accumulate(episode.shaped)
        self.assertAlmostEqual(sum(values) / len(values), episode.outcome, places=6)

    def test_no_submission_scores_the_outcome_zero(self) -> None:
        episode = self.reward()
        self.assertEqual(episode.outcome, 0.0)
        self.assertEqual(episode.blocks, 2)

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

    def test_a_turn_that_closes_and_opens_is_credited_to_what_it_closed(self) -> None:
        episode = episode_reward(
            events=SHARED_STEP_EVENTS,
            completions=SHARED_STEP_COMPLETIONS,
            submission=None,
            truth=TRUTH,
            shaping=1.0,
        )
        values = self.accumulate(episode.shaped)
        # Rows in cache order: turn 1, turn 2, compaction, turn 3, turn 4.
        # Block A is the geo query (rate 1.0) and closes on turn 2; block B is
        # the logs query (rate 0.0) and closes on turn 3. Turn 2 holds both.
        level = (1.0 + 1.0 + 0.0 + 0.0 + 0.0) / 5
        self.assertAlmostEqual(values[0], 1.0 - level)  # block A
        self.assertAlmostEqual(values[1], 1.0 - level)  # closes A, opens B
        self.assertAlmostEqual(values[3], 0.0 - level)  # closes B

    def test_the_submit_turn_inherits_the_last_block_rather_than_zero(self) -> None:
        episode = episode_reward(
            events=SHARED_STEP_EVENTS,
            completions=SHARED_STEP_COMPLETIONS,
            submission=None,
            truth=TRUTH,
            shaping=1.0,
        )
        values = self.accumulate(episode.shaped)
        self.assertAlmostEqual(values[4], values[3])  # submit follows block B

    def test_a_compaction_row_is_neutral(self) -> None:
        """It is exported and trained on, so it cannot inherit a neighbour."""
        episode = episode_reward(
            events=SHARED_STEP_EVENTS,
            completions=SHARED_STEP_COMPLETIONS,
            submission=None,
            truth=TRUTH,
            shaping=1.0,
        )
        values = self.accumulate(episode.shaped)
        self.assertAlmostEqual(values[2], episode.outcome - (1.0 + 1.0) / 5)
        self.assertAlmostEqual(sum(values) / len(values), episode.outcome)

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
        values = self.accumulate(episode.shaped)
        credited = [v for v in values[:-1] if abs(v - values[-1]) > 1e-9]
        self.assertTrue(credited, "the agent turns should not all match the neutral row")


if __name__ == "__main__":
    unittest.main()
