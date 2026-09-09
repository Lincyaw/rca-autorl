"""An element every sibling found separates nothing, and should be worth nothing."""

from __future__ import annotations

import unittest

from autorl.difficulty import AXES, Answer, element_weights, weighted_score

TRUTH = {
    "roots": frozenset({"svc:hard"}),
    "subjects": frozenset({"svc:told", "svc:middle", "svc:hard"}),
    "edges": frozenset(),
}


def group_scores(group: list[Answer]) -> list[float]:
    weights = {axis: element_weights(group, axis) for axis, _ in AXES}
    return [weighted_score(a, weights) for a in group]


def answer(found_subjects: set[str], found_roots: set[str] = frozenset()) -> Answer:
    return Answer(
        found={"roots": frozenset(found_roots), "subjects": frozenset(found_subjects)},
        truth=TRUTH,
    )


class WeightTest(unittest.TestCase):
    def test_an_element_everyone_found_is_worth_nothing(self) -> None:
        """`svc:told` is the service the incident text names."""
        group = [answer({"svc:told"}) for _ in range(8)]
        weights = element_weights(group, "subjects")
        self.assertEqual(weights["svc:told"], 0.0)

    def test_an_element_one_sibling_found_is_worth_almost_everything(self) -> None:
        group = [answer({"svc:told", "svc:hard"})] + [answer({"svc:told"}) for _ in range(7)]
        weights = element_weights(group, "subjects")
        self.assertAlmostEqual(weights["svc:hard"], 7 / 8)

    def test_an_element_nobody_found_keeps_full_weight(self) -> None:
        """The hardest part of a case is not the least relevant part of it."""
        group = [answer({"svc:told"}) for _ in range(8)]
        self.assertEqual(element_weights(group, "subjects")["svc:middle"], 1.0)


class ScoreTest(unittest.TestCase):
    def test_the_sibling_who_found_the_hard_one_pulls_ahead(self) -> None:
        group = [answer({"svc:told", "svc:hard"})] + [answer({"svc:told"}) for _ in range(7)]
        scores = group_scores(group)
        self.assertGreater(scores[0], scores[1])

    def test_a_group_that_all_found_the_same_thing_is_flat(self) -> None:
        group = [answer({"svc:told", "svc:middle"}) for _ in range(8)]
        scores = group_scores(group)
        for score in scores[1:]:
            self.assertAlmostEqual(score, scores[0])

    def test_restating_the_prompt_earns_nothing_once_everyone_does_it(self) -> None:
        """The failure this weighting exists to remove."""
        weights = {"subjects": {"svc:told": 0.0, "svc:middle": 1.0, "svc:hard": 1.0}, "roots": {}}
        only_told = weighted_score(answer({"svc:told"}), weights)
        self.assertEqual(only_told, 0.0)

    def test_a_claim_outside_the_graph_costs_the_same_wherever_it_lands(self) -> None:
        """Difficulty is a property of the truth; a wrong claim has none."""
        weights = {"subjects": {"svc:told": 0.0, "svc:middle": 1.0, "svc:hard": 1.0}, "roots": {}}
        clean = Answer(
            found={"subjects": frozenset({"svc:hard"})},
            truth=TRUTH,
            claimed={"subjects": frozenset({"svc:hard"})},
        )
        noisy = Answer(
            found={"subjects": frozenset({"svc:hard"})},
            truth=TRUTH,
            claimed={"subjects": frozenset({"svc:hard", "svc:invented"})},
        )
        self.assertGreater(weighted_score(clean, weights), weighted_score(noisy, weights))


if __name__ == "__main__":
    unittest.main()


class WorkflowHookTest(unittest.TestCase):
    """`rescore_group` writes every row's final value, or refuses to."""

    def setUp(self) -> None:
        from autorl.agent import DshWorkflow

        self.workflow = DshWorkflow.__new__(DshWorkflow)
        self.workflow._groups = {}
        self.workflow.difficulty = True
        self.workflow.centring = "rloo"

    @staticmethod
    def interaction(reward: float) -> object:
        return type("I", (), {"reward": reward})()

    def result(self, turns: int, value: float = 0.0, prefix: str = "c") -> dict[str, object]:
        """A sample's rows as they arrive: every row already carries its value."""
        return {f"{prefix}{i}": self.interaction(value) for i in range(turns)}

    def sample(self, answer: Answer, turns: int, prefix: str = "c") -> object:
        """The record `run` keeps: the answer and the ids of its own steps."""
        from autorl.agent import Sample

        return Sample(answer, {f"{prefix}{i}" for i in range(turns)})

    def samples(self, *answers) -> None:
        # Records are keyed by the group's task id; the test runs outside one.
        self.workflow._groups = {None: {i: self.sample(a, 3) for i, a in enumerate(answers)}}

    def run_hook(self, results: list[object]) -> object:
        import asyncio

        return asyncio.run(self.workflow.rescore_group(results))

    def rows(self, result: dict[str, object]) -> list[float]:
        return [i.reward for i in result.values()]

    def test_every_row_of_a_trajectory_carries_its_advantage(self) -> None:
        ambitious, told = answer({"svc:told", "svc:hard"}), answer({"svc:told"})
        self.samples(ambitious, told)
        results = [self.result(3, 0.5), self.result(2, 0.3)]
        self.assertIsNotNone(self.run_hook(results))
        first, second = self.rows(results[0]), self.rows(results[1])
        self.assertEqual(len(set(first)), 1)
        self.assertGreater(first[0], 0.0)
        self.assertAlmostEqual(first[0], -second[0])

    def test_the_difficulty_switch_changes_only_the_score(self) -> None:
        told = answer({"svc:told"})
        self.samples(told, told)
        results = [self.result(2, 0.3), self.result(2, 0.3)]
        self.run_hook(results)
        self.assertEqual(self.rows(results[0]), [0.0, 0.0])
        self.workflow.difficulty = False
        self.samples(told, told)
        self.run_hook(results)
        self.assertEqual(self.rows(results[0]), [0.0, 0.0])

    def test_remax_subtracts_the_greedy_sample_and_trains_it_with_nothing(self) -> None:
        """Spec §3.2: the first sample is the baseline, not a competitor."""
        self.workflow.centring = "remax"
        self.workflow.difficulty = False
        greedy, better = answer({"svc:told"}), answer({"svc:told", "svc:hard"})
        self.samples(greedy, better, greedy)
        results = [self.result(2), self.result(2), self.result(2)]
        self.run_hook(results)
        self.assertEqual(self.rows(results[0]), [0.0, 0.0])
        self.assertGreater(self.rows(results[1])[0], 0.0)
        self.assertEqual(self.rows(results[2]), [0.0, 0.0])

    def test_fork_rows_carry_the_fork_advantage_and_nothing_else(self) -> None:
        """Spec §4: the siblings replace the parent's turn; continuations are not trained."""
        from autorl.agent import Sibling

        told, hard = answer({"svc:told"}), answer({"svc:told", "svc:hard"})
        parent = self.sample(told, 3)
        parent.forked_at = "c1"
        parent.siblings = [Sibling("b0", [hard, hard]), Sibling("b1", [told, told])]
        self.workflow._groups = {None: {0: parent, 1: self.sample(told, 2)}}
        first = self.result(3, 0.2)
        first.update({key: self.interaction(0.0) for key in ("b0", "b0m", "b1", "b1m")})
        results = [first, self.result(2, 0.2)]
        self.run_hook(results)
        self.assertEqual(first["c1"].reward, 0.0)
        self.assertGreater(first["b0"].reward, 0.0)
        self.assertAlmostEqual(first["b0"].reward, -first["b1"].reward)
        self.assertEqual(first["b0m"].reward, 0.0)
        self.assertEqual(first["b1m"].reward, 0.0)
        # The parent's own turns still carry its trajectory advantage.
        self.assertEqual(first["c0"].reward, first["c2"].reward)

    def test_a_row_no_step_produced_carries_nothing(self) -> None:
        """The proxy caches the compaction summarizer's call too; it is not trained on."""
        self.samples(answer({"svc:told", "svc:hard"}), answer({"svc:told"}))
        results = [self.result(3, 0.5), self.result(3, 0.3)]
        results[0]["summary"] = self.interaction(0.5)
        self.run_hook(results)
        self.assertEqual(results[0]["summary"].reward, 0.0)
        self.assertGreater(results[0]["c0"].reward, 0.0)

    def test_an_incomplete_group_is_left_alone(self) -> None:
        """Weights read off a partial group would call its missing parts hard."""
        self.samples(answer({"svc:hard"}))
        self.assertIsNone(self.run_hook([self.result(2), None]))

    def test_a_group_that_never_recorded_is_left_alone(self) -> None:
        self.assertIsNone(self.run_hook([self.result(2), self.result(2)]))


class CentreTest(unittest.TestCase):
    def test_the_three_centrings(self) -> None:
        from autorl.agent import centre

        scores = [0.2, 0.6, 0.4]
        rloo = centre(scores, "rloo")
        self.assertAlmostEqual(rloo[0], 0.2 - 0.5)
        grpo = centre(scores, "grpo")
        self.assertAlmostEqual(sum(grpo), 0.0)
        self.assertEqual(centre([0.3, 0.3], "grpo"), [0.0, 0.0])
        for got, want in zip(centre(scores, "remax"), [0.0, 0.4, 0.2], strict=True):
            self.assertAlmostEqual(got, want)
