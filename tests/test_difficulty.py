"""An element every sibling found separates nothing, and should be worth nothing."""

from __future__ import annotations

import unittest

from autorl.difficulty import Answer, element_weights, group_scores, weighted_score

TRUTH = {
    "roots": frozenset({"svc:hard"}),
    "subjects": frozenset({"svc:told", "svc:middle", "svc:hard"}),
    "edges": frozenset(),
}


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
    """`rescore_group` rewrites a whole group's rewards, or refuses to."""

    def setUp(self) -> None:
        from autorl.agent import DshWorkflow

        self.workflow = DshWorkflow.__new__(DshWorkflow)
        self.workflow._answers = {}
        self.workflow.difficulty = True
        self.workflow.remax = False

    @staticmethod
    def interaction(reward: float) -> object:
        return type("I", (), {"reward": reward})()

    def result(self, turns: int, last: float = 0.0) -> dict[str, object]:
        """A sample's completions, the flat score already on its last turn."""
        return {f"c{i}": self.interaction(last if i == turns - 1 else 0.0) for i in range(turns)}

    def run_hook(self, results: list[object]) -> object:
        import asyncio

        return asyncio.run(self.workflow.rescore_group(results))

    def test_the_difficulty_delta_lands_on_the_last_turn(self) -> None:
        ambitious, told = answer({"svc:told", "svc:hard"}), answer({"svc:told"})
        flat = {0: weighted_score(ambitious, {}), 1: weighted_score(told, {})}
        self.workflow._answers = {0: (ambitious, flat[0]), 1: (told, flat[1])}
        results = [self.result(3, flat[0]), self.result(2, flat[1])]
        out = self.run_hook(results)
        self.assertIsNotNone(out)
        first, second = results
        self.assertEqual([i.reward for i in list(first.values())[:-1]], [0.0, 0.0])
        self.assertGreater(list(first.values())[-1].reward, list(second.values())[-1].reward)
        self.assertEqual(list(second.values())[-1].reward, 0.0)

    def test_with_difficulty_off_the_flat_score_stands(self) -> None:
        self.workflow.difficulty = False
        told = answer({"svc:told"})
        self.workflow._answers = {0: (told, 0.3), 1: (told, 0.3)}
        results = [self.result(2, 0.3), self.result(2, 0.3)]
        self.run_hook(results)
        self.assertEqual([list(r.values())[-1].reward for r in results], [0.3, 0.3])

    def test_remax_subtracts_the_greedy_sample_and_trains_it_with_nothing(self) -> None:
        """Spec §3.2: the first sample is the baseline, not a competitor."""
        self.workflow.remax = True
        self.workflow.difficulty = False
        greedy, better = answer({"svc:told"}), answer({"svc:told", "svc:hard"})
        self.workflow._answers = {0: (greedy, 0.2), 1: (better, 0.6), 2: (greedy, 0.2)}
        results = [self.result(2, 0.2), self.result(2, 0.6), self.result(2, 0.2)]
        self.run_hook(results)
        last = [list(r.values())[-1].reward for r in results]
        self.assertEqual(last[0], 0.0)
        self.assertAlmostEqual(last[1], 0.4)
        self.assertAlmostEqual(last[2], 0.0)

    def test_an_incomplete_group_is_left_alone(self) -> None:
        """Weights read off a partial group would call its missing parts hard."""
        self.workflow._answers = {0: (answer({"svc:hard"}), 0.0)}
        self.assertIsNone(self.run_hook([self.result(2), None]))

    def test_a_group_that_never_recorded_is_left_alone(self) -> None:
        self.assertIsNone(self.run_hook([self.result(2), self.result(2)]))
