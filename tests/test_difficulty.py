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

    @staticmethod
    def interaction(reward: float) -> object:
        return type("I", (), {"reward": reward})()

    def result(self, turns: int) -> dict[str, object]:
        return {f"c{i}": self.interaction(0.0) for i in range(turns)}

    def run_hook(self, results: list[object]) -> object:
        import asyncio

        return asyncio.run(self.workflow.rescore_group(results))

    def test_the_whole_score_lands_on_the_last_turn(self) -> None:
        self.workflow._answers = {
            0: answer({"svc:told", "svc:hard"}),
            1: answer({"svc:told"}),
        }
        results = [self.result(3), self.result(2)]
        out = self.run_hook(results)
        self.assertIsNotNone(out)
        first, second = results
        self.assertEqual([i.reward for i in list(first.values())[:-1]], [0.0, 0.0])
        self.assertGreater(list(first.values())[-1].reward, list(second.values())[-1].reward)

    def test_an_incomplete_group_is_left_alone(self) -> None:
        """Weights read off a partial group would call its missing parts hard."""
        self.workflow._answers = {0: answer({"svc:hard"})}
        self.assertIsNone(self.run_hook([self.result(2), None]))

    def test_a_group_that_never_recorded_is_left_alone(self) -> None:
        self.assertIsNone(self.run_hook([self.result(2), self.result(2)]))
