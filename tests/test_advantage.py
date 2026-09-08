"""Being right and having earned it are added, never traded."""

from __future__ import annotations

import unittest

from autorl.advantage import Trajectory, advantages


class OutcomeAxisTest(unittest.TestCase):
    def test_siblings_are_compared_leave_one_out(self) -> None:
        group = [
            Trajectory("case1", (1.0,)),
            Trajectory("case1", (0.0,)),
            Trajectory("case1", (0.0,)),
        ]
        first, second, third = advantages(group)
        self.assertAlmostEqual(first[0], 1.0)  # 1.0 - mean(0, 0)
        self.assertAlmostEqual(second[0], -0.5)  # 0.0 - mean(1, 0)
        self.assertAlmostEqual(third[0], -0.5)

    def test_a_group_that_all_scored_the_same_learns_nothing_from_the_outcome(self) -> None:
        group = [Trajectory("case1", (0.4,)) for _ in range(4)]
        for a in advantages(group):
            self.assertAlmostEqual(a[0], 0.0)

    def test_one_sample_has_no_baseline_to_borrow(self) -> None:
        (only,) = advantages([Trajectory("case1", (0.9, 0.9))])
        self.assertEqual(only, [0.0, 0.0])

    def test_prompts_do_not_borrow_each_other_baselines(self) -> None:
        group = [
            Trajectory("easy", (1.0,)),
            Trajectory("easy", (1.0,)),
            Trajectory("hard", (0.0,)),
            Trajectory("hard", (0.0,)),
        ]
        for a in advantages(group):
            self.assertAlmostEqual(a[0], 0.0)


class CreditAxisTest(unittest.TestCase):
    def test_a_turn_above_its_episode_mean_gets_more_of_the_credit(self) -> None:
        """Two rollouts, same outcome, different shape."""
        flat = Trajectory("case1", (0.5, 0.5, 0.5, 0.5))
        peaked = Trajectory("case1", (0.6, 0.6, 0.4, 0.4))
        a_flat, a_peaked = advantages([flat, peaked])
        for value in a_flat:  # same outcome, no shape
            self.assertAlmostEqual(value, 0.0)
        self.assertAlmostEqual(a_peaked[0], 0.1)
        self.assertAlmostEqual(a_peaked[2], -0.1)
        self.assertAlmostEqual(sum(a_peaked), 0.0)  # credit redistributes, never adds

    def test_shape_cannot_change_the_outcome_a_trajectory_is_judged_on(self) -> None:
        """The hack the centring exists to refuse: shape without substance."""
        honest = Trajectory("case1", (0.3, 0.3, 0.3))
        gamed = Trajectory("case1", (0.9, 0.0, 0.0))
        self.assertAlmostEqual(honest.outcome, gamed.outcome)
        a_honest, a_gamed = advantages([honest, gamed])
        self.assertAlmostEqual(sum(a_honest), sum(a_gamed))

    def test_the_two_axes_add(self) -> None:
        good = Trajectory("case1", (1.2, 0.8))  # outcome 1.0, first turn earned it
        bad = Trajectory("case1", (0.0, 0.0))
        a_good, a_bad = advantages([good, bad])
        self.assertAlmostEqual(a_good[0], 1.0 + 0.2)
        self.assertAlmostEqual(a_good[1], 1.0 - 0.2)
        self.assertAlmostEqual(a_bad[0], -1.0)


if __name__ == "__main__":
    unittest.main()
