from __future__ import annotations

import unittest

from autorl.algorithm import (
    EpisodeSignals,
    RCARewardConfig,
    anomaly_attribution_score,
    compute_episode_reward,
    fork_advantages,
    keep_informative_group,
    rloo_advantages,
)


class RewardTest(unittest.TestCase):
    def test_anomaly_score_penalizes_false_dismissals(self) -> None:
        self.assertEqual(
            anomaly_attribution_score(
                correct=2,
                total=4,
                false_dismissals=1,
                missed_diagnosis_penalty=1.0,
            ),
            0.25,
        )

    def test_terminal_utility_minus_action_cost(self) -> None:
        metrics = compute_episode_reward(
            EpisodeSignals(
                cause_correct=True,
                attribution_score=0.5,
                tool_calls=4,
                invalid_actions=1,
            ),
            RCARewardConfig(tool_call_cost=0.01, invalid_action_cost=0.05),
        )
        self.assertAlmostEqual(metrics["terminal_reward"], 1.25)
        self.assertAlmostEqual(metrics["investigation_cost"], 0.09)
        self.assertAlmostEqual(metrics["reward"], 1.16)

    def test_missing_submission_is_incorrect_and_has_no_attribution_credit(self) -> None:
        metrics = compute_episode_reward(
            EpisodeSignals(
                cause_correct=True,
                attribution_score=1.0,
                has_submission=False,
            ),
            RCARewardConfig(),
        )
        self.assertEqual(metrics["reward"], -1.0)
        self.assertEqual(metrics["attribution_score"], 0.0)

    def test_rejects_unsafe_reward_weights(self) -> None:
        with self.assertRaises(ValueError):
            RCARewardConfig(attribution_weight=1.0)


class AdvantageTest(unittest.TestCase):
    def test_rloo_uses_other_rollouts_as_baseline(self) -> None:
        self.assertEqual(rloo_advantages([1.0, 0.0, -1.0]), [1.5, 0.0, -1.5])

    def test_fork_advantage_averages_continuations_before_rloo(self) -> None:
        self.assertEqual(fork_advantages([[1.0, 3.0], [0.0, 0.0]]), [2.0, -2.0])

    def test_dynamic_filter_drops_only_tied_failures(self) -> None:
        self.assertFalse(keep_informative_group({"rewards": [[-1.0], [-1.0]]}))
        self.assertTrue(keep_informative_group({"rewards": [[-1.0], [-0.5]]}))
        self.assertTrue(keep_informative_group({"rewards": [[1.0], [1.0]]}))


if __name__ == "__main__":
    unittest.main()
