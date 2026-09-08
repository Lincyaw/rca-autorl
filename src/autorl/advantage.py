"""Advantage for a multi-turn episode: outcome across samples, credit within one.

AReaL's own normalization cannot do this, and reading why is what settles the
shape. With `export_style: individual` a rollout's turns are concatenated into
one trajectory, and `concat_batch` reports that trajectory's *row count* as its
group size, so `reward_norm(mean_level="group")` centres each turn against the
other turns of the same rollout. Two consequences follow. A single terminal
reward propagates backward to an identical value on every turn, the group mean
equals it, and the advantage is zero everywhere — no gradient at all. And any
non-uniform reward is centred against position rather than against the sibling
samples, so an episode that was simply right is indistinguishable from one that
was wrong. The comparison RLOO exists to make never happens.

So the two axes are separated here instead. A turn's reward carries both (see
`autorl.reward`): the mean over a trajectory's turns is its outcome, and each
turn's deviation from that mean is its share of the investigation. The outcome
is compared across the samples of one prompt, leave-one-out, which is RLOO. The
deviation is already zero-mean within the trajectory and is passed through. A
turn's advantage is their sum, so being right and having earned it are added,
never traded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Trajectory:
    """One rollout: the per-turn rewards, and which prompt it is a sample of."""

    prompt: str
    turns: tuple[float, ...]

    @property
    def outcome(self) -> float:
        """The trajectory's own return: the mean its shaping was centred around."""
        return sum(self.turns) / len(self.turns) if self.turns else 0.0


def advantages(trajectories: Sequence[Trajectory]) -> list[list[float]]:
    """Per-turn advantages, one list per trajectory, in the order given.

    A prompt sampled once has no sibling to compare against, so its outcome
    advantage is zero and only the within-episode credit survives. That is the
    honest answer rather than a fabricated baseline: with nothing to compare to,
    nothing is known about whether the outcome was good.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for trajectory in trajectories:
        totals[trajectory.prompt] = totals.get(trajectory.prompt, 0.0) + trajectory.outcome
        counts[trajectory.prompt] = counts.get(trajectory.prompt, 0) + 1

    out: list[list[float]] = []
    for trajectory in trajectories:
        siblings = counts[trajectory.prompt] - 1
        baseline = (totals[trajectory.prompt] - trajectory.outcome) / siblings if siblings else None
        outcome_advantage = 0.0 if baseline is None else trajectory.outcome - baseline
        mean = trajectory.outcome
        out.append([outcome_advantage + (turn - mean) for turn in trajectory.turns])
    return out


__all__ = ["Trajectory", "advantages"]
