"""Core learning rules for evidence-grounded RCA policy optimization.

The production trainer delegates advantage computation to AReaL v2. This module
defines reward semantics, validates the required AReaL configuration, and
provides the dynamic group filter used during rollout collection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from autorl.interfaces import AReaLRLOOConfig, TensorLike


@dataclass(slots=True)
class RCARewardConfig:
    """Weights for terminal utility and investigation costs."""

    cause_weight: float = 1.0
    attribution_weight: float = 0.5
    violation_weight: float = 1.0
    correct_cause_reward: float = 1.0
    incorrect_cause_penalty: float = 1.0
    missed_diagnosis_penalty: float = 1.0
    tool_call_cost: float = 0.01
    token_cost: float = 0.0
    elapsed_second_cost: float = 0.0
    redundant_action_cost: float = 0.01
    invalid_action_cost: float = 0.05
    declaration_cost: float = 0.001

    def __post_init__(self) -> None:
        values = (getattr(self, item.name) for item in fields(self))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("reward weights and costs must be finite and non-negative")
        if self.cause_weight * self.incorrect_cause_penalty <= self.attribution_weight:
            raise ValueError(
                "reward safety requires cause_weight * incorrect_cause_penalty > attribution_weight"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> RCARewardConfig:
        if values is None:
            return cls()
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown reward setting(s): {sorted(unknown)}")
        parsed: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"reward setting {key!r} must be numeric")
            parsed[key] = float(value)
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class EpisodeSignals:
    """Verified outcome signals and measured resource use for one episode."""

    cause_correct: bool
    attribution_score: float = 0.0
    violation_count: int = 0
    tool_calls: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0
    redundant_actions: int = 0
    invalid_actions: int = 0
    declared_objects: int = 0
    has_submission: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.attribution_score) or self.attribution_score > 1.0:
            raise ValueError("attribution_score must be finite and no greater than 1")
        counts = (
            self.violation_count,
            self.tool_calls,
            self.tokens,
            self.redundant_actions,
            self.invalid_actions,
            self.declared_objects,
        )
        if any(value < 0 for value in counts) or self.elapsed_seconds < 0:
            raise ValueError("episode costs must be non-negative")


def compute_episode_reward(
    signals: EpisodeSignals,
    config: RCARewardConfig,
) -> dict[str, float]:
    """Compute terminal utility minus measured investigation costs."""
    cause_correct = signals.cause_correct and signals.has_submission
    cause_score = config.correct_cause_reward if cause_correct else -config.incorrect_cause_penalty
    attribution_score = signals.attribution_score if signals.has_submission else 0.0
    terminal_reward = (
        config.cause_weight * cause_score
        + config.attribution_weight * attribution_score
        - config.violation_weight * signals.violation_count
    )
    investigation_cost = (
        config.tool_call_cost * signals.tool_calls
        + config.token_cost * signals.tokens
        + config.elapsed_second_cost * signals.elapsed_seconds
        + config.redundant_action_cost * signals.redundant_actions
        + config.invalid_action_cost * signals.invalid_actions
        + config.declaration_cost * signals.declared_objects
    )
    return {
        "reward": terminal_reward - investigation_cost,
        "terminal_reward": terminal_reward,
        "cause_score": cause_score,
        "attribution_score": attribution_score,
        "investigation_cost": investigation_cost,
        "has_submission": float(signals.has_submission),
    }


def anomaly_attribution_score(
    *,
    correct: int,
    total: int,
    false_dismissals: int,
    missed_diagnosis_penalty: float,
) -> float:
    """Score task-provided anomalies using the method's discrete outcomes."""
    if total <= 0:
        return 0.0
    if correct < 0 or false_dismissals < 0 or correct + false_dismissals > total:
        raise ValueError("invalid anomaly attribution counts")
    if missed_diagnosis_penalty < 0 or not math.isfinite(missed_diagnosis_penalty):
        raise ValueError("missed_diagnosis_penalty must be finite and non-negative")
    return (correct - missed_diagnosis_penalty * false_dismissals) / total


def keep_informative_group(trajectory: Mapping[str, object]) -> bool:
    """Drop tied non-positive groups; keep contrastive and successful groups.

    AReaL v2 calls this after grouped rollouts and refills rejected groups when
    ``dynamic_bs`` is disabled. Keeping all tied successful groups corresponds
    to an all-success retention fraction of one.
    """
    returns = _sequence_returns(trajectory.get("rewards"))
    if len(returns) < 2:
        return False
    if not all(math.isclose(returns[0], value) for value in returns[1:]):
        return True
    return returns[0] > 0.0


def validate_areal_v2_rloo(config: AReaLRLOOConfig) -> None:
    """Fail fast unless AReaL v2 is configured as broadcast RLOO."""
    group_size = int(config.gconfig.n_samples)
    reward_norm = config.actor.reward_norm
    errors: list[str] = []
    if group_size < 2:
        errors.append("gconfig.n_samples must be >= 2")
    if config.gconfig.reward_normalization:
        errors.append("gconfig.reward_normalization must be false (it divides by std)")
    if reward_norm is None:
        errors.append("actor.reward_norm is required")
    else:
        if reward_norm.mean_level != "group" or not reward_norm.mean_leave1out:
            errors.append("actor.reward_norm must use group leave-one-out centering")
        if reward_norm.std_level is not None:
            errors.append("actor.reward_norm.std_level must be null")
        if int(reward_norm.group_size) != group_size:
            errors.append("actor.reward_norm.group_size must equal gconfig.n_samples")
    if config.actor.adv_norm is not None:
        errors.append("actor.adv_norm must be null")
    if float(config.actor.discount) != 1.0 or float(config.actor.gae_lambda) != 1.0:
        errors.append("actor.discount and actor.gae_lambda must both equal 1")
    if config.critic is not None:
        errors.append("critic must be null for critic-free RLOO")
    if errors:
        raise ValueError("invalid RCA RLOO config: " + "; ".join(errors))


def _sequence_returns(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, TensorLike):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return [_as_float(value)]
    result: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            result.append(math.fsum(_as_float(token_reward) for token_reward in item))
        else:
            result.append(_as_float(item))
    return result


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"reward values must be numeric, got {type(value).__name__}")
    return float(value)


__all__ = [
    "EpisodeSignals",
    "RCARewardConfig",
    "anomaly_attribution_score",
    "compute_episode_reward",
    "keep_informative_group",
    "validate_areal_v2_rloo",
]
