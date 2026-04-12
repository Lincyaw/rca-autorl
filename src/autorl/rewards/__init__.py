"""Reward interfaces for autorl."""

from .base import RewardStrategy
from .local import ReferenceMatchRewardStrategy, reference_match_reward
from .rca import RootCauseMatchRewardStrategy, root_cause_f1_reward
from .registry import get_reward_fn, list_rewards, register_reward

register_reward("reference_match", reference_match_reward)
register_reward("root_cause_f1", root_cause_f1_reward)

__all__ = [
    "RewardStrategy",
    "ReferenceMatchRewardStrategy",
    "RootCauseMatchRewardStrategy",
    "reference_match_reward",
    "root_cause_f1_reward",
    "register_reward",
    "get_reward_fn",
    "list_rewards",
]
