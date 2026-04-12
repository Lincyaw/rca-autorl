"""Reward interfaces for autorl."""

from .base import RewardStrategy
from .local import ReferenceMatchRewardStrategy, reference_match_reward
from .registry import get_reward_fn, list_rewards, register_reward

register_reward("reference_match", reference_match_reward)

__all__ = [
    "RewardStrategy",
    "ReferenceMatchRewardStrategy",
    "reference_match_reward",
    "register_reward",
    "get_reward_fn",
    "list_rewards",
]
