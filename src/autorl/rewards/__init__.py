"""Reward interfaces for autorl."""

from .base import RewardStrategy
from .llmharness_auditor import LlmharnessAuditorProcessRewardStrategy
from .llmharness_auditor_outcome import LlmharnessAuditorOutcomeRewardStrategy
from .llmharness_extractor import LlmharnessExtractorRewardStrategy
from .llmharness_extractor_outcome import LlmharnessExtractorOutcomeRewardStrategy
from .local import ReferenceMatchRewardStrategy, reference_match_reward
from .rca import RCABaselineRewardStrategy
from .registry import get_reward_fn, list_rewards, register_reward

register_reward("reference_match", reference_match_reward)

__all__ = [
    "RewardStrategy",
    "ReferenceMatchRewardStrategy",
    "RCABaselineRewardStrategy",
    "LlmharnessExtractorRewardStrategy",
    "LlmharnessExtractorOutcomeRewardStrategy",
    "LlmharnessAuditorProcessRewardStrategy",
    "LlmharnessAuditorOutcomeRewardStrategy",
    "reference_match_reward",
    "register_reward",
    "get_reward_fn",
    "list_rewards",
]
