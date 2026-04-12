from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Metadata


@dataclass(slots=True)
class RewardAssignment:
    """Reward keyed by interaction/completion identifier."""

    interaction_id: str
    reward: float
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class TrainingEpisodeView:
    """Training-ready projection distilled from trajectory + outcome."""

    sample_id: str
    trajectory_id: str
    model_interactions: list[dict[str, Any]] = field(default_factory=list)
    reward_assignments: list[RewardAssignment] = field(default_factory=list)
    discount_config: dict[str, Any] = field(default_factory=dict)
    aux_metrics: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)
