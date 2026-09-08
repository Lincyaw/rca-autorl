"""The PPO actor that computes this task's advantage instead of AReaL's.

AReaL's own normalization is not wrong; it is built for one sample per rollout.
With `export_style: individual` a rollout's turns are concatenated into a single
trajectory and `concat_batch` reports that trajectory's row count as its group
size, so `reward_norm(mean_level="group")` centres each turn against the other
turns of the same rollout. A terminal reward then propagates backward to an
identical value on every turn, the group mean equals it, and the advantage is
zero everywhere; anything non-uniform is centred against position instead of
against the sibling samples, and the leave-one-out comparison RLOO exists to
make never happens.

`autorl.advantage` does that comparison, over the two axes a turn's reward
encodes. What is left here is plumbing: read the rewards before AReaL scales
them, let the base method build every other field it owns, and substitute the
advantage it computed.
"""

from __future__ import annotations

from typing import Any

import torch
from areal import PPOTrainer
from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import TrajBatchMeta

from autorl.advantage import Trajectory, advantages


class RcaPPOActor(PPOActor):  # type: ignore[misc]  # areal ships no stubs
    """`PPOActor` with the RCA advantage in place of the group normalization."""

    def __init__(self, config: PPOActorConfig, engine: Any, *, group_size: int) -> None:
        super().__init__(config, engine)
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2 to have a baseline, got {group_size}")
        self.group_size = group_size

    def _compute_advantages(
        self, data: dict[str, Any], meta: TrajBatchMeta | None = None
    ) -> dict[str, Any]:
        # Before the base method scales, clips and normalizes them: these are the
        # turn values `autorl.reward` built, and both axes are read off them.
        rewards = data["rewards"].detach().to(torch.float64).flatten().tolist()
        data = super()._compute_advantages(data, meta)
        if meta is None or not meta.traj_group_sizes:
            return data
        per_turn = self._rca_advantages(rewards, list(meta.traj_group_sizes))
        if len(per_turn) != data["advantages"].shape[0]:
            # The batch is not shaped the way this reads it; the base method's
            # advantage is still a defensible number, and a silently misaligned
            # one is not.
            return data
        column = torch.tensor(
            per_turn, dtype=data["advantages"].dtype, device=data["advantages"].device
        ).unsqueeze(-1)
        data["advantages"] = column.expand_as(data["advantages"]).contiguous()
        return data

    def _rca_advantages(self, rewards: list[float], group_sizes: list[int]) -> list[float]:
        """One advantage per row, from the trajectories this batch holds.

        `group_sizes` is one entry per trajectory, in batch order, and the
        `group_size` samples of one prompt are consecutive — which is the only
        handle on "same prompt" that survives into the tensors.
        """
        trajectories: list[Trajectory] = []
        offset = 0
        for index, size in enumerate(group_sizes):
            turns = tuple(rewards[offset : offset + size])
            trajectories.append(Trajectory(prompt=str(index // self.group_size), turns=turns))
            offset += size
        return [value for trajectory in advantages(trajectories) for value in trajectory]


class RcaPPOTrainer(PPOTrainer):  # type: ignore[misc]  # areal ships no stubs
    """`PPOTrainer` whose actor computes the RCA advantage."""

    def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
        self._group_size = int(config.gconfig.n_samples)
        super().__init__(config, *args, **kwargs)

    def _create_train_engine(self, actor_config: Any, alloc: Any) -> Any:
        engine = super()._create_train_engine(actor_config, alloc)
        # The reference engine comes through here too. It never computes an
        # advantage, so swapping its actor costs nothing and testing which one
        # this is would rely on ordering.
        if hasattr(engine, "actor"):
            engine.actor = RcaPPOActor(actor_config, engine, group_size=self._group_size)
        return engine


__all__ = ["RcaPPOActor", "RcaPPOTrainer"]
