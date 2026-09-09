"""Trainer configuration dataclasses.

These live outside the entrypoint module on purpose. AReaL ships the config to
each worker over RPC and rebuilds it by looking the class up as
``<module>.<qualname>``; a dataclass defined in the module run as
``python -m autorl.train`` is ``__main__.RCAPPOConfig`` there, which no worker
can resolve. The config then degrades to a plain dict and the worker fails on
the first attribute access (``'dict' object has no attribute 'seed'``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class DshConfig:
    """The RCA workflow's own settings; everything else comes from AReaL."""

    scenario: str = "rca"
    dataset_root: str = ""
    dsh_home: str = ""
    timeout: float = 1800.0
    # Weight recall by how few siblings found each element (method spec §3.1);
    # off scores the flat graph score, for the ablation.
    difficulty: bool = True
    # The centring of spec §3.2, computed by the workflow per trajectory:
    # rloo, grpo, or remax (the group's first sample decoded greedily, its
    # score the baseline, trained with zero advantage).
    centring: str = "rloo"
    # Forking (spec §4): every `fork_every`-th prompt forks one sample at a
    # random step into `fork_siblings` responses, each run to the end
    # `fork_continuations` times. 0 is off.
    fork_every: int = 0
    fork_siblings: int = 4
    fork_continuations: int = 2


@dataclass
class RCAPPOConfig(PPOConfig):  # type: ignore[misc]  # AReaL has no py.typed marker
    econfig: DshConfig = field(default_factory=DshConfig)


__all__ = ["DshConfig", "RCAPPOConfig"]
