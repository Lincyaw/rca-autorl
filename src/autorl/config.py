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


@dataclass
class RCAPPOConfig(PPOConfig):  # type: ignore[misc]  # AReaL has no py.typed marker
    econfig: DshConfig = field(default_factory=DshConfig)


__all__ = ["DshConfig", "RCAPPOConfig"]
