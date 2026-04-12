from __future__ import annotations

from collections.abc import Callable


_REWARD_REGISTRY: dict[str, Callable] = {}


def register_reward(name: str, reward_fn: Callable) -> None:
    _REWARD_REGISTRY[name] = reward_fn


def get_reward_fn(name: str) -> Callable:
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"reward '{name}' is not registered")
    return _REWARD_REGISTRY[name]


def list_rewards() -> list[str]:
    return sorted(_REWARD_REGISTRY.keys())
