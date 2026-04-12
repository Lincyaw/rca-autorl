from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from autorl.contracts import EnvStepResult

from .base import EnvGateway


class NullEnvGateway(EnvGateway):
    """Minimal environment gateway for integration tests and dry runs."""

    def __init__(self, name: str = "null_env") -> None:
        self.name = name
        self._step_count = 0
        self._last_observation: Any = None

    async def reset(self, params: Mapping[str, Any] | None = None) -> EnvStepResult:
        start = time.perf_counter()
        self._step_count = 0
        params = params or {}
        self._last_observation = params.get("initial_observation")
        latency_ms = (time.perf_counter() - start) * 1000.0
        return EnvStepResult(
            ok=True,
            name=self.name,
            observation=self._last_observation,
            reward=0.0,
            done=False,
            info={"step_count": self._step_count},
            latency_ms=latency_ms,
        )

    async def step(self, action: Mapping[str, Any]) -> EnvStepResult:
        start = time.perf_counter()
        self._step_count += 1
        self._last_observation = {"echo_action": dict(action), "step_count": self._step_count}
        latency_ms = (time.perf_counter() - start) * 1000.0
        return EnvStepResult(
            ok=True,
            name=self.name,
            observation=self._last_observation,
            reward=0.0,
            done=False,
            info={"step_count": self._step_count},
            latency_ms=latency_ms,
        )
