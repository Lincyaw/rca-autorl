from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from autorl.contracts import EnvStepResult, ToolCallResult


class ToolGateway(ABC):
    """Framework-agnostic tool invocation interface."""

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolCallResult:
        raise NotImplementedError


class EnvGateway(ABC):
    """Framework-agnostic environment interaction interface."""

    @abstractmethod
    async def reset(
        self,
        params: Mapping[str, Any] | None = None,
    ) -> EnvStepResult:
        raise NotImplementedError

    @abstractmethod
    async def step(
        self,
        action: Mapping[str, Any],
    ) -> EnvStepResult:
        raise NotImplementedError
