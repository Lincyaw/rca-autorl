from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from autorl.contracts import ToolCallResult
from autorl.tool_env.client import ToolEnvClient

from .base import ToolGateway


class ToolEnvGateway(ToolGateway):
    """Tool gateway backed by the existing tool_env client."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: ToolEnvClient | None = None,
    ) -> None:
        self.client = client or ToolEnvClient(base_url=base_url, timeout_s=timeout_s)

    async def list_tools(self) -> list[dict[str, Any]]:
        return await self.client.list_tools()

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolCallResult:
        start = time.perf_counter()
        payload = await self.client.execute(tool_name=name, arguments=dict(arguments))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ToolCallResult(
            ok=bool(payload.get("ok", False)),
            name=str(payload.get("name", name)),
            result=payload.get("result"),
            error=payload.get("error"),
            latency_ms=latency_ms,
            metadata={},
        )
