from __future__ import annotations

from typing import Any

import httpx


class ToolEnvClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 120.0):
        self.base_url = base_url
        self.timeout_s = timeout_s

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self.base_url:
            from .app import list_tools_local

            return await list_tools_local()

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get(f"{self.base_url.rstrip('/')}/tools")
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            return data.get("tools", [])

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            from .app import execute_tool_local

            return await execute_tool_local(tool_name=tool_name, arguments=arguments)

        payload = {"name": tool_name, "arguments": arguments}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/tool/execute",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data
            return {
                "ok": False,
                "name": tool_name,
                "error": "invalid tool env response",
                "result": "",
            }
