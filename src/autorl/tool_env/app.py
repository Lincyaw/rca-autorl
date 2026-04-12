from __future__ import annotations

from fastapi import FastAPI

from autorl.tool_env.tools import Search, Visit

app = FastAPI(title="autorl-tool-env")

_TOOLS = {
    "search": Search(),
    "visit": Visit(),
}


def _tool_spec(tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


async def list_tools_local() -> list[dict]:
    return [_tool_spec(tool) for tool in _TOOLS.values()]


async def execute_tool_local(tool_name: str, arguments: dict) -> dict:
    tool = _TOOLS.get(tool_name)
    if tool is None:
        return {
            "ok": False,
            "name": tool_name,
            "error": f"tool '{tool_name}' not found",
            "result": "",
        }

    try:
        result = await tool.call(arguments)
        return {
            "ok": True,
            "name": tool_name,
            "result": result,
        }
    except Exception as err:
        return {
            "ok": False,
            "name": tool_name,
            "error": str(err),
            "result": "",
        }


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/tools")
async def list_tools():
    return await list_tools_local()


@app.post("/tool/execute")
async def execute_tool(payload: dict):
    tool_name = payload.get("name", "")
    arguments = payload.get("arguments", {})
    return await execute_tool_local(tool_name=tool_name, arguments=arguments)
