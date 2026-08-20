"""Minimal fake MCP server placeholder for runtime integration tests."""


async def echo_tool(arguments: dict) -> dict:
    return {"result": arguments.get("message")}
