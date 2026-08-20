"""Fake MCP provider used by MCP runtime integration tests.

This module only defines deterministic fake tool behavior. It does not depend on
business services.
"""


async def echo_tool(arguments: dict) -> dict:
    return {"result": arguments.get("message")}


async def slow_tool(arguments: dict) -> dict:
    import asyncio

    await asyncio.sleep(arguments.get("delay", 1))
    return {"result": "done"}
