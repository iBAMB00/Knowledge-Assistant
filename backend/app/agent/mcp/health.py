"""MCP server health monitor.

A3.6: provide MCP runtime health state tracking.
"""


class MCPHealthMonitor:
    def __init__(self) -> None:
        self._states: dict[str, str] = {}

    def set_status(self, server_id: str, status: str) -> None:
        self._states[server_id] = status

    def get_status(self, server_id: str) -> str | None:
        return self._states.get(server_id)

    def is_healthy(self, server_id: str) -> bool:
        return self.get_status(server_id) == "READY"
