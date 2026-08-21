"""MCP runtime lifecycle manager.

A3.6: unify startup and shutdown lifecycle.
"""


class MCPLifecycleManager:
    def __init__(
        self,
        *,
        runtime_initializer,
        connection_manager,
    ) -> None:
        self._runtime_initializer = runtime_initializer
        self._connection_manager = connection_manager

    async def startup(self):
        return await self._runtime_initializer.initialize()

    async def shutdown(self):
        await self._connection_manager.disconnect_all()
