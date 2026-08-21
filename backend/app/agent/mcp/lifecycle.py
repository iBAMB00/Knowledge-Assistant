"""MCP runtime lifecycle manager.

A3.6: unify startup and shutdown lifecycle.
v2.2.1: restore persisted MCP server definitions before tool loading.
"""


class MCPLifecycleManager:
    def __init__(
        self,
        *,
        runtime_initializer,
        connection_manager,
        registry_bootstrapper=None,
    ) -> None:
        self._runtime_initializer = runtime_initializer
        self._connection_manager = connection_manager
        self._registry_bootstrapper = registry_bootstrapper

    async def startup(self):
        if self._registry_bootstrapper is not None:
            self._registry_bootstrapper.restore()

        return await self._runtime_initializer.initialize()

    async def shutdown(self):
        await self._connection_manager.disconnect_all()
