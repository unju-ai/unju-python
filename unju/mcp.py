"""
Unju MCP Gateway

Connect to the Unju MCP Gateway for remote tool access including:
- Memory operations (mem0 compatible)
- Knowledge graph
- Checkpoints (LangGraph compatible)

Usage:
    import unju

    # Connect to gateway
    gateway = await unju.mcp.connect()

    # Memory operations
    await gateway.add_memory("User prefers dark mode", user_id="user123")
    results = await gateway.search_memory("preferences")
    context = await gateway.get_context("What settings does the user have?")

    # Checkpoints
    await gateway.save_checkpoint("thread-1", state={"messages": [...]})
    state = await gateway.get_checkpoint("thread-1")

    # Raw tool execution
    result = await gateway.execute("memory-search", {"query": "dark mode"})
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

import aiohttp


logger = logging.getLogger("unju.mcp")

DEFAULT_BASE_URL = "https://api.unju.ai"


# =============================================================================
# Types
# =============================================================================

@dataclass
class Tool:
    """An MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str = "unju"


@dataclass
class Memory:
    """A memory from the knowledge graph."""
    id: str
    content: str
    created_at: str
    metadata: Optional[Dict] = None
    relevance: float = 1.0


# =============================================================================
# Gateway
# =============================================================================

class Gateway:
    """
    Unju MCP Gateway.

    Provides access to remote MCP tools via the Unju API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key or os.environ.get("UNJU_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
        self._tools: Dict[str, Tool] = {}
        self._connected: bool = False

    async def connect(self) -> Gateway:
        """Connect to the MCP gateway."""
        if self._connected:
            return self

        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            }
        )

        await self._discover_tools()
        self._connected = True
        logger.info(f"Connected to Unju MCP Gateway ({len(self._tools)} tools)")
        return self

    async def disconnect(self) -> None:
        """Disconnect from the gateway."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def __aenter__(self) -> Gateway:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    async def _discover_tools(self) -> None:
        """Discover available tools from the API."""
        # Discover A2A tools from the server
        await self._discover_a2a_tools()

        # Register known memory tools
        memory_tools = [
            Tool(
                name="memory-add",
                description="Add a memory to the knowledge graph",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "user_id": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="memory-search",
                description="Search memories by semantic similarity",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="memory-get-context",
                description="Get relevant context for a conversation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="checkpoint-save",
                description="Save a conversation checkpoint",
                input_schema={
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                        "state": {"type": "object"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["thread_id", "state"],
                },
            ),
            Tool(
                name="checkpoint-get",
                description="Retrieve a conversation checkpoint",
                input_schema={
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                    },
                    "required": ["thread_id"],
                },
            ),
            Tool(
                name="entity-create",
                description="Create an entity in the knowledge graph",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "properties": {"type": "object"},
                    },
                    "required": ["name", "type"],
                },
            ),
            Tool(
                name="graph-connect",
                description="Create a relationship between entities",
                input_schema={
                    "type": "object",
                    "properties": {
                        "from_id": {"type": "string"},
                        "to_id": {"type": "string"},
                        "relation": {"type": "string"},
                    },
                    "required": ["from_id", "to_id", "relation"],
                },
            ),
        ]

        for tool in memory_tools:
            self._tools[tool.name] = tool

    async def _discover_a2a_tools(self) -> None:
        """Discover A2A tools from /v1/mcp/a2a endpoint."""
        if not self._session:
            return

        try:
            async with self._session.get(f"{self.base_url}/v1/mcp/a2a") as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to discover A2A tools: {resp.status}")
                    return

                data = await resp.json()
                tools_list = data.get("tools", [])

                for t in tools_list:
                    tool = Tool(
                        name=t["name"],
                        description=t.get("description", ""),
                        input_schema={"type": "object", "properties": {}},
                        server="a2a",
                    )
                    self._tools[tool.name] = tool

                logger.info(f"Discovered {len(tools_list)} A2A tools")
        except Exception as e:
            logger.warning(f"Failed to discover A2A tools: {e}")

    @property
    def tools(self) -> List[Tool]:
        """Get all available tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Execute a tool."""
        if not self._session:
            raise RuntimeError("Gateway not connected. Call connect() first.")

        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Route to correct MCP endpoint based on tool server
        if tool.server == "a2a":
            endpoint = f"{self.base_url}/v1/mcp/a2a"
            request = {
                "jsonrpc": "2.0",
                "method": tool_name,
                "params": arguments,
                "id": 1,
            }
        else:
            endpoint = f"{self.base_url}/v1/mcp/memory-graph"
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 1,
            }

        async with self._session.post(endpoint, json=request) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"MCP call failed: {error}")

            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")

            return data.get("result")

    # =========================================================================
    # Memory Operations (mem0 compatible)
    # =========================================================================

    async def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Add a memory to the knowledge graph."""
        args: Dict[str, Any] = {"content": content}
        if user_id:
            args["user_id"] = user_id
        if metadata:
            args["metadata"] = metadata
        return await self.execute("memory-add", args)

    async def search_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories by semantic similarity."""
        args: Dict[str, Any] = {"query": query, "limit": limit}
        if user_id:
            args["user_id"] = user_id

        result = await self.execute("memory-search", args)
        memories = result.get("memories", []) if isinstance(result, dict) else []

        return [
            Memory(
                id=m.get("id", ""),
                content=m.get("content", ""),
                created_at=m.get("created_at", ""),
                metadata=m.get("metadata"),
                relevance=m.get("relevance", 1.0),
            )
            for m in memories
        ]

    async def get_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Get relevant context for the current conversation."""
        args: Dict[str, Any] = {"query": query, "limit": limit}
        if user_id:
            args["user_id"] = user_id

        result = await self.execute("memory-get-context", args)
        return result.get("context", "") if isinstance(result, dict) else ""

    # =========================================================================
    # Checkpoint Operations (LangGraph compatible)
    # =========================================================================

    async def save_checkpoint(
        self,
        thread_id: str,
        state: Dict,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Save a conversation checkpoint."""
        args: Dict[str, Any] = {"thread_id": thread_id, "state": state}
        if metadata:
            args["metadata"] = metadata
        return await self.execute("checkpoint-save", args)

    async def get_checkpoint(
        self,
        thread_id: str,
    ) -> Optional[Dict]:
        """Get a conversation checkpoint."""
        result = await self.execute("checkpoint-get", {"thread_id": thread_id})
        return result if isinstance(result, dict) else None

    # =========================================================================
    # Entity/Graph Operations
    # =========================================================================

    async def create_entity(
        self,
        name: str,
        type: str,
        properties: Optional[Dict] = None,
    ) -> Dict:
        """Create an entity in the knowledge graph."""
        args: Dict[str, Any] = {"name": name, "type": type}
        if properties:
            args["properties"] = properties
        return await self.execute("entity-create", args)

    async def connect_entities(
        self,
        from_id: str,
        to_id: str,
        relation: str,
    ) -> Dict:
        """Create a relationship between entities."""
        return await self.execute("graph-connect", {
            "from_id": from_id,
            "to_id": to_id,
            "relation": relation,
        })


# =============================================================================
# Module-level Convenience
# =============================================================================

_gateway: Optional[Gateway] = None


async def connect(
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> Gateway:
    """Connect to the MCP gateway."""
    global _gateway
    if _gateway is None:
        _gateway = Gateway(api_key=api_key, base_url=base_url)
        await _gateway.connect()
    return _gateway


async def disconnect() -> None:
    """Disconnect from the MCP gateway."""
    global _gateway
    if _gateway is not None:
        await _gateway.disconnect()
        _gateway = None
