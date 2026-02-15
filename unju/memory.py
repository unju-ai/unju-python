"""Memory — store, search, and retrieve user memories.

Compatible with mem0 API surface. Drop-in replacement.

Usage:
    unju.memory.add("user_123", "Prefers dark mode")
    unju.memory.add("user_123", [
        {"role": "user", "content": "I love Vietnamese food"},
        {"role": "assistant", "content": "Noted! You enjoy Vietnamese cuisine."},
    ])
    results = unju.memory.search("user_123", "food preferences")
    all_memories = unju.memory.list("user_123")
    unju.memory.delete("memory_id")
"""
from __future__ import annotations

from typing import Union
import httpx


class Memory:
    """Synchronous memory client."""

    def __init__(self, client: httpx.Client):
        self._client = client

    def add(
        self,
        user_id: str,
        content: Union[str, list[dict]],
        *,
        metadata: dict | None = None,
    ) -> dict:
        """Add a memory.

        Args:
            user_id: User identifier
            content: String or list of message dicts [{"role": "user", "content": "..."}]
            metadata: Optional metadata to attach
        """
        messages = (
            [{"role": "user", "content": content}]
            if isinstance(content, str)
            else content
        )
        payload = {"messages": messages, "user_id": user_id}
        if metadata:
            payload["metadata"] = metadata

        resp = self._client.post("/v1/memories/", json=payload)
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories by semantic similarity.

        Args:
            user_id: User identifier
            query: Search query
            limit: Max results
        """
        resp = self._client.post(
            "/v1/memories/search/",
            json={"query": query, "user_id": user_id, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def list(self, user_id: str) -> list[dict]:
        """Get all memories for a user."""
        resp = self._client.get("/v1/memories/", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json().get("results", [])

    def get(self, memory_id: str) -> dict:
        """Get a specific memory by ID."""
        resp = self._client.get(f"/v1/memories/{memory_id}/")
        resp.raise_for_status()
        return resp.json()

    def delete(self, memory_id: str) -> dict:
        """Delete a memory."""
        resp = self._client.delete(f"/v1/memories/{memory_id}/")
        resp.raise_for_status()
        return resp.json()

    def delete_all(self, user_id: str) -> dict:
        """Delete all memories for a user."""
        resp = self._client.delete("/v1/memories/", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json()


class AsyncMemory:
    """Async memory client."""

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    async def add(
        self,
        user_id: str,
        content: Union[str, list[dict]],
        *,
        metadata: dict | None = None,
    ) -> dict:
        messages = (
            [{"role": "user", "content": content}]
            if isinstance(content, str)
            else content
        )
        payload = {"messages": messages, "user_id": user_id}
        if metadata:
            payload["metadata"] = metadata

        resp = await self._client.post("/v1/memories/", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict]:
        resp = await self._client.post(
            "/v1/memories/search/",
            json={"query": query, "user_id": user_id, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def list(self, user_id: str) -> list[dict]:
        resp = await self._client.get("/v1/memories/", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def get(self, memory_id: str) -> dict:
        resp = await self._client.get(f"/v1/memories/{memory_id}/")
        resp.raise_for_status()
        return resp.json()

    async def delete(self, memory_id: str) -> dict:
        resp = await self._client.delete(f"/v1/memories/{memory_id}/")
        resp.raise_for_status()
        return resp.json()

    async def delete_all(self, user_id: str) -> dict:
        resp = await self._client.delete("/v1/memories/", params={"user_id": user_id})
        resp.raise_for_status()
        return resp.json()
