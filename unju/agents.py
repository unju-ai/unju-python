"""Agents — discover, connect, and interact with unju.ai agents.

Usage:
    agents = unju.agents.list()
    info = unju.agents.get("kimiko")
    token = unju.agents.connect("kimiko", user_id="user_123")
    card = unju.agents.card("kimiko")
"""
from __future__ import annotations

import httpx


class Agents:
    """Synchronous agents client."""

    def __init__(self, client: httpx.Client, *, api_version: str = "v1"):
        self._client = client
        self._v = api_version

    def list(self, *, include_coming_soon: bool = True) -> list[dict]:
        """List all available agents."""
        resp = self._client.get(
            f"/{self._v}/agents",
            params={"include_coming_soon": str(include_coming_soon).lower()},
        )
        resp.raise_for_status()
        return resp.json().get("agents", [])

    def get(self, agent_id: str) -> dict:
        """Get agent details."""
        resp = self._client.get(f"/{self._v}/agents/{agent_id}")
        resp.raise_for_status()
        return resp.json()

    def connect(
        self,
        agent_id: str,
        *,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Get a LiveKit connection token for an agent.

        Returns:
            dict with `serverUrl`, `participantToken`, `roomName`
        """
        payload: dict = {"agent_id": agent_id}
        if user_id:
            payload["user_id"] = user_id
        if metadata:
            payload["metadata"] = metadata

        resp = self._client.post(f"/{self._v}/agents/connect", json=payload)
        resp.raise_for_status()
        return resp.json()

    def card(self, agent_id: str) -> dict:
        """Get an agent's A2A Agent Card.

        Note: Uses the well-known path (not versioned).
        """
        resp = self._client.get(
            "/.well-known/agent.json",
            params={"agent": agent_id},
        )
        resp.raise_for_status()
        return resp.json()

    def trust(self, agent_id: str) -> dict:
        """Get an agent's trust score and reviews."""
        resp = self._client.get(f"/{self._v}/agents/{agent_id}/trust")
        resp.raise_for_status()
        return resp.json()


class AsyncAgents:
    """Async agents client."""

    def __init__(self, client: httpx.AsyncClient, *, api_version: str = "v1"):
        self._client = client
        self._v = api_version

    async def list(self, *, include_coming_soon: bool = True) -> list[dict]:
        resp = await self._client.get(
            f"/{self._v}/agents",
            params={"include_coming_soon": str(include_coming_soon).lower()},
        )
        resp.raise_for_status()
        return resp.json().get("agents", [])

    async def get(self, agent_id: str) -> dict:
        resp = await self._client.get(f"/{self._v}/agents/{agent_id}")
        resp.raise_for_status()
        return resp.json()

    async def connect(
        self,
        agent_id: str,
        *,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        payload: dict = {"agent_id": agent_id}
        if user_id:
            payload["user_id"] = user_id
        if metadata:
            payload["metadata"] = metadata

        resp = await self._client.post(f"/{self._v}/agents/connect", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def card(self, agent_id: str) -> dict:
        """Get an agent's A2A Agent Card.

        Note: Uses the well-known path (not versioned).
        """
        resp = await self._client.get(
            "/.well-known/agent.json",
            params={"agent": agent_id},
        )
        resp.raise_for_status()
        return resp.json()

    async def trust(self, agent_id: str) -> dict:
        resp = await self._client.get(f"/{self._v}/agents/{agent_id}/trust")
        resp.raise_for_status()
        return resp.json()
