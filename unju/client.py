"""Core client — sync and async."""
from __future__ import annotations

import httpx
from typing import Optional

from unju.memory import Memory, AsyncMemory
from unju.agents import Agents, AsyncAgents
from unju.credits import Credits, AsyncCredits

DEFAULT_BASE_URL = "https://api.unju.ai"
DEFAULT_API_VERSION = "v1"


class Unju:
    """Synchronous unju.ai client.

    Usage:
        unju = Unju(api_key="your-key")
        unju.memory.add("user_123", "Loves ramen")

        # Pin a specific API version (future-proofing)
        unju = Unju(api_key="your-key", api_version="v2")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        api_version: str = DEFAULT_API_VERSION,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.api_version = api_version
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"unju-python/0.1.0",
            },
            timeout=timeout,
        )

        self.memory = Memory(self._client, api_version=api_version)
        self.agents = Agents(self._client, api_version=api_version)
        self.credits = Credits(self._client, api_version=api_version)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"Unju(base_url='{self._base_url}', api_version='{self.api_version}')"


class AsyncUnju:
    """Async unju.ai client.

    Usage:
        async with AsyncUnju(api_key="your-key") as unju:
            await unju.memory.add("user_123", "Loves ramen")

        # Pin a specific API version (future-proofing)
        async with AsyncUnju(api_key="your-key", api_version="v2") as unju:
            ...
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        api_version: str = DEFAULT_API_VERSION,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.api_version = api_version
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"unju-python/0.1.0",
            },
            timeout=timeout,
        )

        self.memory = AsyncMemory(self._client, api_version=api_version)
        self.agents = AsyncAgents(self._client, api_version=api_version)
        self.credits = AsyncCredits(self._client, api_version=api_version)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def __repr__(self):
        return f"AsyncUnju(base_url='{self._base_url}', api_version='{self.api_version}')"
