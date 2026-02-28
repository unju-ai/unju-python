"""Credits — check balance, usage, and purchase credits.

Usage:
    balance = unju.credits.balance()
    usage = unju.credits.usage(days=30)
"""
from __future__ import annotations

import httpx


class Credits:
    """Synchronous credits client."""

    def __init__(self, client: httpx.Client, *, api_version: str = "v1"):
        self._client = client
        self._v = api_version

    def balance(self) -> dict:
        """Get current credit balance.

        Returns:
            dict with `available`, `locked`, `earned_yield`, `total`
        """
        resp = self._client.get(f"/{self._v}/credits/balance")
        resp.raise_for_status()
        return resp.json()

    def usage(self, *, days: int = 30) -> dict:
        """Get credit usage history.

        Args:
            days: Number of days to look back
        """
        resp = self._client.get(f"/{self._v}/credits/usage", params={"days": days})
        resp.raise_for_status()
        return resp.json()

    def yield_info(self) -> dict:
        """Get yield information — current APY, total earned."""
        resp = self._client.get(f"/{self._v}/credits/yield")
        resp.raise_for_status()
        return resp.json()


class AsyncCredits:
    """Async credits client."""

    def __init__(self, client: httpx.AsyncClient, *, api_version: str = "v1"):
        self._client = client
        self._v = api_version

    async def balance(self) -> dict:
        resp = await self._client.get(f"/{self._v}/credits/balance")
        resp.raise_for_status()
        return resp.json()

    async def usage(self, *, days: int = 30) -> dict:
        resp = await self._client.get(f"/{self._v}/credits/usage", params={"days": days})
        resp.raise_for_status()
        return resp.json()

    async def yield_info(self) -> dict:
        resp = await self._client.get(f"/{self._v}/credits/yield")
        resp.raise_for_status()
        return resp.json()
