"""
unju — Python SDK for unju.ai

The AI agent platform SDK. Memory, realtime, agents, credits.

Usage:
    from unju import Unju

    unju = Unju(api_key="your-key")

    # Memory
    unju.memory.add("user_123", "Likes Vietnamese food")
    results = unju.memory.search("user_123", "food preferences")
    all_memories = unju.memory.list("user_123")

    # Agents
    agents = unju.agents.list()
    token = unju.agents.connect("kimiko", user_id="user_123")

    # Credits
    balance = unju.credits.balance()

    # Async
    async with unju:
        await unju.memory.async_add("user_123", "Prefers dark mode")
"""

__version__ = "0.1.0"
__all__ = ["Unju", "AsyncUnju", "Memory", "Agents", "Credits"]

from unju.client import Unju, AsyncUnju
from unju.memory import Memory
from unju.agents import Agents
from unju.credits import Credits

# LiveKit extras — lazy import to avoid hard dependency
# Use: from unju.livekit import LLM, STT, TTS, RealtimeModel
# Requires: pip install unju[livekit]
