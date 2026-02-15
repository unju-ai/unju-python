# 🧠 unju

The Python SDK for [unju.ai](https://unju.ai) — the AI agent platform.

```bash
pip install unju
```

## Quick Start

```python
from unju import Unju

unju = Unju(api_key="your-api-key")

# Store memories
unju.memory.add("user_123", "Prefers dark mode and Vietnamese food")

# Search memories
results = unju.memory.search("user_123", "food preferences")
for r in results:
    print(r["memory"])

# List all memories
memories = unju.memory.list("user_123")

# Connect to an agent
token = unju.agents.connect("kimiko", user_id="user_123")
# Use token["serverUrl"] and token["participantToken"] with LiveKit

# Check credits
balance = unju.credits.balance()
print(f"Available: {balance['available']} credits")
```

## Async

```python
from unju import AsyncUnju

async with AsyncUnju(api_key="your-key") as unju:
    await unju.memory.add("user_123", "Learning Japanese")
    results = await unju.memory.search("user_123", "languages")
    agents = await unju.agents.list()
```

## Memory

Drop-in replacement for mem0. Same API surface, 100x cheaper.

```python
# Simple string
unju.memory.add("user_123", "Allergic to peanuts")

# Conversation format (mem0 compatible)
unju.memory.add("user_123", [
    {"role": "user", "content": "I'm learning Vietnamese"},
    {"role": "assistant", "content": "Great! I'll remember that."},
])

# Search
results = unju.memory.search("user_123", "allergies")

# Get all
all_memories = unju.memory.list("user_123")

# Delete
unju.memory.delete("memory_id")
unju.memory.delete_all("user_123")
```

## Agents

Discover and connect to AI agents.

```python
# List all agents
agents = unju.agents.list()
for agent in agents:
    print(f"{agent['icon']} {agent['name']} — {agent['description']}")

# Get agent details
kimiko = unju.agents.get("kimiko")

# Connect (get LiveKit token)
conn = unju.agents.connect("kimiko", user_id="user_123", metadata={
    "target_language": "vietnamese",
})
# conn = {"serverUrl": "wss://...", "participantToken": "...", "roomName": "..."}

# Get A2A Agent Card
card = unju.agents.card("kimiko")

# Get trust score
trust = unju.agents.trust("kimiko")
print(f"Trust: {trust['trust_score']}/100 ({trust['trust_tier']})")
```

## Credits

```python
# Check balance
balance = unju.credits.balance()
# {"available": 850, "locked": 50, "earned_yield": 12.5, "total": 912.5}

# Usage history
usage = unju.credits.usage(days=7)

# Yield info
yield_info = unju.credits.yield_info()
# {"current_apy": 1.5, "total_earned": 42.0, "next_distribution": "2024-01-15T00:00:00Z"}
```

## Configuration

```python
# Custom base URL (self-hosted)
unju = Unju(api_key="key", base_url="https://your-instance.com")

# Custom timeout
unju = Unju(api_key="key", timeout=60.0)
```

## License

MIT
