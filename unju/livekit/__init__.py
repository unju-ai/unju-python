"""
unju.livekit — LiveKit integration extras

Drop-in replacements for livekit-plugins-openai that route through api.unju.ai.

Requires: pip install unju[livekit]

Usage:
    from unju.livekit import LLM, STT, TTS, RealtimeModel
    from livekit.agents.voice_assistant import VoicePipelineAgent
    from livekit.agents import AgentSession

    # Pipeline agent (STT → LLM → TTS)
    agent = VoicePipelineAgent(
        llm=LLM(model="gpt-4o"),
        stt=STT(),
        tts=TTS(voice="alloy"),
    )

    # Realtime agent (Gemini-compatible WebSocket)
    session = AgentSession(
        llm=RealtimeModel(
            voice="asteria",
            instructions="You are a helpful assistant",
        ),
    )
"""

try:
    from unju.livekit.llm import LLM, STT, TTS
    from unju.livekit.realtime import RealtimeModel
except ImportError as e:
    raise ImportError(
        "unju[livekit] extras required. Install with: pip install unju[livekit]\n"
        f"Missing: {e}"
    ) from e

__all__ = ["LLM", "STT", "TTS", "RealtimeModel"]
