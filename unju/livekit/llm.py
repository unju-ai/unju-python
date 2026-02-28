"""
Unju LLM - LiveKit Compatible AI Models

Provides LLM, STT, and TTS classes that are drop-in replacements
for livekit-plugins-openai, but route through the Unju API.

Usage:
    import unju
    from livekit.agents.voice_assistant import VoicePipelineAgent

    agent = VoicePipelineAgent(
        llm=unju.LLM(model="gpt-4o"),
        stt=unju.STT(),
        tts=unju.TTS(voice="alloy"),
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from openai import AsyncOpenAI

# Import LiveKit OpenAI plugins and wrap them
from livekit.plugins import openai as livekit_openai


DEFAULT_BASE_URL = "https://api.unju.ai/v1"
SDK_VERSION = "0.1.0"


def _get_api_key() -> str:
    """Get Unju API key from environment."""
    key = os.environ.get("UNJU_API_KEY")
    if not key:
        raise ValueError(
            "UNJU_API_KEY environment variable required for unju.LLM"
        )
    return key


def _get_default_headers() -> Dict[str, str]:
    """Get default headers for Unju API requests.

    These headers help identify requests as legitimate SDK traffic
    and avoid WAF/bot detection blocking.
    """
    return {
        "User-Agent": f"unju-sdk/{SDK_VERSION}",
        "X-Client-Name": "unju-agent",
        "X-Client-Version": SDK_VERSION,
    }


def _create_unju_client(
    api_key: str,
    base_url: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> AsyncOpenAI:
    """Create an OpenAI AsyncClient configured for Unju API.

    Uses a custom httpx client with proper headers to avoid WAF blocking.
    """
    headers = _get_default_headers()
    if extra_headers:
        headers.update(extra_headers)

    # Create httpx client with custom headers
    http_client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(60.0, connect=15.0),
    )

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        default_headers=headers,
    )


class LLM(livekit_openai.LLM):
    """
    Unju LLM - OpenAI-compatible LLM routed through Unju API.

    Drop-in replacement for livekit.plugins.openai.LLM.

    Usage:
        import unju

        llm = unju.LLM(model="gpt-4o")

        # Or with custom settings
        llm = unju.LLM(
            model="gpt-4o",
            temperature=0.7,
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        **kwargs,
    ):
        api_key = api_key or _get_api_key()

        # Get any user-provided extra_headers
        extra_headers: Dict[str, Any] = kwargs.pop("extra_headers", None)

        # Create a custom OpenAI client with proper headers for Unju API
        # This avoids WAF blocking by using custom User-Agent and headers
        self._unju_client = _create_unju_client(api_key, base_url, extra_headers)
        self._model = model

        super().__init__(
            model=model,
            client=self._unju_client,
            **kwargs,
        )

    async def chat_with_tools(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Call the LLM with OpenAI-compatible tool definitions.

        This bypasses LiveKit's FunctionContext abstraction to allow
        passing raw MCP tool definitions directly.

        Args:
            messages: List of chat messages in OpenAI format
            tools: Optional list of tool definitions in OpenAI format
            temperature: Sampling temperature
            **kwargs: Additional OpenAI API parameters

        Returns:
            AsyncGenerator yielding OpenAI ChatCompletionChunk objects
        """
        request_params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        response = await self._unju_client.chat.completions.create(**request_params)

        async for chunk in response:
            yield chunk


class STT(livekit_openai.STT):
    """
    Unju STT - Speech-to-Text.

    Currently wraps OpenAI Whisper. Requires OPENAI_API_KEY
    until Unju API supports STT.

    Usage:
        import unju

        stt = unju.STT()
    """

    def __init__(self, **kwargs):
        # STT still uses OpenAI directly for now
        # TODO: Route through Unju when API supports it
        super().__init__(**kwargs)


class TTS(livekit_openai.TTS):
    """
    Unju TTS - Text-to-Speech.

    Currently wraps OpenAI TTS. Requires OPENAI_API_KEY
    until Unju API supports TTS.

    Usage:
        import unju

        tts = unju.TTS(voice="alloy")
    """

    def __init__(self, voice: str = "alloy", **kwargs):
        # TTS still uses OpenAI directly for now
        # TODO: Route through Unju when API supports it
        super().__init__(voice=voice, **kwargs)
