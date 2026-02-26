"""
Unju Realtime Model - LiveKit Compatible Voice Streaming

Provides a RealtimeModel implementation that's compatible with LiveKit's
AgentSession, connecting to the Unju realtime WebSocket endpoint which
implements the Gemini BidiGenerateContent protocol.

Usage:
    from livekit.agents import AgentSession
    import unju

    session = AgentSession(
        llm=unju.RealtimeModel(
            voice="asteria",
            instructions="You are a helpful assistant",
        ),
    )
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import audio as audio_utils, is_given

logger = logging.getLogger("unju.realtime")

# Audio constants matching Gemini protocol
INPUT_AUDIO_SAMPLE_RATE = 16000
INPUT_AUDIO_CHANNELS = 1
OUTPUT_AUDIO_SAMPLE_RATE = 24000
OUTPUT_AUDIO_CHANNELS = 1

# Available voices on Unju realtime
Voice = Literal[
    "asteria", "luna", "stella", "athena", "hera",
    "orion", "arcas", "perseus", "angus", "orpheus", "helios", "zeus"
]

DEFAULT_VOICE: Voice = "asteria"
DEFAULT_MODEL = "models/gemini-realtime"
DEFAULT_REALTIME_URL = "wss://realtime.unju.ai"


@dataclass
class _RealtimeOptions:
    """Configuration options for the realtime session."""
    model: str
    api_key: str
    base_url: str
    voice: str
    instructions: NotGivenOr[str]
    temperature: NotGivenOr[float]
    max_output_tokens: NotGivenOr[int]
    response_modalities: list[str]
    vad_enabled: bool
    start_of_speech_sensitivity: str
    end_of_speech_sensitivity: str
    prefix_padding_ms: int
    silence_duration_ms: int


@dataclass
class _ResponseGeneration:
    """Tracks an active response generation."""
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    response_id: str
    input_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    input_transcription: str = ""
    output_text: str = ""
    _created_timestamp: float = field(default_factory=time.time)
    _first_token_timestamp: float | None = None
    _completed_timestamp: float | None = None
    _done: bool = False

    def push_text(self, text: str) -> None:
        if self.output_text:
            self.output_text += text
        else:
            self.output_text = text
        self.text_ch.send_nowait(text)


# Type for client events we send
ClientEvent = dict[str, Any]


class RealtimeModel(llm.RealtimeModel):
    """
    Unju RealtimeModel - LiveKit compatible realtime voice model.

    Connects to the Unju realtime WebSocket endpoint which implements
    the Gemini BidiGenerateContent protocol.

    Usage:
        from livekit.agents import AgentSession
        import unju

        session = AgentSession(
            llm=unju.RealtimeModel(
                voice="asteria",
                instructions="You are a helpful assistant",
            ),
        )
    """

    def __init__(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        model: str = DEFAULT_MODEL,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        voice: Voice | str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[str]] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int] = NOT_GIVEN,
        vad_enabled: bool = True,
        start_of_speech_sensitivity: str = "SPEECH_SENSITIVITY_HIGH",
        end_of_speech_sensitivity: str = "SPEECH_SENSITIVITY_HIGH",
        prefix_padding_ms: int = 300,
        silence_duration_ms: int = 500,
    ) -> None:
        """
        Initialize the Unju RealtimeModel.

        Args:
            instructions: System instructions for the model
            model: Model identifier (default: models/gemini-realtime)
            api_key: Unju API key (or UNJU_API_KEY env var)
            base_url: WebSocket base URL (or UNJU_REALTIME_URL env var)
            voice: Voice for audio output (default: asteria)
            modalities: Response modalities ["TEXT", "AUDIO"]
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            vad_enabled: Enable voice activity detection
            start_of_speech_sensitivity: VAD start sensitivity
            end_of_speech_sensitivity: VAD end sensitivity
            prefix_padding_ms: VAD prefix padding
            silence_duration_ms: VAD silence duration for turn end
        """
        response_modalities = modalities if is_given(modalities) else ["TEXT", "AUDIO"]

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=vad_enabled,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output="AUDIO" in response_modalities,
                manual_function_calls=False,
            )
        )

        unju_api_key = api_key if is_given(api_key) else os.environ.get("UNJU_API_KEY")
        if not unju_api_key:
            raise ValueError(
                "API key required via api_key argument or UNJU_API_KEY environment variable"
            )

        unju_base_url = (
            base_url if is_given(base_url)
            else os.environ.get("UNJU_REALTIME_URL", DEFAULT_REALTIME_URL)
        )

        self._opts = _RealtimeOptions(
            model=model,
            api_key=unju_api_key,
            base_url=unju_base_url,
            voice=voice,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_modalities=response_modalities,
            vad_enabled=vad_enabled,
            start_of_speech_sensitivity=start_of_speech_sensitivity,
            end_of_speech_sensitivity=end_of_speech_sensitivity,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
        )

        self._sessions: weakref.WeakSet[RealtimeSession] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Unju"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        pass


class RealtimeSession(llm.RealtimeSession):
    """Manages a single realtime session with the Unju WebSocket endpoint."""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._opts = realtime_model._opts
        self._tools = llm.ToolContext.empty()
        self._tool_declarations: list[dict[str, Any]] = []
        self._chat_ctx = llm.ChatContext.empty()
        self._msg_ch: utils.aio.Chan[ClientEvent] = utils.aio.Chan()
        self._input_resampler: rtc.AudioResampler | None = None

        # Audio chunking (50ms chunks)
        self._bstream = audio_utils.AudioByteStream(
            INPUT_AUDIO_SAMPLE_RATE,
            INPUT_AUDIO_CHANNELS,
            samples_per_channel=INPUT_AUDIO_SAMPLE_RATE // 20,
        )

        # WebSocket connection
        self._http_session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        self._main_atask = asyncio.create_task(self._main_task(), name="unju-realtime-session")

        self._current_generation: _ResponseGeneration | None = None
        self._session_should_close = asyncio.Event()
        self._pending_generation_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._in_user_activity = False
        self._session_lock = asyncio.Lock()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def _send_client_event(self, event: ClientEvent) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Send audio data to the session."""
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                audio_data = base64.b64encode(nf.data.tobytes()).decode("utf-8")
                self._send_client_event({
                    "realtimeInput": {
                        "audio": {
                            "data": audio_data,
                            "sampleRate": INPUT_AUDIO_SAMPLE_RATE,
                        }
                    }
                })

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Send video frame to the session (not yet implemented)."""
        logger.warning("push_video not yet implemented for Unju realtime")

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Trigger a response generation."""
        if self._pending_generation_fut and not self._pending_generation_fut.done():
            logger.warning(
                "generate_reply called while another generation is pending, cancelling previous."
            )
            self._pending_generation_fut.cancel("Superseded by new generate_reply call")

        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        self._pending_generation_fut = fut

        if self._in_user_activity:
            self._send_client_event({"realtimeInput": {"activityEnd": {}}})
            self._in_user_activity = False

        # Send text instruction if provided
        if is_given(instructions):
            self._send_client_event({
                "clientContent": {
                    "turns": [
                        {"parts": [{"text": instructions}], "role": "model"},
                        {"parts": [{"text": "."}], "role": "user"},
                    ],
                    "turnComplete": True,
                }
            })

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError("generate_reply timed out waiting for generation_created event.")
                )
                if self._pending_generation_fut is fut:
                    self._pending_generation_fut = None

        timeout_handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: timeout_handle.cancel())

        return fut

    def start_user_activity(self) -> None:
        """Notify the model that user activity has started."""
        if not self._opts.vad_enabled:
            return

        if not self._in_user_activity:
            self._in_user_activity = True
            self._send_client_event({"realtimeInput": {"activityStart": {}}})

    def interrupt(self) -> None:
        """Interrupt the current generation."""
        self.start_user_activity()

    def commit_audio(self) -> None:
        """Commit the audio buffer (no-op for Gemini protocol)."""
        pass

    def clear_audio(self) -> None:
        """Clear the audio buffer (no-op for Gemini protocol)."""
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Truncate a message (not supported by Gemini protocol)."""
        logger.warning("truncate is not supported by the Unju Realtime API.")

    def send_tool_response(
        self,
        call_id: str,
        name: str,
        response: dict[str, Any],
    ) -> None:
        """Send a tool/function response back to the session.

        This should be called after executing a function from a tool call event.

        Args:
            call_id: The ID from the original function call
            name: The function name
            response: The result of the function execution
        """
        self._send_client_event({
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "name": name,
                    "response": response,
                }]
            }
        })

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        """Update session options."""
        if is_given(tool_choice):
            logger.warning("tool_choice is not supported by Unju Realtime API.")

    async def update_instructions(self, instructions: str) -> None:
        """Update system instructions (requires reconnect)."""
        if not is_given(self._opts.instructions) or self._opts.instructions != instructions:
            self._opts.instructions = instructions
            self._mark_restart_needed()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update the chat context."""
        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        """Update available tools."""
        new_declarations = self._convert_tools(tools)
        current_names = {d.get("name") for d in self._tool_declarations}
        new_names = {d.get("name") for d in new_declarations}

        if current_names != new_names:
            self._tool_declarations = new_declarations
            self._tools = llm.ToolContext(tools)
            self._mark_restart_needed()

    def _convert_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> list[dict]:
        """Convert LiveKit tools to Gemini function declarations."""
        declarations = []
        for tool in tools:
            if hasattr(tool, "metadata"):
                meta = tool.metadata
                declarations.append({
                    "name": meta.name,
                    "description": meta.description or "",
                    "parameters": self._convert_schema(meta.schema if hasattr(meta, "schema") else {}),
                })
        return declarations

    def _convert_schema(self, schema: dict) -> dict:
        """Convert JSON schema to Gemini schema format."""
        if not schema:
            return {"type": "OBJECT", "properties": {}}

        type_map = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }

        result = {}
        if "type" in schema:
            result["type"] = type_map.get(schema["type"], "STRING")
        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if "items" in schema:
            result["items"] = self._convert_schema(schema["items"])
        if "properties" in schema:
            result["properties"] = {
                k: self._convert_schema(v) for k, v in schema["properties"].items()
            }
        if "required" in schema:
            result["required"] = schema["required"]

        return result

    def _mark_restart_needed(self) -> None:
        """Signal that the session needs to reconnect."""
        if not self._session_should_close.is_set():
            self._session_should_close.set()
            self._msg_ch = utils.aio.Chan()

    async def aclose(self) -> None:
        """Close the session."""
        self._msg_ch.close()
        self._session_should_close.set()

        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

        if self._ws:
            await self._ws.close()
        if self._http_session:
            await self._http_session.close()

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            self._pending_generation_fut.cancel("Session closed")

        if self._current_generation:
            self._mark_current_generation_done()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """Main task that manages the WebSocket connection."""
        while not self._msg_ch.closed:
            self._session_should_close.clear()

            try:
                logger.debug("connecting to Unju Realtime API...")
                await self._connect_and_run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unju Realtime API error: {e}", exc_info=e)
                if not self._msg_ch.closed:
                    self._emit_error(e, recoverable=True)
                    await asyncio.sleep(1.0)  # Brief delay before retry

    async def _connect_and_run(self) -> None:
        """Connect to WebSocket and run send/receive loops."""
        self._http_session = aiohttp.ClientSession()

        # Build WebSocket URL with session ID and auth
        session_id = utils.shortuuid("sess_")
        ws_url = f"{self._opts.base_url}/ws/v1/realtime?session_id={session_id}&key={self._opts.api_key}"
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "X-Client-Name": "unju-agent",
        }

        self._ws = await self._http_session.ws_connect(
            ws_url,
            headers=headers,
            heartbeat=30.0,
        )

        # Send setup message
        await self._send_setup()

        # Run send and receive tasks
        send_task = asyncio.create_task(self._send_task(), name="unju-realtime-send")
        recv_task = asyncio.create_task(self._recv_task(), name="unju-realtime-recv")
        restart_wait = asyncio.create_task(
            self._session_should_close.wait(), name="unju-restart-wait"
        )

        done, pending = await asyncio.wait(
            [send_task, recv_task, restart_wait],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            if task is not restart_wait and task.exception():
                logger.error(f"error in task {task.get_name()}: {task.exception()}")

        for task in pending:
            await utils.aio.cancel_and_wait(task)

        await self._ws.close()
        await self._http_session.close()

    async def _send_setup(self) -> None:
        """Send the initial setup message matching Gemini BidiGenerateContentSetup."""
        generation_config: dict[str, Any] = {
            "responseModalities": self._opts.response_modalities,
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": self._opts.voice
                    }
                }
            }
        }

        if is_given(self._opts.temperature):
            generation_config["temperature"] = self._opts.temperature

        if is_given(self._opts.max_output_tokens):
            generation_config["maxOutputTokens"] = self._opts.max_output_tokens

        setup_payload: dict[str, Any] = {
            "model": self._opts.model,
            "generationConfig": generation_config,
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "disabled": not self._opts.vad_enabled,
                    "startOfSpeechSensitivity": self._opts.start_of_speech_sensitivity,
                    "endOfSpeechSensitivity": self._opts.end_of_speech_sensitivity,
                    "prefixPaddingMs": self._opts.prefix_padding_ms,
                    "silenceDurationMs": self._opts.silence_duration_ms,
                }
            },
        }

        # Add system instruction if provided
        if is_given(self._opts.instructions):
            setup_payload["systemInstruction"] = {
                "parts": [{"text": self._opts.instructions}],
                "role": "user",
            }

        # Add tools if configured
        if self._tool_declarations:
            setup_payload["tools"] = [{"functionDeclarations": self._tool_declarations}]

        # Enable transcription
        setup_payload["inputAudioTranscription"] = {}
        setup_payload["outputAudioTranscription"] = {}

        await self._ws.send_json({"setup": setup_payload})
        logger.debug(f"Sent setup message for model {self._opts.model}")

    async def _send_task(self) -> None:
        """Task that sends messages from the queue to WebSocket."""
        try:
            async for msg in self._msg_ch:
                if self._session_should_close.is_set() or not self._ws:
                    break
                await self._ws.send_json(msg)
        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in send task: {e}", exc_info=e)
                self._mark_restart_needed()

    async def _recv_task(self) -> None:
        """Task that receives messages from WebSocket."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_server_message(data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary audio data
                    pass
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
        except Exception as e:
            if not self._session_should_close.is_set():
                logger.error(f"error in receive task: {e}", exc_info=e)
                self._mark_restart_needed()
        finally:
            self._mark_current_generation_done()

    async def _handle_server_message(self, data: dict) -> None:
        """Handle a message from the server."""
        # Setup complete
        if "setupComplete" in data:
            logger.info("Unju realtime session setup complete")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        # Server content (text, audio, transcriptions)
        if "serverContent" in data:
            content = data["serverContent"]

            # Check if this is a new generation
            if not self._current_generation or self._current_generation._done:
                if content.get("interrupted"):
                    self._handle_input_speech_started()
                if self._is_new_generation(content):
                    self._start_new_generation()

            if self._current_generation:
                self._handle_server_content(content)

        # Tool calls
        if "toolCall" in data:
            if not self._current_generation or self._current_generation._done:
                self._start_new_generation()
            self._handle_tool_calls(data["toolCall"])

        # Tool call cancellation
        if "toolCallCancellation" in data:
            ids = data["toolCallCancellation"].get("ids", [])
            logger.warning(f"Tool calls cancelled: {ids}")

        # Go away (session ending)
        if "goAway" in data:
            logger.warning(f"Server indicates disconnection: {data['goAway']}")
            self._mark_restart_needed()

    def _is_new_generation(self, content: dict) -> bool:
        """Check if this content represents a new generation."""
        if content.get("modelTurn"):
            return True
        if content.get("inputTranscription", {}).get("text"):
            return True
        if content.get("outputTranscription", {}).get("text"):
            return True
        return False

    def _start_new_generation(self) -> None:
        """Start tracking a new response generation."""
        if self._current_generation and not self._current_generation._done:
            logger.warning("starting new generation while another is active")
            self._mark_current_generation_done()

        response_id = utils.shortuuid("GR_")
        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            response_id=response_id,
            input_id=utils.shortuuid("GI_"),
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
        )

        if not self._realtime_model.capabilities.audio_output:
            self._current_generation.audio_ch.close()

        msg_modalities: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
        msg_modalities.set_result(
            ["audio", "text"] if self._realtime_model.capabilities.audio_output else ["text"]
        )

        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=response_id,
                text_stream=self._current_generation.text_ch,
                audio_stream=self._current_generation.audio_ch,
                modalities=msg_modalities,
            )
        )

        generation_event = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
            response_id=response_id,
        )

        if self._pending_generation_fut and not self._pending_generation_fut.done():
            generation_event.user_initiated = True
            self._pending_generation_fut.set_result(generation_event)
            self._pending_generation_fut = None
        else:
            self._handle_input_speech_started()

        self.emit("generation_created", generation_event)

    def _handle_server_content(self, content: dict) -> None:
        """Handle server content (text, audio, transcriptions)."""
        gen = self._current_generation
        if not gen:
            return

        # Model turn with text/audio
        if model_turn := content.get("modelTurn"):
            for part in model_turn.get("parts", []):
                if text := part.get("text"):
                    gen.push_text(text)
                if inline_data := part.get("inlineData"):
                    if not gen._first_token_timestamp:
                        gen._first_token_timestamp = time.time()
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(inline_data.get("data", ""))
                    if audio_bytes:
                        frame = rtc.AudioFrame(
                            data=audio_bytes,
                            sample_rate=OUTPUT_AUDIO_SAMPLE_RATE,
                            num_channels=OUTPUT_AUDIO_CHANNELS,
                            samples_per_channel=len(audio_bytes) // (2 * OUTPUT_AUDIO_CHANNELS),
                        )
                        gen.audio_ch.send_nowait(frame)

        # Input transcription
        if input_trans := content.get("inputTranscription"):
            if text := input_trans.get("text"):
                if not gen.input_transcription:
                    text = text.lstrip()
                gen.input_transcription += text
                self.emit(
                    "input_audio_transcription_completed",
                    llm.InputTranscriptionCompleted(
                        item_id=gen.input_id,
                        transcript=gen.input_transcription,
                        is_final=False,
                    ),
                )

        # Output transcription
        if output_trans := content.get("outputTranscription"):
            if text := output_trans.get("text"):
                gen.push_text(text)

        # Generation/turn complete
        if content.get("generationComplete") or content.get("turnComplete"):
            gen._completed_timestamp = time.time()

        if content.get("interrupted"):
            self._handle_input_speech_started()

        if content.get("turnComplete"):
            self._mark_current_generation_done()

    def _handle_tool_calls(self, tool_call: dict) -> None:
        """Handle function/tool calls from the server."""
        gen = self._current_generation
        if not gen:
            return

        for fnc_call in tool_call.get("functionCalls", []):
            gen.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=fnc_call.get("id", utils.shortuuid("fnc-call-")),
                    name=fnc_call.get("name", ""),
                    arguments=json.dumps(fnc_call.get("args", {})),
                )
            )
        self._mark_current_generation_done()

    def _mark_current_generation_done(self) -> None:
        """Mark the current generation as complete."""
        if not self._current_generation or self._current_generation._done:
            return

        self._handle_input_speech_stopped()

        gen = self._current_generation

        # Emit final transcription
        if gen.input_transcription:
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=gen.input_id,
                    transcript=gen.input_transcription,
                    is_final=True,
                ),
            )
            self._chat_ctx.add_message(
                role="user",
                content=gen.input_transcription,
                id=gen.input_id,
            )

        if gen.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=gen.output_text,
                id=gen.response_id,
            )

        # Close channels
        if not gen.text_ch.closed:
            gen.text_ch.close()
        if not gen.audio_ch.closed:
            gen.audio_ch.close()
        gen.function_ch.close()
        gen.message_ch.close()
        gen._done = True

    def _handle_input_speech_started(self) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_speech_stopped(self) -> None:
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        """Resample audio to the required input format."""
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != INPUT_AUDIO_SAMPLE_RATE
            or frame.num_channels != INPUT_AUDIO_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_AUDIO_SAMPLE_RATE,
                num_channels=INPUT_AUDIO_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )
