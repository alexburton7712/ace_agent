import asyncio
import json
import logging

from llm import generate_stream
from stt import transcribe_wav
from tools.registry import (
    discover_tools,
    execute_tool,
    get_tool_schemas,
)

logger = logging.getLogger(__name__)


class NoSpeechDetected(RuntimeError):
    """Raised when STT returns no usable transcript."""


class Agent:
    def __init__(self):
        # Discover all @tool decorated functions once at startup.
        discover_tools()

        # Build the schemas once and give them to the LLM.
        self.tools = get_tool_schemas()

        self.messages = [
            {
                "role": "system",
                "content": self._load_system_prompt(),
            }
        ]
        # Keep conversation history ordered when multiple HTTP requests arrive.
        self._conversation_lock = asyncio.Lock()

    def _load_system_prompt(self):
        with open(
            "assistant_prompt.md",
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()

    async def run(self):
        while True:
            prompt = await asyncio.to_thread(
                input,
                "You: ",
            )

            response = await self.process(prompt)
            print(f"Ace: {response}")

    async def process(self, prompt: str) -> str:
        """Process one user message and return the agent's final response."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        async with self._conversation_lock:
            logger.info("User: %s", prompt)
            self.messages.append({
                "role": "user",
                "content": prompt,
            })

            return await self.respond()

    async def process_audio(self, audio: bytes) -> tuple[str, str]:
        """Transcribe audio, send the transcript to the LLM, and return both."""
        transcript = await transcribe_wav(audio)
        if not transcript:
            raise NoSpeechDetected("No speech was detected.")

        response = await self.process(transcript)
        return transcript, response

    async def respond(self) -> str:
        # Allow several rounds of tool calls before giving up.
        for round_number in range(1, 9):
            logger.debug(
                "Starting model round %d with %d messages",
                round_number,
                len(self.messages),
            )

            response = ""

            tool_calls = []

            async for chunk_type, chunk in generate_stream(
                self.messages,
                tools=self.tools,
            ):
                if chunk_type == "thinking":
                    logger.debug("Model thinking: %s", chunk)

                elif chunk_type == "response":
                    response += chunk

                elif chunk_type == "tool_calls":
                    tool_calls = chunk

            logger.debug(
                "Model round %d completed; tool calls found: %d",
                round_number,
                len(tool_calls),
            )

            # ----------------------------------------
            # Normal response
            # ----------------------------------------
            if not tool_calls:
                if not response.strip():
                    raise RuntimeError("LLM returned an empty response.")

                self.messages.append({
                    "role": "assistant",
                    "content": response,
                })

                logger.info("Ace: %s", response)
                return response

            # ----------------------------------------
            # Tool calls
            # ----------------------------------------

            # Save vLLM's structured tool requests into conversation history.
            self.messages.append({
                "role": "assistant",
                "content": response or None,
                "tool_calls": tool_calls,
            })

            # Execute every requested tool.
            for tool_call in tool_calls:
                name = (
                    tool_call["function"]["name"]
                )

                try:
                    arguments = json.loads(
                        tool_call["function"]["arguments"]
                    )
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Malformed tool arguments from model: %r",
                        tool_call["function"]["arguments"],
                    )
                    arguments = {}
                    result = {
                        "success": False,
                        "error": f"Invalid tool arguments: {exc}",
                    }
                else:
                    result = None

                logger.info(
                    "Tool call started: id=%s name=%s arguments=%s",
                    tool_call["id"],
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                )

                try:
                    if result is None:
                        result = await execute_tool(
                            name,
                            arguments,
                        )

                except Exception as exc:
                    logger.exception(
                        "Tool call failed: id=%s name=%s",
                        tool_call["id"],
                        name,
                    )

                    # Give the error back to Qwen instead of
                    # crashing the entire agent.
                    result = {
                        "success": False,
                        "error": str(exc),
                    }

                logger.info(
                    "Tool call completed: id=%s name=%s result=%s",
                    tool_call["id"],
                    name,
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                )

                # Give the result back to Qwen.
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result),
                })

            # Loop again.
            # Qwen now sees the tool results and can either:
            #   1. answer the user
            #   2. make another tool call

        raise RuntimeError(
            "Too many consecutive tool calls."
        )
