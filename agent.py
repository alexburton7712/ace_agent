import asyncio
import json
import logging
import re
import uuid

from llm import generate_stream
from tools.registry import (
    discover_tools,
    execute_tool,
    get_tool_schemas,
)

logger = logging.getLogger(__name__)


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

            self.messages.append({
                "role": "user",
                "content": prompt,
            })

            await self.respond()

    async def respond(self):
        print(
            "Ace:",
            end=" ",
            flush=True,
        )

        # Allow several rounds of tool calls before giving up.
        for round_number in range(1, 9):
            logger.debug(
                "Starting model round %d with %d messages",
                round_number,
                len(self.messages),
            )

            response = ""

            showing_thinking = False
            showing_response = False
            response_buffer = ""

            async for chunk_type, chunk in generate_stream(
                self.messages,
                tools=self.tools,
            ):
                if chunk_type == "thinking":
                    if not showing_thinking:
                        print("<think>")
                        showing_thinking = True

                    print(
                        chunk,
                        end="",
                        flush=True,
                    )

                elif chunk_type == "response":
                    response += chunk
                    response_buffer += chunk

                    # Qwen may be starting a tool call.
                    # Don't print tool-call markup to the user.
                    stripped = response_buffer.lstrip()

                    if (
                        "<tool_call>".startswith(stripped)
                        or stripped.startswith("<tool_call>")
                    ):
                        continue

                    # Normal assistant response.
                    if not showing_response:
                        if showing_thinking:
                            print("\n</think>")

                        showing_response = True

                    print(
                        response_buffer,
                        end="",
                        flush=True,
                    )

                    response_buffer = ""

            tool_calls = self._parse_tool_calls(response)

            logger.debug(
                "Model round %d completed; tool calls found: %d",
                round_number,
                len(tool_calls),
            )

            # ----------------------------------------
            # Normal response
            # ----------------------------------------
            if not tool_calls:
                if response_buffer:
                    if (
                        not showing_response
                        and showing_thinking
                    ):
                        print("\n</think>")

                    print(
                        response_buffer,
                        end="",
                        flush=True,
                    )

                print()

                self.messages.append({
                    "role": "assistant",
                    "content": response,
                })

                return

            # Close thinking display if necessary.
            if showing_thinking:
                print("\n</think>")

            # ----------------------------------------
            # Tool calls
            # ----------------------------------------

            assistant_tool_calls = []

            for tool_call in tool_calls:
                call_id = (
                    f"call_{uuid.uuid4().hex[:12]}"
                )

                name = tool_call["name"]
                arguments = tool_call.get(
                    "arguments",
                    {},
                )

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                assistant_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            arguments
                        ),
                    },
                })

            # Save Qwen's tool requests into conversation history.
            self.messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": assistant_tool_calls,
            })

            # Execute every requested tool.
            for tool_call in assistant_tool_calls:
                name = (
                    tool_call["function"]["name"]
                )

                arguments = json.loads(
                    tool_call["function"]["arguments"]
                )

                logger.info(
                    "Tool call started: id=%s name=%s arguments=%s",
                    tool_call["id"],
                    name,
                    json.dumps(arguments, ensure_ascii=False),
                )

                try:
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

    def _parse_tool_calls(
        self,
        text: str,
    ):
        calls = []

        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL,
        )

        for match in matches:
            try:
                calls.append(
                    json.loads(match)
                )

            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring malformed tool call payload: %r",
                    match,
                )
                continue

        return calls
