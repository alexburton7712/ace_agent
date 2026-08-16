import asyncio
import json
import re
import uuid

from llm import generate_stream
from tools.registry import execute_tool, get_tool_schemas


class Agent:
    def __init__(self):
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
            prompt = await asyncio.to_thread(input, "You: ")

            self.messages.append({
                "role": "user",
                "content": prompt,
            })

            await self.respond()

    async def respond(self):
        print("Ace:", end=" ", flush=True)

        # Allow several tool calls before giving up
        for _ in range(8):
            response = ""

            showing_thinking = False
            showing_response = False
            response_buffer = ""

            async for chunk_type, chunk in generate_stream(
                self.messages,
                tools=get_tool_schemas(),
            ):
                if chunk_type == "thinking":
                    if not showing_thinking:
                        print("<think>")
                        showing_thinking = True

                    print(chunk, end="", flush=True)

                elif chunk_type == "response":
                    response += chunk
                    response_buffer += chunk

                    # Qwen may be starting a tool call.
                    # Don't print it to the user.
                    stripped = response_buffer.lstrip()

                    if (
                        "<tool_call>".startswith(stripped)
                        or stripped.startswith("<tool_call>")
                    ):
                        continue

                    # Normal assistant response
                    if not showing_response:
                        if showing_thinking:
                            print("\n</think>")

                        showing_response = True

                    print(response_buffer, end="", flush=True)
                    response_buffer = ""

            tool_calls = self._parse_tool_calls(response)

            # ----------------------------------------
            # Normal response
            # ----------------------------------------
            if not tool_calls:
                # Print anything still buffered
                if response_buffer:
                    if not showing_response and showing_thinking:
                        print("\n</think>")

                    print(response_buffer, end="", flush=True)

                print()

                self.messages.append({
                    "role": "assistant",
                    "content": response,
                })

                return

            # Close thinking display if necessary
            if showing_thinking:
                print("\n</think>")

            # ----------------------------------------
            # Tool call
            # ----------------------------------------

            assistant_tool_calls = []

            for tool_call in tool_calls:
                call_id = f"call_{uuid.uuid4().hex[:12]}"

                name = tool_call["name"]
                arguments = tool_call.get("arguments", {})

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                assistant_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                })

            # Save the assistant's tool request
            self.messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": assistant_tool_calls,
            })

            # Execute each requested tool
            for tool_call in assistant_tool_calls:
                name = tool_call["function"]["name"]
                arguments = json.loads(
                    tool_call["function"]["arguments"]
                )

                result = await execute_tool(
                    name,
                    arguments,
                )

                # Give result back to Qwen
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result),
                })

            # Loop again.
            # Qwen now sees the tool result and can answer the user.

        raise RuntimeError("Too many consecutive tool calls.")

    def _parse_tool_calls(self, text: str):
        calls = []

        matches = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL,
        )

        for match in matches:
            try:
                calls.append(json.loads(match))
            except json.JSONDecodeError:
                continue

        return calls