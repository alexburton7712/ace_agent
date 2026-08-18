import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from settings import GENERATION

BASE_URL = os.getenv("VLLM_BASE_URL")
API_KEY = os.getenv("VLLM_API_KEY")
MODEL = os.getenv("VLLM_MODEL")
ENABLE_THINKING = os.getenv("ACE_ENABLE_THINKING", "false").lower() in {
    "1", "true", "yes", "on",
}

if not BASE_URL:
    raise ValueError("VLLM_BASE_URL environment variable is not set.")
if not API_KEY:
    raise ValueError("VLLM_API_KEY environment variable is not set.")
if not MODEL:
    raise ValueError("VLLM_MODEL environment variable is not set.")

client = AsyncOpenAI(base_url=BASE_URL.rstrip("/"), api_key=API_KEY)

print(f"Model: {MODEL}")
print(f"vLLM server: {BASE_URL}")


async def generate_stream(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Stream response text, thinking text, and structured tool calls."""

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools or None,
        tool_choice="auto" if tools else None,
        max_tokens=GENERATION["max_tokens"],
        temperature=GENERATION["temperature"],
        top_p=GENERATION["top_p"],
        stream=True,
        extra_body={
            "top_k": GENERATION["top_k"],
            "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        },
    )

    pending_tool_calls: dict[int, dict[str, Any]] = {}

    async for event in stream:
        if not event.choices:
            continue

        delta = event.choices[0].delta
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            yield "thinking", reasoning

        if delta.content:
            yield "response", delta.content

        for tool_delta in delta.tool_calls or []:
            pending = pending_tool_calls.setdefault(
                tool_delta.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tool_delta.id:
                pending["id"] += tool_delta.id
            if tool_delta.function:
                if tool_delta.function.name:
                    pending["function"]["name"] += tool_delta.function.name
                if tool_delta.function.arguments:
                    pending["function"]["arguments"] += tool_delta.function.arguments

    if pending_tool_calls:
        yield "tool_calls", [
            pending_tool_calls[index]
            for index in sorted(pending_tool_calls)
        ]
