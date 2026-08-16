import asyncio
import os

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.renderers import ChatParams

# AWQ-quantized checkpoint: ~4-bit weights instead of bf16, roughly 1/3
# the VRAM footprint. Needed because full bf16 Qwen3-4B (~8GB) doesn't
# fit on an 8GB card alongside the KV cache.
MODEL_NAME = "Qwen/Qwen3-4B-AWQ"
PROMPT_FILE = "assistant_prompt.md"

# Load Ace's system prompt from the markdown file
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# AsyncLLMEngine gives us token-by-token streaming without a separate
# server process.
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.75,
    max_model_len=8192,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)

_request_counter = 0


def _next_request_id() -> str:
    global _request_counter
    _request_counter += 1
    return str(_request_counter)


async def generate_stream(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
    )

    # vLLM 0.27.1 renderer handles:
    #   1. chat-template formatting
    #   2. tokenization
    #   3. creation of the EngineInput expected by AsyncLLM
    chat_params = ChatParams(
    chat_template_kwargs={"add_generation_prompt": True},
)

    _, engine_inputs = await engine.renderer.render_chat_async(
        [messages],
        chat_params,
    )

    request_id = _next_request_id()

    results_generator = engine.generate(
        engine_inputs[0],
        sampling_params,
        request_id,
    )

    previous_text = ""

    async for request_output in results_generator:
        current_text = request_output.outputs[0].text
        new_text = current_text[len(previous_text):]
        previous_text = current_text

        if new_text:
            yield new_text


async def main():
    while True:
        prompt = await asyncio.to_thread(input, "You: ")

        print("Ace:", end=" ", flush=True)

        async for chunk in generate_stream(prompt):
            print(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
        os._exit(0)