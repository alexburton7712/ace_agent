import asyncio
import os

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.renderers import ChatParams

from settings import PROFILES

PROMPT_FILE = "assistant_prompt.md"

# Load Ace's system prompt
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# --------------------------------------------------
# Select hardware/model profile
# --------------------------------------------------

PROFILE_NAME = os.getenv("ACE_PROFILE", "local").lower()

if PROFILE_NAME not in PROFILES:
    raise ValueError(
        f"Unknown ACE_PROFILE '{PROFILE_NAME}'. "
        f"Valid profiles: {', '.join(PROFILES.keys())}"
    )

PROFILE = PROFILES[PROFILE_NAME]

print(f"Profile: {PROFILE_NAME}")
print(f"Model: {PROFILE['model']}")


# --------------------------------------------------
# vLLM engine
# --------------------------------------------------

engine_args = AsyncEngineArgs(
    model=PROFILE["model"],
    quantization=PROFILE["quantization"],
    dtype=PROFILE["dtype"],
    gpu_memory_utilization=PROFILE["gpu_memory_utilization"],
    max_model_len=PROFILE["max_model_len"],
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
        max_tokens=PROFILE["max_tokens"],
        temperature=PROFILE["temperature"],
    )

    chat_params = ChatParams(
        chat_template_kwargs={
            "add_generation_prompt": True
        },
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