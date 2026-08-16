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


# --------------------------------------------------
# Qwen thinking stream parser
# --------------------------------------------------

class ThinkStreamParser:
    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self):
        self.buffer = ""
        self.inside_think = False

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        """
        Find how much of the end of `text` could be the
        beginning of `tag`.

        This prevents streaming chunks like "<thi" from
        accidentally being emitted before "nk>" arrives.
        """
        max_length = min(len(text), len(tag) - 1)

        for length in range(max_length, 0, -1):
            if text.endswith(tag[:length]):
                return length

        return 0

    def feed(self, text: str):
        """
        Returns a list of:

            ("thinking", text)
            ("response", text)

        <think> tags themselves are not included.
        """
        self.buffer += text
        chunks = []

        while self.buffer:
            if self.inside_think:
                close_index = self.buffer.find(self.CLOSE_TAG)

                # Found </think>
                if close_index != -1:
                    if close_index > 0:
                        chunks.append(
                            ("thinking", self.buffer[:close_index])
                        )

                    self.buffer = self.buffer[
                        close_index + len(self.CLOSE_TAG):
                    ]

                    self.inside_think = False
                    continue

                # Closing tag may be split across stream chunks
                keep = self._partial_tag_length(
                    self.buffer,
                    self.CLOSE_TAG,
                )

                safe_length = len(self.buffer) - keep

                if safe_length > 0:
                    chunks.append(
                        ("thinking", self.buffer[:safe_length])
                    )

                self.buffer = self.buffer[safe_length:]
                break

            else:
                open_index = self.buffer.find(self.OPEN_TAG)

                # Found <think>
                if open_index != -1:
                    if open_index > 0:
                        chunks.append(
                            ("response", self.buffer[:open_index])
                        )

                    self.buffer = self.buffer[
                        open_index + len(self.OPEN_TAG):
                    ]

                    self.inside_think = True
                    continue

                # Opening tag may be split across stream chunks
                keep = self._partial_tag_length(
                    self.buffer,
                    self.OPEN_TAG,
                )

                safe_length = len(self.buffer) - keep

                if safe_length > 0:
                    chunks.append(
                        ("response", self.buffer[:safe_length])
                    )

                self.buffer = self.buffer[safe_length:]
                break

        return chunks

    def finish(self):
        """
        Flush anything left when generation ends.
        """
        if not self.buffer:
            return []

        chunk_type = (
            "thinking"
            if self.inside_think
            else "response"
        )

        remaining = self.buffer
        self.buffer = ""

        return [(chunk_type, remaining)]


# --------------------------------------------------
# Generation
# --------------------------------------------------

async def generate_stream(messages):
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
    parser = ThinkStreamParser()

    async for request_output in results_generator:
        current_text = request_output.outputs[0].text

        # vLLM gives us the full generated text so far,
        # so extract only the newly generated part.
        new_text = current_text[len(previous_text):]
        previous_text = current_text

        if not new_text:
            continue

        for chunk_type, chunk in parser.feed(new_text):
            if chunk:
                yield chunk_type, chunk

    # Flush any remaining buffered text
    for chunk_type, chunk in parser.finish():
        if chunk:
            yield chunk_type, chunk


# --------------------------------------------------
# Main conversation loop
# --------------------------------------------------

async def main():
    # This list now holds the entire conversation.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    while True:
        prompt = await asyncio.to_thread(input, "You: ")

        # Save user's message to conversation history
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        print("Ace:", end=" ", flush=True)

        response = ""

        showing_thinking = False
        showing_response = False

        async for chunk_type, chunk in generate_stream(messages):

            # ------------------------------------------
            # Thinking
            # ------------------------------------------
            if chunk_type == "thinking":
                if not showing_thinking:
                    print("<think>", flush=True)
                    showing_thinking = True

                print(chunk, end="", flush=True)

            # ------------------------------------------
            # Actual response
            # ------------------------------------------
            elif chunk_type == "response":
                if not showing_response:
                    if showing_thinking:
                        print("\n</think>")

                    showing_response = True

                # Show actual response
                print(chunk, end="", flush=True)

                # ONLY actual response gets saved
                response += chunk

        print()

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down.")
        os._exit(0)