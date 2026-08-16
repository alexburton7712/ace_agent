import os

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.renderers import ChatParams

from settings import PROFILES

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


# --------------------------------------------------
# Request ID handling
# --------------------------------------------------

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

        max_length = min(
            len(text),
            len(tag) - 1,
        )

        for length in range(max_length, 0, -1):
            if text.endswith(tag[:length]):
                return length

        return 0

    def feed(self, text: str):
        """
        Parse newly generated text.

        Returns a list containing:

            ("thinking", text)
            ("response", text)

        The <think> and </think> tags themselves are
        removed and are not emitted.
        """

        self.buffer += text

        chunks = []

        while self.buffer:

            # ------------------------------------------
            # Currently inside <think>
            # ------------------------------------------

            if self.inside_think:
                close_index = self.buffer.find(
                    self.CLOSE_TAG
                )

                # Found </think>
                if close_index != -1:

                    if close_index > 0:
                        chunks.append(
                            (
                                "thinking",
                                self.buffer[:close_index],
                            )
                        )

                    self.buffer = self.buffer[
                        close_index
                        + len(self.CLOSE_TAG):
                    ]

                    self.inside_think = False

                    continue

                # Closing tag may be split across
                # multiple streaming chunks.
                keep = self._partial_tag_length(
                    self.buffer,
                    self.CLOSE_TAG,
                )

                safe_length = len(self.buffer) - keep

                if safe_length > 0:
                    chunks.append(
                        (
                            "thinking",
                            self.buffer[:safe_length],
                        )
                    )

                self.buffer = self.buffer[
                    safe_length:
                ]

                break

            # ------------------------------------------
            # Currently outside <think>
            # ------------------------------------------

            else:
                open_index = self.buffer.find(
                    self.OPEN_TAG
                )

                # Found <think>
                if open_index != -1:

                    if open_index > 0:
                        chunks.append(
                            (
                                "response",
                                self.buffer[:open_index],
                            )
                        )

                    self.buffer = self.buffer[
                        open_index
                        + len(self.OPEN_TAG):
                    ]

                    self.inside_think = True

                    continue

                # Opening tag may be split across
                # multiple streaming chunks.
                keep = self._partial_tag_length(
                    self.buffer,
                    self.OPEN_TAG,
                )

                safe_length = len(self.buffer) - keep

                if safe_length > 0:
                    chunks.append(
                        (
                            "response",
                            self.buffer[:safe_length],
                        )
                    )

                self.buffer = self.buffer[
                    safe_length:
                ]

                break

        return chunks

    def finish(self):
        """
        Flush anything remaining in the parser when
        generation ends.
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

        return [
            (
                chunk_type,
                remaining,
            )
        ]


# --------------------------------------------------
# Generation
# --------------------------------------------------

async def generate_stream(messages, tools=None):
    """
    Generate a streaming response from Qwen.

    Yields:

        ("thinking", text)
        ("response", text)

    Thinking is emitted so the UI / terminal can display
    it, but it is up to the agent layer whether thinking
    gets saved into conversation history.
    """

    sampling_params = SamplingParams(
        max_tokens=PROFILE["max_tokens"],
        temperature=PROFILE["temperature"],
    )

    chat_params = ChatParams(
        chat_template_kwargs={
            "add_generation_prompt": True,
            "tools": tools or [],
        },
    )

    # Render the conversation into Qwen's chat template.
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

        current_text = (
            request_output
            .outputs[0]
            .text
        )

        # vLLM returns the entire generated response
        # each iteration, rather than only the new token.
        #
        # Strip off everything we have already seen.
        new_text = current_text[
            len(previous_text):
        ]

        previous_text = current_text

        if not new_text:
            continue

        # Split streamed output into thinking vs
        # user-visible response chunks.
        for chunk_type, chunk in parser.feed(
            new_text
        ):
            if chunk:
                yield (
                    chunk_type,
                    chunk,
                )

    # Flush any text still buffered after generation
    # finishes.
    for chunk_type, chunk in parser.finish():
        if chunk:
            yield (
                chunk_type,
                chunk,
            )