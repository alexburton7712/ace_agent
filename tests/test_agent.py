import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("VLLM_BASE_URL", "http://example.invalid/v1")
os.environ.setdefault("VLLM_API_KEY", "test-key")
os.environ.setdefault("VLLM_MODEL", "test-model")

from agent import MAX_USER_TURNS, Agent, NoSpeechDetected


class AgentAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_is_transcribed_then_sent_to_llm(self):
        agent = Agent()

        with (
            patch(
                "agent.transcribe_wav",
                new=AsyncMock(return_value="Turn on the kitchen light."),
            ) as transcribe,
            patch.object(
                agent,
                "process",
                new=AsyncMock(return_value="The kitchen light is on."),
            ) as process,
        ):
            result = await agent.process_audio(b"wav-data")

        transcribe.assert_awaited_once_with(b"wav-data")
        process.assert_awaited_once_with("Turn on the kitchen light.")
        self.assertEqual(
            result,
            (
                "Turn on the kitchen light.",
                "The kitchen light is on.",
            ),
        )

    async def test_empty_transcript_is_not_sent_to_llm(self):
        agent = Agent()

        with (
            patch("agent.transcribe_wav", new=AsyncMock(return_value="")),
            patch.object(agent, "process", new=AsyncMock()) as process,
        ):
            with self.assertRaisesRegex(NoSpeechDetected, "No speech was detected"):
                await agent.process_audio(b"wav-data")

        process.assert_not_awaited()

    async def test_process_logs_user_and_agent_response(self):
        agent = Agent()

        async def generate_response(*args, **kwargs):
            yield "response", "The kitchen light is on."

        with (
            patch("agent.generate_stream", new=generate_response),
            self.assertLogs("agent", level="INFO") as logs,
        ):
            response = await agent.process("Turn on the kitchen light.")

        self.assertEqual(response, "The kitchen light is on.")
        combined_logs = "\n".join(logs.output)
        self.assertIn("User: Turn on the kitchen light.", combined_logs)
        self.assertIn("Ace: The kitchen light is on.", combined_logs)


class AgentHistoryTests(unittest.TestCase):
    def test_history_keeps_only_the_newest_complete_user_turns(self):
        agent = Agent()
        system_message = agent.messages[0]

        for turn in range(MAX_USER_TURNS + 1):
            agent.messages.extend([
                {"role": "user", "content": f"user-{turn}"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call-{turn}"}],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{turn}",
                    "content": f"result-{turn}",
                },
                {"role": "assistant", "content": f"assistant-{turn}"},
            ])

        agent._trim_conversation_history()

        self.assertIs(agent.messages[0], system_message)
        self.assertEqual(
            sum(message["role"] == "user" for message in agent.messages),
            MAX_USER_TURNS,
        )
        self.assertNotIn(
            "user-0",
            [message.get("content") for message in agent.messages],
        )
        self.assertNotIn(
            "call-0",
            [message.get("tool_call_id") for message in agent.messages],
        )
        self.assertEqual(agent.messages[1]["content"], "user-1")
        self.assertEqual(agent.messages[2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(agent.messages[3]["tool_call_id"], "call-1")

    def test_history_does_not_trim_at_the_limit(self):
        agent = Agent()
        for turn in range(MAX_USER_TURNS):
            agent.messages.extend([
                {"role": "user", "content": f"user-{turn}"},
                {"role": "assistant", "content": f"assistant-{turn}"},
            ])
        original_messages = agent.messages

        agent._trim_conversation_history()

        self.assertIs(agent.messages, original_messages)


if __name__ == "__main__":
    unittest.main()
