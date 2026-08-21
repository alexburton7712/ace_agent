import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("VLLM_BASE_URL", "http://example.invalid/v1")
os.environ.setdefault("VLLM_API_KEY", "test-key")
os.environ.setdefault("VLLM_MODEL", "test-model")

import httpx

import api


class AudioEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        transport = httpx.ASGITransport(app=api.app)
        self.client = httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health(self):
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    async def test_audio_transcribes_and_processes_text(self):
        process_audio = AsyncMock(
            return_value=(
                "Turn on the kitchen light.",
                "The kitchen light is on.",
            )
        )
        with patch.object(api.agent, "process_audio", new=process_audio):
            response = await self.client.post(
                "/audio",
                files={"file": ("command.wav", b"wav-data", "audio/wav")},
            )

        process_audio.assert_awaited_once_with(b"wav-data")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "transcript": "Turn on the kitchen light.",
                "response": "The kitchen light is on.",
            },
        )

    async def test_audio_rejects_invalid_wav(self):
        response = await self.client.post(
            "/audio",
            files={"file": ("command.wav", b"not-a-wave-file", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Audio must be a valid WAV file.")

    async def test_audio_requires_file(self):
        response = await self.client.post("/audio")

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
