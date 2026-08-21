import asyncio
import io
import os
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import wav_to_chunks
from wyoming.client import AsyncClient
from wyoming.error import Error


class TranscriptionError(RuntimeError):
    """Raised when the configured speech-to-text service cannot transcribe."""


def _whisper_uri() -> str:
    uri = os.getenv("WHISPER_URI")
    if not uri:
        raise TranscriptionError("WHISPER_URI environment variable is not set.")
    return uri


async def transcribe_wav(audio: bytes) -> str:
    """Send a WAV recording to the configured Wyoming Whisper service."""
    if not audio:
        raise ValueError("The uploaded audio file is empty.")

    try:
        wav_io = io.BytesIO(audio)
        wav_file = wave.open(wav_io, "rb")
    except (EOFError, wave.Error) as exc:
        raise ValueError("Audio must be a valid WAV file.") from exc

    try:
        chunks = list(
            wav_to_chunks(
                wav_file,
                samples_per_chunk=1024,
                start_event=True,
                stop_event=True,
            )
        )
    finally:
        wav_file.close()

    try:
        async with asyncio.timeout(120):
            async with AsyncClient.from_uri(_whisper_uri()) as client:
                await client.write_event(
                    Transcribe(
                        name=os.getenv("WHISPER_MODEL") or None,
                        language=os.getenv("WHISPER_LANGUAGE") or None,
                    ).event()
                )

                for chunk in chunks:
                    await client.write_event(chunk.event())

                while True:
                    event = await client.read_event()
                    if event is None:
                        raise TranscriptionError(
                            "Whisper disconnected before returning a transcript."
                        )

                    if Transcript.is_type(event.type):
                        return Transcript.from_event(event).text.strip()

                    if Error.is_type(event.type):
                        error = Error.from_event(event)
                        raise TranscriptionError(
                            f"Whisper error ({error.code}): {error.text}"
                        )
    except TimeoutError as exc:
        raise TranscriptionError("Whisper transcription timed out.") from exc
    except (ConnectionError, OSError) as exc:
        raise TranscriptionError("Could not connect to Whisper.") from exc
