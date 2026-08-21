import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from agent import Agent, NoSpeechDetected
from stt import TranscriptionError

logging.basicConfig(
    level=os.getenv("ACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ace Agent", version="1.0.0")
agent = Agent()

MAX_AUDIO_BYTES = int(os.getenv("ACE_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))


class AudioResponse(BaseModel):
    transcript: str
    response: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/audio", response_model=AudioResponse)
async def upload_audio(file: UploadFile = File(...)) -> AudioResponse:
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    try:
        transcript, response = await agent.process_audio(audio)
    except NoSpeechDetected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptionError as exc:
        logger.exception("Speech-to-text request failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio processing failed")
        raise HTTPException(status_code=502, detail="Audio processing failed.") from exc

    return AudioResponse(transcript=transcript, response=response)
