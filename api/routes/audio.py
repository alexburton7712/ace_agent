import logging
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from agent import NoSpeechDetected
from api.dependencies import agent
from stt import TranscriptionError

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_AUDIO_BYTES = int(os.getenv("ACE_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))


class AudioResponse(BaseModel):
    transcription: str
    response: str


@router.post("/audio", response_model=AudioResponse)
async def upload_audio(file: UploadFile = File(...)) -> AudioResponse:
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    try:
        transcription, response = await agent.process_audio(audio)
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

    return AudioResponse(transcription=transcription, response=response)
