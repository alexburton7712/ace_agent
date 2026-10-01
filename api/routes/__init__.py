from fastapi import APIRouter

from api.routes.audio import router as audio_router
from api.routes.conversation import router as conversation_router
from api.routes.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(audio_router)
router.include_router(conversation_router)

__all__ = ["router"]
