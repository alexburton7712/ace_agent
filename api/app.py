import logging
import os

from fastapi import FastAPI

from api.routes import router

logging.basicConfig(
    level=os.getenv("ACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Ace Agent", version="1.0.0")
app.include_router(router, prefix="/api")
