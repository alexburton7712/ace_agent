import logging
import time

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from tools.tool import tool

from .client import SearchProviderError, create_search_client
from .fetch import fetch_document, fetch_documents

logger = logging.getLogger(__name__)


class ToolParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebSearchParameters(ToolParameters):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=10)


class WebFetchParameters(ToolParameters):
    url: HttpUrl


class WebFetchManyParameters(ToolParameters):
    urls: list[HttpUrl] = Field(min_length=1, max_length=10)


@tool(WebSearchParameters)
async def web_search(query: str, max_results: int = 10) -> list[dict] | dict:
    """Search the live web and return compact, structured results."""
    started = time.monotonic()
    try:
        results = await create_search_client().search(query, max_results)
    except SearchProviderError as exc:
        return {"success": False, "error": "search_provider_unavailable", "message": str(exc)}
    logger.info("web_search query=%r results=%d duration=%.2fs", query, len(results), time.monotonic() - started)
    return [result.to_dict() for result in results]


@tool(WebFetchParameters)
async def web_fetch(url: HttpUrl) -> dict:
    """Download a webpage and return bounded readable text with source metadata."""
    return await fetch_document(str(url))


@tool(WebFetchManyParameters)
async def web_fetch_many(urls: list[HttpUrl]) -> list[dict]:
    """Fetch up to ten webpages concurrently and return readable source documents."""
    return await fetch_documents([str(url) for url in urls])
