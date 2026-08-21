import asyncio
import os
from urllib.parse import urlparse

import httpx

from .parser import extract_document

DEFAULT_USER_AGENT = "AceWebResearch/1.0 (+personal assistant)"


def validate_web_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an absolute http:// or https:// URL.")


async def fetch_document(url: str) -> dict:
    validate_web_url(url)
    timeout = float(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", "20"))
    max_bytes = int(os.getenv("WEB_FETCH_MAX_BYTES", "5000000"))
    max_characters = int(os.getenv("WEB_FETCH_MAX_CHARACTERS", "20000"))
    headers = {"User-Agent": os.getenv("WEB_FETCH_USER_AGENT", DEFAULT_USER_AGENT)}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").casefold()
                if "html" not in content_type and "text/" not in content_type:
                    return {
                        "success": False,
                        "error": "page_blocked",
                        "message": f"Unsupported content type: {content_type or 'unknown'}",
                        "url": url,
                    }
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        return {
                            "success": False,
                            "error": "page_too_large",
                            "message": f"Page exceeded the {max_bytes}-byte download limit.",
                            "url": url,
                        }
                encoding = response.encoding or "utf-8"
                html = bytes(chunks).decode(encoding, errors="replace")
                final_url = str(response.url)
    except httpx.TimeoutException:
        return {"success": False, "error": "page_fetch_failed", "message": "Page request timed out.", "url": url}
    except httpx.HTTPError as exc:
        return {"success": False, "error": "page_fetch_failed", "message": str(exc), "url": url}

    try:
        return extract_document(final_url, html, max_characters)
    except Exception:
        return {
            "success": False,
            "error": "page_parse_failed",
            "message": "The page downloaded, but readable content could not be extracted.",
            "url": final_url,
        }


async def fetch_documents(urls: list[str], concurrency: int | None = None) -> list[dict]:
    limit = concurrency or int(os.getenv("WEB_FETCH_CONCURRENCY", "4"))
    semaphore = asyncio.Semaphore(max(1, limit))

    async def bounded_fetch(url: str) -> dict:
        async with semaphore:
            try:
                return await fetch_document(url)
            except ValueError as exc:
                return {"success": False, "error": "page_fetch_failed", "message": str(exc), "url": url}

    return await asyncio.gather(*(bounded_fetch(url) for url in urls))
