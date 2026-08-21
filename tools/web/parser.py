from datetime import datetime, timezone
from typing import Any

import trafilatura
from bs4 import BeautifulSoup


def extract_document(url: str, html: str, max_characters: int) -> dict[str, Any]:
    """Extract article-like text and compact metadata from HTML."""
    metadata = trafilatura.extract_metadata(html)
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    soup = BeautifulSoup(html, "html.parser")
    if not text:
        for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())

    title = getattr(metadata, "title", None)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)

    text = (text or "").strip()
    truncated = len(text) > max_characters
    if truncated:
        text = text[:max_characters].rsplit(" ", 1)[0]

    return {
        "success": True,
        "url": url,
        "title": title,
        "text": text,
        "author": getattr(metadata, "author", None),
        "published_at": getattr(metadata, "date", None),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "truncated": truncated,
    }
