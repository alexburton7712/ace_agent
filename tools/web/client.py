import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import httpx


class SearchProviderError(RuntimeError):
    """Raised when a configured search provider cannot return results."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str | None = None
    source: str | None = None
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchClient(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search for a query and return normalized results."""


class SearxngSearchClient(SearchClient):
    def __init__(self, base_url: str | None = None, timeout: float = 15.0):
        self.base_url = (base_url or os.getenv("SEARXNG_URL", "")).rstrip("/")
        if not self.base_url:
            raise SearchProviderError("SEARXNG_URL environment variable is not set.")
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={"q": query, "format": "json"},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchProviderError(f"SearXNG search failed: {exc}") from exc

        results = []
        for item in payload.get("results", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title") or item.get("url", "Untitled result"),
                url=item.get("url", ""),
                snippet=item.get("content") or None,
                source=item.get("engine") or None,
                published_at=item.get("publishedDate") or None,
            ))
        return [result for result in results if result.url]


def create_search_client() -> SearchClient:
    """Create Ace's single supported search backend: SearXNG."""
    return SearxngSearchClient()
