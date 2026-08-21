import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("VLLM_BASE_URL", "http://example.invalid/v1")
os.environ.setdefault("VLLM_API_KEY", "test-key")
os.environ.setdefault("VLLM_MODEL", "test-model")

from tools.web.client import SearchResult, SearxngSearchClient, create_search_client
from tools.web.parser import extract_document
from tools.web.tools import web_fetch_many, web_search


class WebToolTests(unittest.IsolatedAsyncioTestCase):
    def test_search_backend_is_always_searxng(self):
        with patch.dict(
            os.environ,
            {
                "SEARXNG_URL": "http://searxng.test:8080",
            },
        ):
            client = create_search_client()

        self.assertIsInstance(client, SearxngSearchClient)

    async def test_search_returns_structured_results(self):
        client = AsyncMock()
        client.search.return_value = [
            SearchResult(
                title="Example",
                url="https://example.com/story",
                snippet="Summary",
                source="Example News",
                published_at="2026-08-20",
            )
        ]
        with patch("tools.web.tools.create_search_client", return_value=client):
            result = await web_search("current event", 3)

        client.search.assert_awaited_once_with("current event", 3)
        self.assertEqual(result[0]["title"], "Example")
        self.assertEqual(result[0]["source"], "Example News")

    async def test_fetch_many_normalizes_urls_and_delegates_concurrently(self):
        expected = [{"success": True, "url": "https://example.com/"}]
        with patch("tools.web.tools.fetch_documents", new=AsyncMock(return_value=expected)) as fetch:
            result = await web_fetch_many(["https://example.com/"])

        fetch.assert_awaited_once_with(["https://example.com/"])
        self.assertEqual(result, expected)

    def test_extraction_is_readable_and_bounded(self):
        html = """
            <html><head><title>Test article</title></head><body>
            <nav>Navigation noise</nav>
            <article><h1>Useful headline</h1><p>This is the useful article body with enough words to extract.</p></article>
            </body></html>
        """
        result = extract_document("https://example.com/article", html, max_characters=30)

        self.assertTrue(result["success"])
        self.assertIn(result["title"], {"Test article", "Useful headline"})
        self.assertLessEqual(len(result["text"]), 30)
        self.assertTrue(result["truncated"])
