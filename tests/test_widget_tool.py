import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

os.environ.setdefault("VLLM_BASE_URL", "http://example.invalid/v1")
os.environ.setdefault("VLLM_API_KEY", "test-key")
os.environ.setdefault("VLLM_MODEL", "test-model")

from tools.widgets.context import get_widget_callback, widget_callback_scope
from tools.widgets.create import create_widget


class WidgetToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_when_app_callback_is_unavailable(self):
        result = await create_widget("weather", "<p>Sunny</p>")

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "widget_unavailable")

    async def test_posts_widget_to_request_scoped_callback(self):
        response = httpx.Response(
            201,
            json={"id": "weather", "status": "created", "revision": 1},
        )
        client = AsyncMock()
        client.post.return_value = response
        client.__aenter__.return_value = client

        with (
            widget_callback_scope(
                "http://127.0.0.1:4321/api/widgets",
                "launch-secret",
                "1",
            ),
            patch("tools.widgets.create.httpx.AsyncClient", return_value=client),
        ):
            result = await create_widget("weather", "<p>Sunny</p>")

        self.assertEqual(
            result,
            {"success": True, "id": "weather", "status": "created", "revision": 1},
        )
        client.post.assert_awaited_once_with(
            "http://127.0.0.1:4321/api/widgets",
            headers={"Authorization": "Bearer launch-secret"},
            json={"id": "weather", "widget": "<p>Sunny</p>"},
        )
        self.assertIsNone(get_widget_callback())

    async def test_rejects_non_loopback_callback_without_sending(self):
        with (
            widget_callback_scope(
                "http://example.com/api/widgets",
                "launch-secret",
                "1",
            ),
            patch("tools.widgets.create.httpx.AsyncClient") as client_type,
        ):
            result = await create_widget("weather", "<p>Sunny</p>")

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "invalid_widget_callback")
        client_type.assert_not_called()

    async def test_returns_safe_delivery_error(self):
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError("launch-secret")
        client.__aenter__.return_value = client

        with (
            widget_callback_scope(
                "http://127.0.0.1:4321/api/widgets",
                "launch-secret",
                "1",
            ),
            patch("tools.widgets.create.httpx.AsyncClient", return_value=client),
        ):
            result = await create_widget("weather", "<p>Sunny</p>")

        self.assertEqual(
            result,
            {
                "success": False,
                "error": "widget_delivery_failed",
                "message": "Could not reach Ace App's widget renderer.",
            },
        )
        self.assertNotIn("launch-secret", str(result))


if __name__ == "__main__":
    unittest.main()
