"""Create and replace widgets through Ace App's loopback callback."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from tools.tool import tool

from .context import WidgetCallback, get_widget_callback

MAX_WIDGET_BYTES = 256 * 1024
CALLBACK_TIMEOUT_SECONDS = 5.0


class CreateWidgetParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Stable widget ID. Reusing an ID replaces that widget in place.",
    )
    widget: str = Field(
        min_length=1,
        description=(
            "Self-contained HTML fragment with inline CSS and optional inline JavaScript. "
            "Network access and external assets are unavailable."
        ),
    )


def _validated_callback(callback: WidgetCallback) -> str | None:
    """Return a client-safe error code when a callback is not protocol-1 loopback."""
    if callback.protocol != "1" or not callback.token or len(callback.token) > 512:
        return "invalid_widget_callback"

    try:
        parsed = urlsplit(callback.url)
        port = parsed.port
    except ValueError:
        return "invalid_widget_callback"

    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.path != "/api/widgets"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return "invalid_widget_callback"
    return None


def _failure(error: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": error, "message": message}


@tool(CreateWidgetParameters)
async def create_widget(id: str, widget: str) -> dict[str, Any]:
    """Create a widget in Ace App, or replace the existing widget with the same ID.

    Use a self-contained HTML fragment. Inline CSS and JavaScript are supported,
    but network requests, external assets, forms, storage, popups, and access to
    the surrounding Ace App are blocked by the widget sandbox.
    """
    callback = get_widget_callback()
    if callback is None:
        return _failure(
            "widget_unavailable",
            "No Ace App widget callback is available for this request.",
        )

    callback_error = _validated_callback(callback)
    if callback_error:
        return _failure(
            callback_error,
            "Ace App provided an invalid widget callback.",
        )

    if "\x00" in widget:
        return _failure("invalid_widget", "Widget HTML must not contain NUL characters.")
    if len(widget.encode("utf-8")) > MAX_WIDGET_BYTES:
        return _failure("payload_too_large", "Widget HTML exceeds the 256 KiB limit.")

    try:
        async with httpx.AsyncClient(
            timeout=CALLBACK_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.post(
                callback.url,
                headers={"Authorization": f"Bearer {callback.token}"},
                json={"id": id, "widget": widget},
            )
    except httpx.HTTPError:
        return _failure(
            "widget_delivery_failed",
            "Could not reach Ace App's widget renderer.",
        )

    if response.status_code not in {200, 201}:
        messages = {
            400: "Ace App rejected the widget payload.",
            401: "Ace App rejected the widget callback credentials.",
            413: "Ace App rejected the widget because it was too large.",
            415: "Ace App rejected the widget request format.",
            503: "Ace App's widget renderer is unavailable.",
        }
        return _failure(
            "widget_delivery_failed",
            messages.get(response.status_code, "Ace App could not render the widget."),
        )

    try:
        payload = response.json()
    except ValueError:
        return _failure(
            "invalid_widget_response",
            "Ace App returned an invalid widget response.",
        )

    if (
        not isinstance(payload, dict)
        or payload.get("id") != id
        or payload.get("status") not in {"created", "replaced"}
        or isinstance(payload.get("revision"), bool)
        or not isinstance(payload.get("revision"), int)
    ):
        return _failure(
            "invalid_widget_response",
            "Ace App returned an invalid widget response.",
        )

    return {
        "success": True,
        "id": id,
        "status": payload["status"],
        "revision": payload["revision"],
    }


__all__ = ["CreateWidgetParameters", "create_widget"]
