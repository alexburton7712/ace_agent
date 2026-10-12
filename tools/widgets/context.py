"""Request-scoped Ace App widget callback credentials."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class WidgetCallback:
    url: str
    token: str
    protocol: str


_WIDGET_CALLBACK: ContextVar[WidgetCallback | None] = ContextVar(
    "ace_widget_callback",
    default=None,
)


def get_widget_callback() -> WidgetCallback | None:
    """Return the callback advertised for the current request, if any."""
    return _WIDGET_CALLBACK.get()


@contextmanager
def widget_callback_scope(
    url: str | None,
    token: str | None,
    protocol: str | None,
) -> Iterator[None]:
    """Expose callback headers only while one audio request is processed."""
    callback = None
    if url is not None or token is not None or protocol is not None:
        callback = WidgetCallback(url or "", token or "", protocol or "")
    context_token = _WIDGET_CALLBACK.set(callback)
    try:
        yield
    finally:
        _WIDGET_CALLBACK.reset(context_token)


__all__ = ["WidgetCallback", "get_widget_callback", "widget_callback_scope"]
