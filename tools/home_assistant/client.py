import os
from typing import Any

import httpx


class HomeAssistantError(Exception):
    """Raised when a Home Assistant API request fails."""


class HomeAssistantClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("HOME_ASSISTANT_URL")
        )

        self.token = (
            token
            or os.getenv("HOME_ASSISTANT_TOKEN")
        )

        if not self.base_url:
            raise ValueError(
                "HOME_ASSISTANT_URL environment variable is not set."
            )

        if not self.token:
            raise ValueError(
                "HOME_ASSISTANT_TOKEN environment variable is not set."
            )

        # Prevent URLs like:
        # http://homeassistant:8123//api/states
        self.base_url = self.base_url.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Any:
        """
        Send a request to Home Assistant and handle common errors.
        """
        try:
            response = await self._client.request(
                method,
                endpoint,
                **kwargs,
            )

            response.raise_for_status()

        except httpx.ConnectError as exc:
            raise HomeAssistantError(
                f"Could not connect to Home Assistant at "
                f"{self.base_url}"
            ) from exc

        except httpx.TimeoutException as exc:
            raise HomeAssistantError(
                "Home Assistant request timed out."
            ) from exc

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code

            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text

            raise HomeAssistantError(
                f"Home Assistant returned HTTP {status}: {detail}"
            ) from exc

        # Some endpoints may theoretically return no body.
        if not response.content:
            return None

        try:
            return response.json()
        except ValueError:
            return response.text

    async def ping(self) -> bool:
        """
        Check whether Home Assistant is reachable and authenticated.
        """
        try:
            await self._request(
                "GET",
                "/api/",
            )
            return True
        except HomeAssistantError:
            return False

    async def get_states(self) -> list[dict[str, Any]]:
        """
        Get the current state of every Home Assistant entity.

        Example entities:
            light.office_corner_lamp
            sensor.office_temperature
            switch.desk_fan
        """
        return await self._request(
            "GET",
            "/api/states",
        )

    async def get_state(
        self,
        entity_id: str,
    ) -> dict[str, Any]:
        """
        Get the current state of a specific entity.
        """
        return await self._request(
            "GET",
            f"/api/states/{entity_id}",
        )

    async def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
    ) -> Any:
        """
        Call any Home Assistant service.

        Example:

            await client.call_service(
                domain="light",
                service="turn_off",
                data={
                    "entity_id": "light.office_corner_lamp"
                },
            )
        """
        return await self._request(
            "POST",
            f"/api/services/{domain}/{service}",
            json=data or {},
        )