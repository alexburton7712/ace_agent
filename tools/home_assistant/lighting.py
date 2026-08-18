from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tools.tool import tool

from .client import HomeAssistantClient, HomeAssistantError


class ToolParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListLightsParameters(ToolParameters):
    pass


class LightParameters(ToolParameters):
    light: str = Field(
        description="Friendly name or Home Assistant entity ID of the target light, preferably an entity ID returned by list_lights.",
        min_length=1,
    )


class BrightnessParameters(LightParameters):
    brightness: int = Field(
        description="Desired brightness percentage, where 0 is off and 100 is full brightness.",
        ge=0,
        le=100,
    )


class ColorParameters(LightParameters):
    red: int = Field(description="Red color channel intensity from 0 to 255.", ge=0, le=255)
    green: int = Field(description="Green color channel intensity from 0 to 255.", ge=0, le=255)
    blue: int = Field(description="Blue color channel intensity from 0 to 255.", ge=0, le=255)


class ColorTemperatureParameters(LightParameters):
    temperature: int = Field(
        description="White color temperature in Kelvin; lower values are warmer and higher values are cooler.",
        ge=1000,
        le=10000,
        examples=[2000, 6500],
    )


def _light_summary(state: dict[str, Any]) -> dict[str, Any]:
    attributes = state.get("attributes", {})
    raw_brightness = attributes.get("brightness")
    brightness_pct = (
        round(raw_brightness / 255 * 100)
        if isinstance(raw_brightness, int)
        else None
    )
    return {
        "name": attributes.get("friendly_name", state["entity_id"]),
        "entity_id": state["entity_id"],
        "state": state.get("state", "unknown"),
        "brightness_pct": brightness_pct,
        "supported_color_modes": attributes.get("supported_color_modes", []),
    }


def _light_details(state: dict[str, Any]) -> dict[str, Any]:
    """Return convenient light fields while preserving every HA attribute."""
    attributes = state.get("attributes", {})
    raw_brightness = attributes.get("brightness")
    brightness_pct = (
        round(raw_brightness / 255 * 100)
        if isinstance(raw_brightness, int)
        else None
    )
    current_state = state.get("state", "unknown")

    return {
        "success": True,
        "name": attributes.get("friendly_name", state["entity_id"]),
        "entity_id": state["entity_id"],
        "state": current_state,
        "is_on": current_state == "on",
        "brightness_pct": brightness_pct,
        "brightness_raw": raw_brightness,
        "color_mode": attributes.get("color_mode"),
        "rgb_color": attributes.get("rgb_color"),
        "xy_color": attributes.get("xy_color"),
        "hs_color": attributes.get("hs_color"),
        "rgbw_color": attributes.get("rgbw_color"),
        "rgbww_color": attributes.get("rgbww_color"),
        "color_temperature_kelvin": attributes.get("color_temp_kelvin"),
        "color_temperature_mireds": attributes.get("color_temp"),
        "effect": attributes.get("effect"),
        "supported_color_modes": attributes.get("supported_color_modes", []),
        "supported_features": attributes.get("supported_features"),
        "last_changed": state.get("last_changed"),
        "last_updated": state.get("last_updated"),
        # Home Assistant integrations can expose device-specific attributes. Keep
        # the complete payload so the agent can report information we do not yet
        # normalize above.
        "attributes": attributes,
        "context": state.get("context"),
    }


def _resolve_light(
    light: str,
    states: list[dict[str, Any]],
) -> tuple[str, str] | dict[str, Any]:
    query = light.casefold().strip()
    lights = [state for state in states if state.get("entity_id", "").startswith("light.")]

    matches = []
    for state in lights:
        entity_id = state["entity_id"]
        friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)
        if query in {entity_id.casefold(), friendly_name.casefold()}:
            matches.append(state)

    if len(matches) == 1:
        match = matches[0]
        name = match.get("attributes", {}).get("friendly_name", match["entity_id"])
        return name, match["entity_id"]

    if len(matches) > 1:
        return {
            "success": False,
            "error": f"More than one light is named '{light}'. Use an entity_id instead.",
            "matches": [_light_summary(state) for state in matches],
        }

    return {
        "success": False,
        "error": f"Unknown light: {light}",
        "available_lights": [_light_summary(state) for state in lights],
    }


async def _call_light_service(
    light: str,
    service: str,
    **service_data: Any,
) -> dict[str, Any]:
    try:
        async with HomeAssistantClient() as client:
            states = await client.get_states()
            resolved = _resolve_light(light, states)
            if isinstance(resolved, dict):
                return resolved

            light_name, entity_id = resolved
            await client.call_service(
                domain="light",
                service=service,
                data={"entity_id": entity_id, **service_data},
            )
    except (HomeAssistantError, ValueError) as exc:
        return {
            "success": False,
            "error": str(exc),
            "light": light,
        }

    return {
        "success": True,
        "light": light_name,
        "entity_id": entity_id,
        "action": service,
    }


@tool(ListLightsParameters)
async def list_lights() -> dict:
    """List the names of lights and light groups available for use with lighting tools.

    Call this before using a lighting control tool and pass one of the returned
    names to that tool.
    """
    try:
        async with HomeAssistantClient() as client:
            states = await client.get_states()
    except (HomeAssistantError, ValueError) as exc:
        return {"success": False, "error": str(exc)}

    light_names = sorted(
        (
            state.get("attributes", {}).get("friendly_name", state["entity_id"])
            for state in states
            if state.get("entity_id", "").startswith("light.")
        ),
        key=str.casefold,
    )
    return {
        "success": True,
        "lights": light_names,
    }


@tool(LightParameters)
async def get_light_state(light: str) -> dict:
    """Get all currently available state, color, brightness, and capability information for one Home Assistant light."""
    try:
        async with HomeAssistantClient() as client:
            states = await client.get_states()
            resolved = _resolve_light(light, states)
            if isinstance(resolved, dict):
                return resolved

            _, entity_id = resolved
            state = next(
                state for state in states
                if state.get("entity_id") == entity_id
            )
    except (HomeAssistantError, ValueError) as exc:
        return {
            "success": False,
            "error": str(exc),
            "light": light,
        }

    return _light_details(state)


@tool(LightParameters)
async def turn_on_light(light: str) -> dict:
    """Turn on a Home Assistant light after calling list_lights."""
    return await _call_light_service(light, "turn_on")


@tool(LightParameters)
async def turn_off_light(light: str) -> dict:
    """Turn off a Home Assistant light after calling list_lights."""
    return await _call_light_service(light, "turn_off")


@tool(BrightnessParameters)
async def set_light_brightness(light: str, brightness: int) -> dict:
    """Set a light's brightness after calling list_lights, and turn it on."""
    if isinstance(brightness, bool) or not isinstance(brightness, int):
        return {
            "success": False,
            "error": "Brightness must be an integer from 0 to 100.",
        }

    if not 0 <= brightness <= 100:
        return {
            "success": False,
            "error": "Brightness must be between 0 and 100.",
        }

    result = await _call_light_service(
        light,
        "turn_on",
        brightness_pct=brightness,
    )
    if result["success"]:
        result["brightness"] = brightness
    return result


@tool(ColorParameters)
async def set_light_color(light: str, red: int, green: int, blue: int) -> dict:
    """Set a light to RGB after list_lights confirms color support, and turn it on."""
    channels = {"red": red, "green": green, "blue": blue}

    for name, value in channels.items():
        if isinstance(value, bool) or not isinstance(value, int):
            return {
                "success": False,
                "error": f"{name.capitalize()} must be an integer from 0 to 255.",
            }
        if not 0 <= value <= 255:
            return {
                "success": False,
                "error": f"{name.capitalize()} must be between 0 and 255.",
            }

    result = await _call_light_service(
        light,
        "turn_on",
        rgb_color=[red, green, blue],
    )
    if result["success"]:
        result["color"] = channels
    return result


@tool(ColorTemperatureParameters)
async def set_light_color_temperature(light: str, temperature: int) -> dict:
    """Set white temperature after list_lights confirms support, and turn the light on."""
    if isinstance(temperature, bool) or not isinstance(temperature, int):
        return {
            "success": False,
            "error": "Color temperature must be an integer in Kelvin.",
        }

    if not 1000 <= temperature <= 10000:
        return {
            "success": False,
            "error": "Color temperature must be between 1000 and 10000 Kelvin.",
        }

    result = await _call_light_service(
        light,
        "turn_on",
        color_temp_kelvin=temperature,
    )
    if result["success"]:
        result["temperature_kelvin"] = temperature
    return result
