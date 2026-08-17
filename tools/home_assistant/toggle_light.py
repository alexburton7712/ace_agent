# tools/toggle_light.py

from tools.tool import tool

from .client import HomeAssistantClient

LIGHTS = {
    "office lamp": "light.alex_s_office_lamp",
}

@tool
async def toggle_light(light: str) -> dict:
    """
    Toggle a hardcoded Home Assistant light.

    Args:
        light: Friendly name of the light to toggle.
    """

    light_key = light.lower().strip()

    entity_id = LIGHTS.get(light_key)

    if not entity_id:
        return {
            "success": False,
            "error": f"Unknown light: {light}",
            "available_lights": list(LIGHTS.keys()),
        }

    async with HomeAssistantClient() as client:
        await client.call_service(
            domain="light",
            service="toggle",
            data={
                "entity_id": entity_id,
            },
        )

    return {
        "success": True,
        "light": light_key,
        "entity_id": entity_id,
    }