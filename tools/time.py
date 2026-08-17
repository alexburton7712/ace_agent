from datetime import datetime

from tools.tool import tool


@tool
async def get_current_time():
    """Get the current local date and time."""

    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(),
    }