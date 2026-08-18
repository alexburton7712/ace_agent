from datetime import datetime

from pydantic import BaseModel, ConfigDict

from tools.tool import tool


class CurrentTimeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


@tool(CurrentTimeParameters)
async def get_current_time():
    """Get the current local date and time."""

    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(),
    }
