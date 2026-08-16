from datetime import datetime


async def get_current_time():
    now = datetime.now().astimezone()

    return {
        "datetime": now.isoformat(),
        "timezone": now.tzname(),
    }


TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time using the timezone configured on the machine running Ace.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}