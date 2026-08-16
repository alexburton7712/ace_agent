from tools.time import TOOL as TIME_TOOL
from tools.time import get_current_time

TOOLS = [
    TIME_TOOL,
]


TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
}


def get_tool_schemas():
    return TOOLS


async def execute_tool(name: str, arguments: dict):
    function = TOOL_FUNCTIONS.get(name)

    if function is None:
        raise ValueError(f"Unknown tool: {name}")

    return await function(**arguments)