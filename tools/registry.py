import importlib
import pkgutil

import tools
from tools.tool import (
    build_tool_schema,
    get_registered_tools,
)


def discover_tools():
    """
    Import all modules inside the tools package so that
    @tool decorators can register their functions.
    """

    for module_info in pkgutil.walk_packages(
        tools.__path__,
        prefix="tools.",
    ):
        module_name = module_info.name

        # Don't try to discover tools inside infrastructure modules.
        if module_name in {
            "tools.registry",
            "tools.tool",
        }:
            continue

        importlib.import_module(module_name)


def get_tool_schemas():
    registered_tools = get_registered_tools()

    return [
        build_tool_schema(function)
        for function in registered_tools.values()
    ]


async def execute_tool(
    name: str,
    arguments: dict,
):
    registered_tools = get_registered_tools()

    function = registered_tools.get(name)

    if function is None:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    return await function(**arguments)