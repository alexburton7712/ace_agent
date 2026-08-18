import importlib
import pkgutil

import tools
from tools.tool import (
    build_tool_schema,
    get_registered_tools,
    validate_tool_arguments,
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
        build_tool_schema(definition)
        for definition in registered_tools.values()
    ]


async def execute_tool(
    name: str,
    arguments: dict,
):
    registered_tools = get_registered_tools()

    definition = registered_tools.get(name)

    if definition is None:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    arguments = validate_tool_arguments(definition, arguments)
    return await definition.function(**arguments)
