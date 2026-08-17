import inspect
from typing import (
    Any,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

_REGISTERED_TOOLS = {}


def tool(function):
    """
    Mark an async function as an agent tool.
    """

    name = function.__name__

    if name in _REGISTERED_TOOLS:
        raise ValueError(
            f"Tool '{name}' is already registered."
        )

    if not inspect.iscoroutinefunction(function):
        raise TypeError(
            f"Tool '{name}' must be async."
        )

    _REGISTERED_TOOLS[name] = function

    return function


def get_registered_tools():
    return _REGISTERED_TOOLS.copy()


def python_type_to_json_schema(annotation):
    """
    Convert a Python type annotation into JSON schema.
    """

    origin = get_origin(annotation)
    args = get_args(annotation)

    # str
    if annotation is str:
        return {
            "type": "string"
        }

    # int
    if annotation is int:
        return {
            "type": "integer"
        }

    # float
    if annotation is float:
        return {
            "type": "number"
        }

    # bool
    if annotation is bool:
        return {
            "type": "boolean"
        }

    # list[T]
    if origin is list:
        item_type = args[0] if args else Any

        return {
            "type": "array",
            "items": python_type_to_json_schema(item_type),
        }

    # Literal["on", "off"]
    if origin is Literal:
        values = list(args)

        schema = {
            "enum": values,
        }

        if values:
            first = values[0]

            if isinstance(first, str):
                schema["type"] = "string"
            elif isinstance(first, int):
                schema["type"] = "integer"
            elif isinstance(first, float):
                schema["type"] = "number"
            elif isinstance(first, bool):
                schema["type"] = "boolean"

        return schema

    # dict
    if origin is dict or annotation is dict:
        return {
            "type": "object"
        }

    # fallback
    return {}


def build_tool_schema(function):
    signature = inspect.signature(function)
    type_hints = get_type_hints(function)

    properties = {}
    required = []

    for name, parameter in signature.parameters.items():
        annotation = type_hints.get(
            name,
            Any,
        )

        properties[name] = python_type_to_json_schema(
            annotation
        )

        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    description = inspect.getdoc(function) or ""

    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }