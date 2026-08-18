import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

ToolFunction = Callable[..., Any]
ParametersModel = type[BaseModel]
F = TypeVar("F", bound=ToolFunction)


@dataclass(frozen=True)
class ToolDefinition:
    function: ToolFunction
    parameters_model: ParametersModel


_REGISTERED_TOOLS: dict[str, ToolDefinition] = {}


def tool(parameters_model: ParametersModel):
    """Register an async function using a Pydantic argument model."""
    if not inspect.isclass(parameters_model) or not issubclass(parameters_model, BaseModel):
        raise TypeError("@tool expects a Pydantic BaseModel subclass.")

    def decorator(function: F) -> F:
        name = function.__name__
        if name in _REGISTERED_TOOLS:
            raise ValueError(f"Tool '{name}' is already registered.")
        if not inspect.iscoroutinefunction(function):
            raise TypeError(f"Tool '{name}' must be async.")

        _REGISTERED_TOOLS[name] = ToolDefinition(function, parameters_model)
        return function

    return decorator


def get_registered_tools() -> dict[str, ToolDefinition]:
    return _REGISTERED_TOOLS.copy()


def build_tool_schema(definition: ToolDefinition) -> dict[str, Any]:
    function = definition.function
    return {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function) or "",
            "parameters": definition.parameters_model.model_json_schema(),
        },
    }


def validate_tool_arguments(
    definition: ToolDefinition,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        parameters = definition.parameters_model.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"Invalid arguments for tool: {exc}") from exc
    return parameters.model_dump()
