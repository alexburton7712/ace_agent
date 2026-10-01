import asyncio
from copy import deepcopy
from typing import Any

DEFAULT_MAX_USER_TURNS = 10


class ConversationManager:
    """Own the full conversation while bounding context sent to the model."""

    def __init__(
        self,
        system_prompt: str,
        max_user_turns: int = DEFAULT_MAX_USER_TURNS,
    ) -> None:
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be at least 1.")

        self._system_message: dict[str, Any] = {
            "role": "system",
            "content": system_prompt,
        }
        self._history: list[dict[str, Any]] = []
        self.max_user_turns = max_user_turns
        self.lock = asyncio.Lock()

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a snapshot of the complete conversation history."""
        return deepcopy(self._history)

    def append(self, message: dict[str, Any]) -> None:
        """Append a message to the permanent history for this process."""
        self._history.append(deepcopy(message))

    def model_messages(self) -> list[dict[str, Any]]:
        """Build model context from the system prompt and newest user turns."""
        user_message_indexes = [
            index
            for index, message in enumerate(self._history)
            if message["role"] == "user"
        ]

        first_retained_index = 0
        if len(user_message_indexes) > self.max_user_turns:
            first_retained_index = user_message_indexes[-self.max_user_turns]

        return [
            deepcopy(self._system_message),
            *deepcopy(self._history[first_retained_index:]),
        ]

    def clear(self) -> None:
        """Remove all conversation history while preserving the system prompt."""
        self._history.clear()
