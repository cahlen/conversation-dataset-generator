"""Chat backends: HF pipeline and OpenAI-compatible HTTP."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ChatBackend(Protocol):
    """A chat-completion backend.

    Implementations take a list of role/content message dicts and return the
    generated assistant text (just the new tokens, not the prompt). On any
    failure — network error, parse error, empty response — return None.
    """

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        ...
