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


def _extract_generated_text(full_output: str, prompt_text: str) -> str | None:
    """Strip the prompt prefix from a pipeline's output. Return None if empty."""
    if full_output.startswith(prompt_text):
        generated = full_output[len(prompt_text):]
    else:
        generated = full_output
    generated = generated.strip()
    return generated if generated else None


class HFBackend:
    """ChatBackend backed by a transformers text-generation pipeline + tokenizer."""

    def __init__(self, pipeline, tokenizer):
        self._pipeline = pipeline
        self._tokenizer = tokenizer

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as exc:
            logger.error("apply_chat_template failed: %s", exc)
            return None

        try:
            outputs = self._pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception as exc:
            logger.error("HF pipeline call failed: %s", exc)
            return None

        if not outputs or not isinstance(outputs, list):
            logger.warning("Pipeline returned unexpected output: %r", outputs)
            return None

        raw = outputs[0].get("generated_text", "")
        return _extract_generated_text(raw, prompt_text)


class OpenAIBackend:
    """ChatBackend backed by an OpenAI-compatible HTTP API.

    Works with the real OpenAI API as well as LM Studio, Ollama (`/v1` shim),
    vLLM, TGI, OpenRouter — anything that speaks `chat.completions`.

    Pass ``client`` to inject a pre-built or mocked client (used in tests);
    otherwise the constructor creates one from ``base_url`` + ``api_key``.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        *,
        client=None,
    ):
        self.model_id = model_id
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            logger.error("OpenAI chat.completions call failed: %s", exc)
            return None

        if not completion.choices:
            logger.warning("OpenAI response had no choices.")
            return None

        content = completion.choices[0].message.content
        if not content:
            return None
        content = content.strip()
        return content if content else None
