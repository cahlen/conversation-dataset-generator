"""LLM generation wrappers with retry logic for conversation dataset generation."""

from __future__ import annotations

import logging
import re
import time

from conversation_dataset_generator.parsing import (
    parse_arg_generation_output,
    parse_conversation_to_sharegpt,
    parse_variation_output,
)
from conversation_dataset_generator.prompts import (
    build_arg_generation_messages,
    build_continuation_messages,
    build_conversation_messages,
    build_variation_messages,
)

logger = logging.getLogger(__name__)

_ARG_DEFAULTS = {
    "persona1_desc": "A character in the conversation",
    "persona2_desc": "A character in the conversation",
    "topic": "General discussion",
    "scenario": "An unspecified setting",
    "style": "Casual",
}


def generate_args_from_brief(
    brief: str,
    backend,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Generate conversation arguments from a creative brief."""
    persona1_context: str | None = None
    persona2_context: str | None = None

    if persona1_search_term or persona2_search_term:
        import time as _time
        from conversation_dataset_generator.web_search import get_persona_context

    if persona1_search_term:
        try:
            persona1_context = get_persona_context(persona1_search_term)
        except Exception as exc:
            logger.warning("Web search for persona1 failed: %s", exc)

    if persona1_search_term and persona2_search_term:
        _time.sleep(1.5)

    if persona2_search_term:
        try:
            persona2_context = get_persona_context(persona2_search_term)
        except Exception as exc:
            logger.warning("Web search for persona2 failed: %s", exc)

    messages = build_arg_generation_messages(
        brief,
        persona1_context=persona1_context,
        persona2_context=persona2_context,
        search_term1=persona1_search_term,
        search_term2=persona2_search_term,
    )

    delay = 1
    for attempt in range(max_retries):
        text = backend.complete(messages)
        if text:
            result = parse_arg_generation_output(text)
            if result is not None:
                return result
            logger.warning(
                "Attempt %d/%d: failed to parse arg generation output.",
                attempt + 1, max_retries,
            )
        else:
            logger.warning(
                "Attempt %d/%d: backend returned no text.",
                attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2

    logger.error("generate_args_from_brief exhausted %d retries.", max_retries)
    return None


def generate_args_from_brief_safe(
    brief: str,
    backend,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Wrapper that fills missing optional fields with defaults."""
    result = generate_args_from_brief(
        brief, backend,
        persona1_search_term=persona1_search_term,
        persona2_search_term=persona2_search_term,
        max_retries=max_retries,
    )
    if result is None:
        return None
    for key, default in _ARG_DEFAULTS.items():
        if key not in result or not result[key]:
            result[key] = default
            logger.info("Applied default for missing key '%s': %r", key, default)
    return result


def generate_topic_variation(
    persona1: str, persona1_desc: str,
    persona2: str, persona2_desc: str,
    initial_topic: str, initial_scenario: str, initial_style: str,
    backend,
    original_brief: str | None = None,
) -> dict | None:
    """Generate a topic/scenario/style variation for existing personas."""
    messages = build_variation_messages(
        persona1=persona1, persona1_desc=persona1_desc,
        persona2=persona2, persona2_desc=persona2_desc,
        initial_topic=initial_topic, initial_scenario=initial_scenario,
        initial_style=initial_style, original_brief=original_brief,
    )
    text = backend.complete(messages)
    if not text:
        logger.warning("generate_topic_variation: backend returned no text.")
        return None
    result = parse_variation_output(text)
    if result is None:
        logger.warning("generate_topic_variation: failed to parse variation output.")
        return None
    if "style" not in result or not result["style"]:
        result["style"] = initial_style
    return result


def _add_speaker_names(turns, text, persona_names):
    name_pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(n) for n in persona_names) + r")\s*:",
        re.IGNORECASE,
    )
    lines = text.strip().split("\n")
    turn_idx = 0
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        m = name_pattern.match(line_s)
        if m and turn_idx < len(turns):
            speaker = m.group(1).strip()
            for name in persona_names:
                if speaker.lower() == name.lower():
                    turns[turn_idx]["speaker_name"] = name
                    break
            turn_idx += 1


def generate_conversation(
    topic: str,
    persona1: str | None = None,
    persona2: str | None = None,
    persona1_desc: str | None = None,
    persona2_desc: str | None = None,
    scenario: str = "",
    style: str = "",
    backend=None,
    max_new_tokens: int = 2048,
    include_points: str | None = None,
    role_mapping: dict | None = None,
    *,
    personas: list[tuple[str, str]] | None = None,
) -> list[dict] | None:
    """Generate a conversation and parse it into ShareGPT turn format."""
    if personas is None:
        personas = [
            (persona1, persona1_desc or ""),
            (persona2, persona2_desc or ""),
        ]
    persona_names = [name for name, _ in personas]

    messages = build_conversation_messages(
        topic=topic, personas=personas,
        scenario=scenario, style=style, include_points=include_points,
    )
    text = backend.complete(messages, max_new_tokens=max_new_tokens)
    if not text:
        logger.warning("generate_conversation: backend returned no text.")
        return None

    stripped = text.strip()
    if not any(
        stripped.lower().startswith(f"{name.lower()}:") for name in persona_names
    ):
        logger.warning(
            "generate_conversation: output does not start with a persona prefix. "
            "Got: %r", stripped[:80],
        )

    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping
    )
    if turns is None:
        logger.warning("generate_conversation: failed to parse conversation output.")
        return None

    _add_speaker_names(turns, text, persona_names)
    return turns


def generate_continuation(
    personas: list[tuple[str, str]],
    prior_turns: list[dict],
    topic: str, scenario: str, style: str,
    backend,
    max_new_tokens: int = 2048,
    role_mapping: dict | None = None,
) -> list[dict] | None:
    """Generate a continuation of an existing conversation."""
    persona_names = [name for name, _ in personas]
    messages = build_continuation_messages(
        personas=personas, prior_turns=prior_turns,
        topic=topic, scenario=scenario, style=style,
    )
    text = backend.complete(messages, max_new_tokens=max_new_tokens)
    if not text:
        logger.warning("generate_continuation: backend returned no text.")
        return None
    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping
    )
    if turns is None:
        logger.warning("generate_continuation: failed to parse continuation output.")
        return None
    _add_speaker_names(turns, text, persona_names)
    return turns
