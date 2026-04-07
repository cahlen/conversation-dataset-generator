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

_REQUIRED_ARG_KEYS = (
    "persona1",
    "persona1_desc",
    "persona2",
    "persona2_desc",
    "topic",
    "scenario",
    "style",
)

_ARG_DEFAULTS = {
    "persona1_desc": "A character in the conversation",
    "persona2_desc": "A character in the conversation",
    "topic": "General discussion",
    "scenario": "An unspecified setting",
    "style": "Casual",
}


def extract_generated_text(full_output: str, prompt_text: str) -> str | None:
    """Strip the prompt prefix from LLM output.

    Args:
        full_output: The complete text returned by the pipeline
            (prompt + generated portion).
        prompt_text: The prompt string that was fed to the pipeline.

    Returns:
        The generated portion only, or None if the result is empty.
    """
    if full_output.startswith(prompt_text):
        generated = full_output[len(prompt_text):]
    else:
        generated = full_output

    generated = generated.strip()
    return generated if generated else None


def _call_pipeline(
    generator_pipeline,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str | None:
    """Apply chat template, call the pipeline, and extract generated text.

    Args:
        generator_pipeline: A HuggingFace text-generation pipeline (or mock).
        tokenizer: The tokenizer paired with the pipeline.
        messages: List of role/content message dicts.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.

    Returns:
        The generated text string, or None on failure.
    """
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        logger.error("apply_chat_template failed: %s", exc)
        return None

    try:
        outputs = generator_pipeline(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception as exc:
        logger.error("Pipeline call failed: %s", exc)
        return None

    if not outputs or not isinstance(outputs, list):
        logger.warning("Pipeline returned unexpected output: %r", outputs)
        return None

    raw = outputs[0].get("generated_text", "")
    return extract_generated_text(raw, prompt_text)


def generate_args_from_brief(
    brief: str,
    generator_pipeline,
    tokenizer,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Generate conversation arguments from a creative brief.

    Uses exponential backoff on failure. Optionally enriches prompts with
    web-search context for the two personas.

    Args:
        brief: The creative brief describing the desired conversation.
        generator_pipeline: HuggingFace text-generation pipeline.
        tokenizer: Paired tokenizer.
        persona1_search_term: Optional search term for persona 1 web context.
        persona2_search_term: Optional search term for persona 2 web context.
        max_retries: Number of attempts before giving up.

    Returns:
        Dict of parsed arguments, or None if all retries fail.
    """
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
        _time.sleep(1.5)  # avoid DuckDuckGo rate limiting

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
        text = _call_pipeline(generator_pipeline, tokenizer, messages)
        if text:
            result = parse_arg_generation_output(text)
            if result is not None:
                return result
            logger.warning(
                "Attempt %d/%d: failed to parse arg generation output.",
                attempt + 1,
                max_retries,
            )
        else:
            logger.warning(
                "Attempt %d/%d: pipeline returned no text.",
                attempt + 1,
                max_retries,
            )

        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2

    logger.error("generate_args_from_brief exhausted %d retries.", max_retries)
    return None


def generate_args_from_brief_safe(
    brief: str,
    generator_pipeline,
    tokenizer,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Wrapper around generate_args_from_brief that fills in missing optional fields.

    Args:
        brief: The creative brief describing the desired conversation.
        generator_pipeline: HuggingFace text-generation pipeline.
        tokenizer: Paired tokenizer.
        persona1_search_term: Optional search term for persona 1 web context.
        persona2_search_term: Optional search term for persona 2 web context.
        max_retries: Number of attempts before giving up.

    Returns:
        Dict with all required keys (defaults applied for missing ones),
        or None if persona1 or persona2 are missing entirely.
    """
    result = generate_args_from_brief(
        brief,
        generator_pipeline,
        tokenizer,
        persona1_search_term=persona1_search_term,
        persona2_search_term=persona2_search_term,
        max_retries=max_retries,
    )

    if result is None:
        return None

    # Apply defaults for optional/missing fields
    for key, default in _ARG_DEFAULTS.items():
        if key not in result or not result[key]:
            result[key] = default
            logger.info("Applied default for missing key '%s': %r", key, default)

    return result


def generate_topic_variation(
    persona1: str,
    persona1_desc: str,
    persona2: str,
    persona2_desc: str,
    initial_topic: str,
    initial_scenario: str,
    initial_style: str,
    generator_pipeline,
    tokenizer,
    original_brief: str | None = None,
) -> dict | None:
    """Generate a topic/scenario/style variation for existing personas.

    Args:
        persona1: Name of the first persona.
        persona1_desc: Description of persona1.
        persona2: Name of the second persona.
        persona2_desc: Description of persona2.
        initial_topic: The original topic.
        initial_scenario: The original scenario.
        initial_style: The original style (used as fallback if LLM omits it).
        generator_pipeline: HuggingFace text-generation pipeline.
        tokenizer: Paired tokenizer.
        original_brief: Optional original creative brief for additional context.

    Returns:
        Dict with 'topic', 'scenario', and 'style' keys, or None on failure.
    """
    messages = build_variation_messages(
        persona1=persona1,
        persona1_desc=persona1_desc,
        persona2=persona2,
        persona2_desc=persona2_desc,
        initial_topic=initial_topic,
        initial_scenario=initial_scenario,
        initial_style=initial_style,
        original_brief=original_brief,
    )

    text = _call_pipeline(generator_pipeline, tokenizer, messages)
    if not text:
        logger.warning("generate_topic_variation: pipeline returned no text.")
        return None

    result = parse_variation_output(text)
    if result is None:
        logger.warning("generate_topic_variation: failed to parse variation output.")
        return None

    # Fall back to initial_style if not present in parsed output
    if "style" not in result or not result["style"]:
        result["style"] = initial_style

    return result


def _add_speaker_names(
    turns: list[dict],
    text: str,
    persona_names: list[str],
) -> None:
    """Add ``speaker_name`` to each turn dict by scanning the raw LLM text.

    Modifies *turns* in place.
    """
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
    generator_pipeline=None,
    tokenizer=None,
    max_new_tokens: int = 2048,
    include_points: str | None = None,
    role_mapping: dict | None = None,
    *,
    personas: list[tuple[str, str]] | None = None,
) -> list[dict] | None:
    """Generate a conversation and parse it into ShareGPT turn format.

    Supports both the legacy 2-speaker interface (persona1/persona2) and the
    N-speaker interface (personas kwarg).

    Args:
        topic: The main topic of conversation.
        persona1: Name of the first speaker (legacy, ignored if personas given).
        persona2: Name of the second speaker (legacy, ignored if personas given).
        persona1_desc: Description of persona1 (legacy, ignored if personas given).
        persona2_desc: Description of persona2 (legacy, ignored if personas given).
        scenario: The setting or situation.
        style: The conversational style.
        generator_pipeline: HuggingFace text-generation pipeline.
        tokenizer: Paired tokenizer.
        max_new_tokens: Maximum tokens to generate.
        include_points: Optional comma-separated points to cover.
        role_mapping: Optional dict mapping speakers to roles.
        personas: List of (name, desc) tuples for N speakers (keyword-only).

    Returns:
        List of ShareGPT turn dicts, or None on failure.
    """
    # Resolve personas list
    if personas is None:
        personas = [
            (persona1, persona1_desc or ""),
            (persona2, persona2_desc or ""),
        ]

    persona_names = [name for name, _ in personas]

    messages = build_conversation_messages(
        topic=topic,
        personas=personas,
        scenario=scenario,
        style=style,
        include_points=include_points,
    )

    text = _call_pipeline(
        generator_pipeline,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )

    if not text:
        logger.warning("generate_conversation: pipeline returned no text.")
        return None

    # Validate the output starts with one of the persona prefixes
    stripped = text.strip()
    if not any(
        stripped.lower().startswith(f"{name.lower()}:")
        for name in persona_names
    ):
        logger.warning(
            "generate_conversation: output does not start with a persona prefix. "
            "Got: %r", stripped[:80]
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
    topic: str,
    scenario: str,
    style: str,
    generator_pipeline,
    tokenizer,
    max_new_tokens: int = 2048,
    role_mapping: dict | None = None,
) -> list[dict] | None:
    """Generate a continuation of an existing conversation.

    Args:
        personas: List of (name, desc) tuples for all speakers.
        prior_turns: List of turn dicts with 'speaker_name' and 'value' keys.
        topic: The main topic of conversation.
        scenario: The setting or situation.
        style: The conversational style.
        generator_pipeline: HuggingFace text-generation pipeline.
        tokenizer: Paired tokenizer.
        max_new_tokens: Maximum tokens to generate.
        role_mapping: Optional dict mapping speakers to roles.

    Returns:
        List of ShareGPT turn dicts, or None on failure.
    """
    persona_names = [name for name, _ in personas]

    messages = build_continuation_messages(
        personas=personas,
        prior_turns=prior_turns,
        topic=topic,
        scenario=scenario,
        style=style,
    )

    text = _call_pipeline(
        generator_pipeline,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
    )

    if not text:
        logger.warning("generate_continuation: pipeline returned no text.")
        return None

    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping
    )

    if turns is None:
        logger.warning("generate_continuation: failed to parse continuation output.")
        return None

    _add_speaker_names(turns, text, persona_names)

    return turns
