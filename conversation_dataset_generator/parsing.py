"""Parsers for converting raw LLM output into structured formats."""

import logging
import re

logger = logging.getLogger(__name__)


def parse_conversation_to_sharegpt(
    conversation_text: str,
    persona1: str | None = None,
    persona2: str | None = None,
    role_mapping: dict | None = None,
    *,
    personas: list[str] | None = None,
) -> tuple[list[dict] | None, list[str] | None]:
    """Parse raw conversation text into ShareGPT turn structure.

    Args:
        conversation_text: Raw text output from the LLM.
        persona1: Name of the first speaker (legacy 2-speaker interface).
        persona2: Name of the second speaker (legacy 2-speaker interface).
        role_mapping: Optional dict mapping speakers to roles.
            Accepts legacy format {"p1": "human", "p2": "gpt"} or
            name-based format {"Alice": "human", "Bob": "gpt"}.
            Defaults to first speaker -> "human", all others -> "gpt".
        personas: List of N speaker names (keyword-only).

    Returns:
        Tuple of (turns_list, persona_names_list) or (None, None).
    """
    if not conversation_text or not conversation_text.strip():
        return None, None

    # Resolve personas list from either new or legacy args
    if personas is None:
        if persona1 and persona2:
            personas = [persona1, persona2]
        else:
            return None, None

    # Resolve role_mapping
    if role_mapping is None:
        # Default: first speaker -> "human", rest -> "gpt"
        resolved_mapping = {personas[0].lower(): "human"}
        for name in personas[1:]:
            resolved_mapping[name.lower()] = "gpt"
    elif "p1" in role_mapping or "p2" in role_mapping:
        # Legacy format: convert p1/p2 keys to name-based
        resolved_mapping = {}
        for i, name in enumerate(personas):
            legacy_key = f"p{i + 1}"
            if legacy_key in role_mapping:
                resolved_mapping[name.lower()] = role_mapping[legacy_key]
    else:
        # Name-based format: normalize keys to lowercase
        resolved_mapping = {k.lower(): v for k, v in role_mapping.items()}

    # Build regex pattern matching any persona name
    escaped_names = "|".join(re.escape(name) for name in personas)
    turn_pattern = re.compile(
        rf"^\s*({escaped_names})\s*:\s*(.*)",
        re.IGNORECASE | re.MULTILINE,
    )

    conversations = []
    current_turn = None
    accumulated_text = ""

    for line in conversation_text.strip().split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        match = turn_pattern.match(line_stripped)
        if match:
            if current_turn is not None:
                current_turn["value"] = accumulated_text.strip()
                if current_turn["value"]:
                    conversations.append(current_turn)

            speaker = match.group(1).strip()
            value_start = match.group(2).strip()

            role = resolved_mapping.get(speaker.lower())
            if role is None:
                current_turn = None
                accumulated_text = ""
                continue

            current_turn = {"from": role, "value": None}
            accumulated_text = value_start

        elif current_turn is not None:
            accumulated_text += "\n" + line_stripped
        else:
            logger.warning("Skipping unmatched line: %s", line_stripped)

    if current_turn is not None:
        current_turn["value"] = accumulated_text.strip()
        if current_turn["value"]:
            conversations.append(current_turn)

    if not conversations:
        logger.warning("Could not parse any valid turns from conversation text.")
        return None, None

    return conversations, personas


def parse_variation_output(text: str) -> dict | None:
    """Parse LLM variation output into topic/scenario/style dict.

    Handles both one-arg-per-line and all-args-on-one-line formats.
    Supports both single and double quotes. No re.DOTALL.

    Returns:
        Dict with 'topic', 'scenario', and optionally 'style' keys,
        or None if topic or scenario are missing.
    """
    if not text or not text.strip():
        return None

    result = {}
    # Match --key "value" or --key 'value' anywhere in text (not just line-anchored)
    pattern = re.compile(
        r"--(topic|scenario|style)\s+[\"'](.+?)[\"']",
    )

    for match in pattern.finditer(text):
        key = match.group(1)
        value = match.group(2).strip()
        result[key] = value

    if "topic" not in result or "scenario" not in result:
        logger.warning(
            "Variation parse failed — missing topic or scenario. "
            "Matched keys: %s. Raw text:\n%s",
            list(result.keys()),
            text,
        )
        return None

    return result


_ARG_KEYS = (
    "persona1",
    "persona1-desc",
    "persona2",
    "persona2-desc",
    "topic",
    "scenario",
    "style",
    "include-points",
)

_REQUIRED_ARG_KEYS = (
    "persona1",
    "persona1_desc",
    "persona2",
    "persona2_desc",
    "topic",
    "scenario",
    "style",
)


def parse_arg_generation_output(text: str) -> dict | None:
    """Parse LLM arg-generation output into a dict of generation parameters.

    Same regex fix as parse_variation_output: no re.DOTALL, strips lines,
    supports both quote styles.

    Returns:
        Dict with keys like 'persona1', 'persona1_desc', 'topic', etc.
        Keys are normalized (hyphens -> underscores).
        Returns None if any required key is missing.
    """
    if not text or not text.strip():
        return None

    keys_pattern = "|".join(re.escape(k) for k in _ARG_KEYS)
    # Match --key "value" anywhere in text (handles both one-per-line and inline formats)
    pattern = re.compile(
        rf"--({keys_pattern})\s+[\"'](.+?)[\"']",
    )

    result = {}
    for match in pattern.finditer(text):
        key = match.group(1).replace("-", "_")
        value = match.group(2).strip()
        result[key] = value

    missing = [k for k in _REQUIRED_ARG_KEYS if k not in result]
    if missing:
        logger.warning(
            "Arg generation parse missing required keys: %s. "
            "Matched keys: %s. Raw text:\n%s",
            missing,
            list(result.keys()),
            text,
        )
        return None

    return result
