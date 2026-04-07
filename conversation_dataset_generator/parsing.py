"""Parsers for converting raw LLM output into structured formats."""

import logging
import re

logger = logging.getLogger(__name__)


def parse_conversation_to_sharegpt(
    conversation_text: str,
    persona1: str,
    persona2: str,
    role_mapping: dict | None = None,
) -> tuple[list[dict] | None, str | None, str | None]:
    """Parse raw conversation text into ShareGPT turn structure.

    Args:
        conversation_text: Raw text output from the LLM.
        persona1: Name of the first speaker.
        persona2: Name of the second speaker.
        role_mapping: Optional dict with keys "p1" and "p2" mapping to roles.
            Defaults to {"p1": "human", "p2": "gpt"}.

    Returns:
        Tuple of (turns_list, persona1_name, persona2_name) or (None, None, None).
    """
    if not conversation_text or not conversation_text.strip():
        return None, None, None

    if role_mapping is None:
        role_mapping = {"p1": "human", "p2": "gpt"}

    turn_pattern = re.compile(
        rf"^\s*({re.escape(persona1)}|{re.escape(persona2)})\s*:\s*(.*)",
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

            if speaker.lower() == persona1.lower():
                role = role_mapping["p1"]
            elif speaker.lower() == persona2.lower():
                role = role_mapping["p2"]
            else:
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
        return None, None, None

    return conversations, persona1, persona2


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
