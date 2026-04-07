"""Pure functions that build message lists for LLM calls. No LLM calls here."""

from __future__ import annotations

ARG_GENERATION_SYSTEM_PROMPT = """\
You are a creative writing assistant that generates conversation parameters.
Given a creative brief, produce CLI arguments in the following format:

--persona1 "Name of the first character"
--persona1-desc "Description of the first character's personality and background"
--persona2 "Name of the second character"
--persona2-desc "Description of the second character's personality and background"
--topic "The main topic of conversation"
--scenario "The setting or situation where the conversation takes place"
--style "The conversational style (e.g., Casual, Formal, Dramatic, Humorous)"
--include-points "Optional comma-separated points to cover in the conversation"

Rules:
- Always output all required fields (persona1, persona1-desc, persona2, persona2-desc, topic, scenario, style)
- Use double or single quotes around each value
- One argument per line
- Values must be concise but descriptive
- The include-points field is optional
"""

TOPIC_VARIATION_SYSTEM_PROMPT = """\
You are a creative writing assistant that generates topic variations for conversations.
Given information about two personas and an existing conversation setup, produce new
topic, scenario, and style parameters in the following format:

--topic "A new topic for the conversation"
--scenario "A new setting or situation"
--style "A new conversational style"

Rules:
- Always output all three fields (topic, scenario, style)
- Use double or single quotes around each value
- One argument per line
- Keep the personas in mind — the new topic should suit their personalities
- Be creative and varied; avoid repeating the initial setup
"""


def build_conversation_messages(
    topic: str,
    persona1: str,
    persona2: str,
    persona1_desc: str,
    persona2_desc: str,
    scenario: str,
    style: str,
    include_points: str | None = None,
) -> list[dict]:
    """Build system + user messages for generating a conversation.

    Args:
        topic: The main topic of conversation.
        persona1: Name of the first speaker.
        persona2: Name of the second speaker.
        persona1_desc: Description of persona1's personality/background.
        persona2_desc: Description of persona2's personality/background.
        scenario: The setting or situation for the conversation.
        style: The conversational style.
        include_points: Optional comma-separated points to include.

    Returns:
        List with two dicts: system message and user message.
    """
    system_content = (
        f"You are writing a realistic dialogue between two characters.\n\n"
        f"Characters:\n"
        f"- {persona1}: {persona1_desc}\n"
        f"- {persona2}: {persona2_desc}\n\n"
        f"Topic: {topic}\n"
        f"Scenario: {scenario}\n"
        f"Style: {style}\n\n"
        f"Format each turn as:\n"
        f"{persona1}: <dialogue>\n"
        f"{persona2}: <dialogue>\n\n"
        f"Write a natural, engaging conversation that reflects each character's "
        f"personality. Avoid repetition and keep it flowing naturally."
    )

    user_content = "Generate a conversation between the two characters."
    if include_points:
        user_content += (
            f"\n\nMake sure to include the following points: {include_points}"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_arg_generation_messages(
    brief: str,
    persona1_context: str | None = None,
    persona2_context: str | None = None,
    search_term1: str | None = None,
    search_term2: str | None = None,
) -> list[dict]:
    """Build system + user messages for generating conversation arguments from a brief.

    Args:
        brief: The creative brief describing the desired conversation.
        persona1_context: Optional web search context for persona1.
        persona2_context: Optional web search context for persona2.
        search_term1: Optional search term used to retrieve persona1_context.
        search_term2: Optional search term used to retrieve persona2_context.

    Returns:
        List with two dicts: system message and user message.
    """
    system_content = ARG_GENERATION_SYSTEM_PROMPT

    if persona1_context or persona2_context:
        system_content += "\n\n## Web Context\n"
        if persona1_context:
            label = search_term1 if search_term1 else "Persona 1"
            system_content += f"\n### {label}\n{persona1_context}\n"
        if persona2_context:
            label = search_term2 if search_term2 else "Persona 2"
            system_content += f"\n### {label}\n{persona2_context}\n"

    user_content = f"Generate conversation parameters for the following brief:\n\n{brief}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_variation_messages(
    persona1: str,
    persona1_desc: str,
    persona2: str,
    persona2_desc: str,
    initial_topic: str,
    initial_scenario: str,
    initial_style: str,
    original_brief: str | None = None,
) -> list[dict]:
    """Build system + user messages for generating topic/scenario/style variations.

    Args:
        persona1: Name of the first persona.
        persona1_desc: Description of persona1.
        persona2: Name of the second persona.
        persona2_desc: Description of persona2.
        initial_topic: The original topic to vary from.
        initial_scenario: The original scenario to vary from.
        initial_style: The original style to vary from.
        original_brief: Optional original creative brief for additional context.

    Returns:
        List with two dicts: system message and user message.
    """
    user_content = (
        f"Generate a new topic, scenario, and style variation for a conversation "
        f"between the following characters:\n\n"
        f"- {persona1}: {persona1_desc}\n"
        f"- {persona2}: {persona2_desc}\n\n"
        f"Original setup:\n"
        f"- Topic: {initial_topic}\n"
        f"- Scenario: {initial_scenario}\n"
        f"- Style: {initial_style}\n"
    )

    if original_brief:
        user_content += f"\nOriginal brief: {original_brief}\n"

    user_content += (
        "\nProduce a creative variation that fits these characters but explores "
        "a different topic, setting, or tone."
    )

    return [
        {"role": "system", "content": TOPIC_VARIATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
