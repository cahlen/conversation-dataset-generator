# Conversation Dataset Generator Modernization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 1800-line generate.py monolith into a testable Python package, fix the broken variation regex, remove dead features, and update defaults.

**Architecture:** Python package `conversation_dataset_generator/` with 9 focused modules. TDD throughout — tests first, then implementation. The existing `generate.py` becomes a thin entry point. CLI contract preserved for backward compat with `batch_generate.py` and YAML configs.

**Tech Stack:** Python 3.10+, transformers, torch, datasets, huggingface_hub, pyyaml, duckduckgo-search, tqdm, bitsandbytes, pytest

---

## File Map

### New files to create:
- `conversation_dataset_generator/__init__.py` — package init, version
- `conversation_dataset_generator/parsing.py` — ShareGPT parser, arg output parser, variation output parser
- `conversation_dataset_generator/prompts.py` — all system prompts and prompt builder functions
- `conversation_dataset_generator/character_pool.py` — YAML pool loading, validation, random pair selection
- `conversation_dataset_generator/output.py` — JSONL writing, dataset card templates, HF dataset creation
- `conversation_dataset_generator/web_search.py` — DuckDuckGo persona context search
- `conversation_dataset_generator/models.py` — model/tokenizer loading, pipeline creation
- `conversation_dataset_generator/generation.py` — LLM call wrappers with retry
- `conversation_dataset_generator/cli.py` — argparse, mode detection, orchestration loop
- `tests/__init__.py`
- `tests/test_parsing.py`
- `tests/test_prompts.py`
- `tests/test_character_pool.py`
- `tests/test_output.py`
- `tests/test_generation.py`
- `tests/test_cli.py`
- `requirements-dev.txt`

### Files to modify:
- `generate.py` — replace with thin entry point
- `batch_generate.py` — update default model reference in comments only
- `requirements.txt` — remove pandas, peft, trl; add pyyaml explicitly

### Files to delete:
- None (old generate.py content is replaced, not deleted)

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `conversation_dataset_generator/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`

- [ ] **Step 1: Create package directory and __init__.py**

```python
# conversation_dataset_generator/__init__.py
"""Conversation Dataset Generator — synthetic dialogue data for LLM fine-tuning."""

__version__ = "2.0.0"
```

- [ ] **Step 2: Create tests directory and __init__.py**

```python
# tests/__init__.py
```

(Empty file, just marks it as a package.)

- [ ] **Step 3: Create requirements-dev.txt**

```
pytest
```

- [ ] **Step 4: Update requirements.txt**

```
torch
transformers
accelerate
datasets
huggingface_hub
pyyaml
duckduckgo-search
tqdm>=4.62.0
bitsandbytes
```

(Removed: `pandas`, `peft`, `trl`. Added: `pyyaml` explicitly.)

- [ ] **Step 5: Install dev dependencies and verify**

Run: `pip install -r requirements-dev.txt`
Expected: pytest installs successfully

Run: `pytest --version`
Expected: prints pytest version

- [ ] **Step 6: Commit**

```bash
git add conversation_dataset_generator/__init__.py tests/__init__.py requirements.txt requirements-dev.txt
git commit -m "scaffold package structure and update dependencies"
```

---

### Task 2: Parsing Module (Critical Bug Fix)

This is the most important module — it contains the broken variation regex. Pure functions, no dependencies on LLM or torch.

**Files:**
- Create: `conversation_dataset_generator/parsing.py`
- Create: `tests/test_parsing.py`

- [ ] **Step 1: Write failing tests for parse_conversation_to_sharegpt**

```python
# tests/test_parsing.py
import pytest
from conversation_dataset_generator.parsing import parse_conversation_to_sharegpt


class TestParseConversationToSharegpt:
    def test_basic_two_turn(self):
        text = "Alice: Hello there!\nBob: Hi Alice, how are you?"
        turns, p1, p2 = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert turns == [
            {"from": "human", "value": "Hello there!"},
            {"from": "gpt", "value": "Hi Alice, how are you?"},
        ]
        assert p1 == "Alice"
        assert p2 == "Bob"

    def test_multi_turn(self):
        text = (
            "Alice: First line\n"
            "Bob: Second line\n"
            "Alice: Third line\n"
            "Bob: Fourth line"
        )
        turns, p1, p2 = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 4
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "human"
        assert turns[3]["from"] == "gpt"

    def test_multiline_turn(self):
        text = (
            "Alice: This is line one\n"
            "and this continues the same turn\n"
            "Bob: Got it"
        )
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert "line one\nand this continues" in turns[0]["value"]

    def test_case_insensitive_matching(self):
        text = "alice: Hello\nBOB: Hi"
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2

    def test_empty_text_returns_none(self):
        turns, p1, p2 = parse_conversation_to_sharegpt("", "Alice", "Bob")
        assert turns is None
        assert p1 is None
        assert p2 is None

    def test_no_matching_speakers_returns_none(self):
        text = "Charlie: Hello\nDave: Hi"
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert turns is None

    def test_blank_lines_ignored(self):
        text = "Alice: Hello\n\n\nBob: Hi"
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2

    def test_trailing_whitespace_on_lines(self):
        text = "Alice: Hello   \nBob: Hi   "
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert turns[0]["value"] == "Hello"
        assert turns[1]["value"] == "Hi"

    def test_custom_role_mapping(self):
        text = "Alice: Hello\nBob: Hi"
        turns, _, _ = parse_conversation_to_sharegpt(
            text, "Alice", "Bob", role_mapping={"p1": "gpt", "p2": "human"}
        )
        assert turns[0]["from"] == "gpt"
        assert turns[1]["from"] == "human"

    def test_skips_empty_turns(self):
        text = "Alice: \nBob: Hi there"
        turns, _, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 1
        assert turns[0]["from"] == "gpt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parsing.py::TestParseConversationToSharegpt -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'conversation_dataset_generator.parsing'`

- [ ] **Step 3: Implement parse_conversation_to_sharegpt**

```python
# conversation_dataset_generator/parsing.py
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
            # Save previous turn
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

    # Save final turn
    if current_turn is not None:
        current_turn["value"] = accumulated_text.strip()
        if current_turn["value"]:
            conversations.append(current_turn)

    if not conversations:
        logger.warning("Could not parse any valid turns from conversation text.")
        return None, None, None

    return conversations, persona1, persona2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parsing.py::TestParseConversationToSharegpt -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Write failing tests for parse_variation_output (the critical bug fix)**

Add to `tests/test_parsing.py`:

```python
from conversation_dataset_generator.parsing import parse_variation_output


class TestParseVariationOutput:
    def test_basic_three_fields(self):
        text = (
            '--topic "New topic here"\n'
            '--scenario "New scenario here"\n'
            '--style "New style here"'
        )
        result = parse_variation_output(text)
        assert result == {
            "topic": "New topic here",
            "scenario": "New scenario here",
            "style": "New style here",
        }

    def test_trailing_whitespace_on_lines(self):
        text = (
            '--topic "Topic value"   \n'
            '--scenario "Scenario value"  \n'
            '--style "Style value"  '
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Topic value"
        assert result["scenario"] == "Scenario value"
        assert result["style"] == "Style value"

    def test_single_quotes(self):
        text = (
            "--topic 'Single quoted topic'\n"
            "--scenario 'Single quoted scenario'\n"
            "--style 'Single quoted style'"
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Single quoted topic"

    def test_mixed_quotes(self):
        text = (
            '--topic "Double quoted"\n'
            "--scenario 'Single quoted'\n"
            '--style "Another double"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert len(result) == 3

    def test_missing_topic_returns_none(self):
        text = (
            '--scenario "Only scenario"\n'
            '--style "Only style"'
        )
        result = parse_variation_output(text)
        assert result is None

    def test_missing_scenario_returns_none(self):
        text = (
            '--topic "Only topic"\n'
            '--style "Only style"'
        )
        result = parse_variation_output(text)
        assert result is None

    def test_style_optional(self):
        text = (
            '--topic "Just topic"\n'
            '--scenario "Just scenario"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert "style" not in result

    def test_preamble_text_ignored(self):
        text = (
            "Here are the new parameters:\n"
            '--topic "Actual topic"\n'
            '--scenario "Actual scenario"\n'
            '--style "Actual style"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Actual topic"

    def test_empty_string_returns_none(self):
        result = parse_variation_output("")
        assert result is None

    def test_garbage_input_returns_none(self):
        result = parse_variation_output("This is just random text with no args")
        assert result is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_parsing.py::TestParseVariationOutput -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Implement parse_variation_output**

Add to `conversation_dataset_generator/parsing.py`:

```python
def parse_variation_output(text: str) -> dict | None:
    """Parse LLM variation output into topic/scenario/style dict.

    Fixes the critical regex bug: no re.DOTALL, strips lines, supports
    both single and double quotes.

    Returns:
        Dict with 'topic', 'scenario', and optionally 'style' keys,
        or None if topic or scenario are missing.
    """
    if not text or not text.strip():
        return None

    result = {}
    pattern = re.compile(
        r"^--(topic|scenario|style)\s+[\"'](.+?)[\"']\s*$",
        re.MULTILINE,
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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_parsing.py::TestParseVariationOutput -v`
Expected: All 10 tests PASS

- [ ] **Step 9: Write failing tests for parse_arg_generation_output**

Add to `tests/test_parsing.py`:

```python
from conversation_dataset_generator.parsing import parse_arg_generation_output


class TestParseArgGenerationOutput:
    def test_all_required_fields(self):
        text = (
            '--persona1 "Alice"\n'
            '--persona1-desc "A friendly person"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "A grumpy person"\n'
            '--topic "Weather"\n'
            '--scenario "At a bus stop"\n'
            '--style "Casual chat"'
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["persona1"] == "Alice"
        assert result["persona1_desc"] == "A friendly person"
        assert result["persona2"] == "Bob"
        assert result["persona2_desc"] == "A grumpy person"
        assert result["topic"] == "Weather"
        assert result["scenario"] == "At a bus stop"
        assert result["style"] == "Casual chat"

    def test_with_include_points(self):
        text = (
            '--persona1 "Alice"\n'
            '--persona1-desc "Desc"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "Desc"\n'
            '--topic "Topic"\n'
            '--scenario "Scenario"\n'
            '--style "Style"\n'
            '--include-points "rain, sun, wind"'
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["include_points"] == "rain, sun, wind"

    def test_trailing_whitespace(self):
        text = (
            '--persona1 "Alice"   \n'
            '--persona1-desc "Desc"   \n'
            '--persona2 "Bob"   \n'
            '--persona2-desc "Desc"   \n'
            '--topic "Topic"   \n'
            '--scenario "Scenario"   \n'
            '--style "Style"   '
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["persona1"] == "Alice"

    def test_single_quotes(self):
        text = (
            "--persona1 'Alice'\n"
            "--persona1-desc 'Desc'\n"
            "--persona2 'Bob'\n"
            "--persona2-desc 'Desc'\n"
            "--topic 'Topic'\n"
            "--scenario 'Scenario'\n"
            "--style 'Style'"
        )
        result = parse_arg_generation_output(text)
        assert result is not None

    def test_missing_persona1_returns_none(self):
        text = (
            '--persona1-desc "Desc"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "Desc"\n'
            '--topic "Topic"\n'
            '--scenario "Scenario"\n'
            '--style "Style"'
        )
        result = parse_arg_generation_output(text)
        assert result is None

    def test_empty_returns_none(self):
        result = parse_arg_generation_output("")
        assert result is None
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_parsing.py::TestParseArgGenerationOutput -v`
Expected: FAIL — `ImportError`

- [ ] **Step 11: Implement parse_arg_generation_output**

Add to `conversation_dataset_generator/parsing.py`:

```python
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
    pattern = re.compile(
        rf"^--({keys_pattern})\s+[\"'](.+?)[\"']\s*$",
        re.MULTILINE,
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
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `pytest tests/test_parsing.py::TestParseArgGenerationOutput -v`
Expected: All 6 tests PASS

- [ ] **Step 13: Run all parsing tests**

Run: `pytest tests/test_parsing.py -v`
Expected: All 26 tests PASS

- [ ] **Step 14: Commit**

```bash
git add conversation_dataset_generator/parsing.py tests/test_parsing.py
git commit -m "add parsing module with fixed variation regex and full test coverage"
```

---

### Task 3: Prompts Module

Pure functions that build message lists. No LLM calls.

**Files:**
- Create: `conversation_dataset_generator/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompts.py
import pytest
from conversation_dataset_generator.prompts import (
    build_conversation_messages,
    build_arg_generation_messages,
    build_variation_messages,
)


class TestBuildConversationMessages:
    def test_returns_two_messages(self):
        msgs = build_conversation_messages(
            topic="Weather",
            persona1="Alice",
            persona2="Bob",
            persona1_desc="Friendly",
            persona2_desc="Grumpy",
            scenario="Bus stop",
            style="Casual",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_personas_in_system_message(self):
        msgs = build_conversation_messages(
            topic="Weather",
            persona1="Alice",
            persona2="Bob",
            persona1_desc="Friendly",
            persona2_desc="Grumpy",
            scenario="Bus stop",
            style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]
        assert "Friendly" in msgs[0]["content"]
        assert "Grumpy" in msgs[0]["content"]

    def test_topic_and_scenario_in_system(self):
        msgs = build_conversation_messages(
            topic="Quantum physics",
            persona1="A",
            persona2="B",
            persona1_desc="d1",
            persona2_desc="d2",
            scenario="Coffee shop",
            style="Educational",
        )
        assert "Quantum physics" in msgs[0]["content"]
        assert "Coffee shop" in msgs[0]["content"]

    def test_include_points_in_user_message(self):
        msgs = build_conversation_messages(
            topic="T",
            persona1="A",
            persona2="B",
            persona1_desc="d1",
            persona2_desc="d2",
            scenario="S",
            style="St",
            include_points="rain, sun, wind",
        )
        assert "rain" in msgs[1]["content"]

    def test_no_include_points(self):
        msgs = build_conversation_messages(
            topic="T",
            persona1="A",
            persona2="B",
            persona1_desc="d1",
            persona2_desc="d2",
            scenario="S",
            style="St",
        )
        # Should not crash, should still have 2 messages
        assert len(msgs) == 2


class TestBuildArgGenerationMessages:
    def test_returns_two_messages(self):
        msgs = build_arg_generation_messages(brief="Sherlock meets Watson")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_brief_in_user_message(self):
        msgs = build_arg_generation_messages(brief="Sherlock meets Watson")
        assert "Sherlock meets Watson" in msgs[1]["content"]

    def test_web_context_appended(self):
        msgs = build_arg_generation_messages(
            brief="Test brief",
            persona1_context="Context about persona 1",
            persona2_context="Context about persona 2",
            search_term1="Term1",
            search_term2="Term2",
        )
        assert "Context about persona 1" in msgs[0]["content"]
        assert "Context about persona 2" in msgs[0]["content"]


class TestBuildVariationMessages:
    def test_returns_two_messages(self):
        msgs = build_variation_messages(
            persona1="A",
            persona1_desc="d1",
            persona2="B",
            persona2_desc="d2",
            initial_topic="T",
            initial_scenario="S",
            initial_style="St",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_personas_in_user_context(self):
        msgs = build_variation_messages(
            persona1="Sherlock",
            persona1_desc="Detective",
            persona2="Watson",
            persona2_desc="Doctor",
            initial_topic="Crime",
            initial_scenario="Baker Street",
            initial_style="Dramatic",
        )
        assert "Sherlock" in msgs[1]["content"]
        assert "Watson" in msgs[1]["content"]

    def test_original_brief_included(self):
        msgs = build_variation_messages(
            persona1="A",
            persona1_desc="d1",
            persona2="B",
            persona2_desc="d2",
            initial_topic="T",
            initial_scenario="S",
            initial_style="St",
            original_brief="Original brief text here",
        )
        assert "Original brief text here" in msgs[1]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement prompts.py**

```python
# conversation_dataset_generator/prompts.py
"""System prompts and message builders for LLM interactions."""

ARG_GENERATION_SYSTEM_PROMPT = """You are an expert creative assistant specializing in setting up parameters for a conversational dialogue generation script. Your goal is to take a user's request, which typically names two entities (characters, concepts, etc.), and generate a complete set of *realistic* and *grounded* arguments suitable for the `generate.py` script.

The `generate.py` script requires the following arguments:
--persona1 "<name>"
--persona1-desc "<detailed description, including potential conversational tics or speech patterns>"
--persona2 "<name>"
--persona2-desc "<detailed description, including potential conversational tics or speech patterns>"
--topic "<plausible conversation topic>"
--scenario "<realistic setting/context>"
--style "<dialogue style/tone - aim for natural interaction>"
--include-points "<comma-separated keywords>" (Optional, but helpful)

Your Task:
1.  Analyze the user's request (e.g., "Generate a conversation between Character A and Character B").
2.  Identify the two main personas.
3.  Write detailed descriptions that capture the essence of each persona. Consider the context or era implied by the request. Focus on how they might *actually speak*, incorporating characteristic slang, attitude, and speech patterns. Include potential conversational tics.
4.  Determine a *plausible* topic grounded in their personas, a shared interest, recent event, or a potential point of disagreement.
5.  Define a *realistic* scenario where this conversation might naturally occur.
6.  Describe the desired style of the dialogue, focusing on natural interaction.
7.  (Optional) List relevant keywords as include-points that should appear naturally.
8.  Format your entire output *strictly* as key-value pairs, with each argument on a new line. Use double quotes around the values. Do not include any other text.

Example Output:
--persona1 "Barnaby"
--persona1-desc "An older tabby cat. Values quiet. Communicates with sighs and minimal meows."
--persona2 "Sunshine"
--persona2-desc "A young golden retriever. High energy. Often whines or yips excitedly."
--topic "Trying to share the same small patch of sun"
--scenario "Both attempting to lie down on a rug near a sunny window"
--style "Comedic contrast, annoyed brevity vs. cheerful rambling"
--include-points "sunbeam, nap, tail wag, sigh, bark, space"

Now, analyze the user's request and generate the arguments."""

TOPIC_VARIATION_SYSTEM_PROMPT = """You are an expert creative assistant helping refine parameters for dialogue generation. You will be given fixed personas and initial context.

Your Task:
1.  Review the fixed personas and context.
2.  Generate a *new*, *related but distinct* --topic and --scenario for a conversation between these personas. The new topic/scenario should fit the spirit of the original but offer variety.
3.  Optionally, suggest a slightly adjusted --style if it makes sense, otherwise reuse the original style.
4.  Format your output *strictly* as key-value pairs for --topic, --scenario, and --style ONLY. Use double quotes around the values. Each on a new line. Do not include any other text.

Example Output:
--topic "The strategic placement of condiments at a diner table"
--scenario "Still at Monk's diner, waiting for food"
--style "Observational comedy, escalating neuroticism about minor details"

Now, analyze the provided context and generate a new topic, scenario, and style variation."""


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
    """Build chat messages for conversation generation."""
    system_message = (
        f"You are a creative assistant skilled at generating *realistic* and "
        f"*natural-sounding* conversational dialogue between two described personas "
        f"in a specific scenario.\n"
        f"The conversation should be between {persona1} ({persona1_desc}) and "
        f"{persona2} ({persona2_desc}).\n"
        f"The scenario is: '{scenario}'.\n"
        f"The central topic is: '{topic}'.\n"
        f"The requested interaction style is: '{style}'.\n\n"
        f"**IMPORTANT: Aim for a realistic, spontaneous conversation.** Avoid overly "
        f"formal, dramatic, philosophical, or scripted-sounding language unless truly "
        f"fitting for the specific personas and situation. Incorporate natural "
        f"conversational elements like brief pauses, slight hesitations (use '...' "
        f"sparingly), agreeing/disagreeing naturally, or occasional minor topic shifts. "
        f"Focus on realism over perfect grammatical structure or constant back-and-forth "
        f"argument. The conversation should have a **natural length, perhaps 5-15 turns.**\n\n"
        f"Start the output directly with the first turn (e.g., '{persona1}: ...'). "
        f"Do not include any preamble or explanatory text outside the dialogue turns."
    )

    user_request = (
        f"Generate the conversation now, following all the instructions in the system "
        f"message, especially the emphasis on naturalness and realism. Make sure each "
        f"turn starts with either '{persona1}:' or '{persona2}:'."
    )

    if include_points:
        points_list = [p.strip() for p in include_points.split(",") if p.strip()]
        if points_list:
            user_request += (
                f" Try to naturally incorporate discussion of the following "
                f"points/keywords if possible: {', '.join(points_list)}."
            )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_request},
    ]


def build_arg_generation_messages(
    brief: str,
    persona1_context: str | None = None,
    persona2_context: str | None = None,
    search_term1: str | None = None,
    search_term2: str | None = None,
) -> list[dict]:
    """Build chat messages for generating args from a creative brief."""
    system_content = ARG_GENERATION_SYSTEM_PROMPT

    if persona1_context or persona2_context:
        system_content += "\n\n--- Additional Context from Web Search ---"
        if persona1_context:
            label = search_term1 or "Persona 1"
            system_content += f"\nContext for '{label}':\n{persona1_context}"
        if persona2_context:
            label = search_term2 or "Persona 2"
            system_content += f"\nContext for '{label}':\n{persona2_context}"
        system_content += "\n--- End of Context ---"
        system_content += (
            "\n\nUse the provided web context to inform the persona descriptions "
            "where appropriate, ensuring they are realistic and grounded."
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": brief},
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
    """Build chat messages for generating topic/scenario variations."""
    if original_brief:
        context = (
            f'Original Brief: "{original_brief}"\n'
            f"Persona 1: {persona1} ({persona1_desc})\n"
            f"Persona 2: {persona2} ({persona2_desc})\n"
            f'Initial Topic: "{initial_topic}"\n'
            f'Initial Scenario: "{initial_scenario}"\n'
            f'Initial Style: "{initial_style}"'
        )
    else:
        context = (
            f"Fixed Persona 1: {persona1} ({persona1_desc})\n"
            f"Fixed Persona 2: {persona2} ({persona2_desc})\n"
            f'Initial Topic: "{initial_topic}"\n'
            f'Initial Scenario: "{initial_scenario}"\n'
            f'Initial Style: "{initial_style}"\n\n'
            f"Generate a NEW, related topic and scenario based on the INITIAL "
            f"context above, while keeping the personas in mind."
        )

    return [
        {"role": "system", "content": TOPIC_VARIATION_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/prompts.py tests/test_prompts.py
git commit -m "add prompts module with system prompts and message builders"
```

---

### Task 4: Character Pool Module

**Files:**
- Create: `conversation_dataset_generator/character_pool.py`
- Create: `tests/test_character_pool.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_character_pool.py
import os
import pytest
import tempfile
import yaml
from conversation_dataset_generator.character_pool import (
    load_character_pool,
    load_description_pool,
    validate_pools,
    select_random_pair,
)


@pytest.fixture
def pool_dir(tmp_path):
    """Create temporary YAML pool files for testing."""
    characters = {"characters": ["Alice", "Bob", "Charlie"]}
    descriptions = {
        "descriptions": {
            "Alice": "A friendly person",
            "Bob": "A grumpy person",
            "Charlie": "A quiet person",
        }
    }
    char_file = tmp_path / "characters.yaml"
    desc_file = tmp_path / "descriptions.yaml"
    char_file.write_text(yaml.dump(characters))
    desc_file.write_text(yaml.dump(descriptions))
    return str(char_file), str(desc_file)


class TestLoadCharacterPool:
    def test_loads_characters(self, pool_dir):
        char_file, _ = pool_dir
        pool = load_character_pool(char_file)
        assert pool == ["Alice", "Bob", "Charlie"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_character_pool("/nonexistent/file.yaml")

    def test_missing_characters_key_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(yaml.dump({"wrong_key": ["a", "b"]}))
        with pytest.raises(ValueError, match="characters"):
            load_character_pool(str(bad_file))

    def test_too_few_characters_raises(self, tmp_path):
        bad_file = tmp_path / "one.yaml"
        bad_file.write_text(yaml.dump({"characters": ["OnlyOne"]}))
        with pytest.raises(ValueError, match="at least 2"):
            load_character_pool(str(bad_file))


class TestLoadDescriptionPool:
    def test_loads_descriptions(self, pool_dir):
        _, desc_file = pool_dir
        pool = load_description_pool(desc_file)
        assert pool["Alice"] == "A friendly person"

    def test_missing_descriptions_key_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(yaml.dump({"wrong_key": {}}))
        with pytest.raises(ValueError, match="descriptions"):
            load_description_pool(str(bad_file))


class TestValidatePools:
    def test_valid_pools_pass(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        validate_pools(characters, descriptions)  # Should not raise

    def test_missing_description_raises(self):
        characters = ["Alice", "Bob", "Charlie"]
        descriptions = {"Alice": "Desc", "Bob": "Desc"}  # Missing Charlie
        with pytest.raises(ValueError, match="Charlie"):
            validate_pools(characters, descriptions)


class TestSelectRandomPair:
    def test_returns_two_different_characters(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        p1_name, p1_desc, p2_name, p2_desc = select_random_pair(
            characters, descriptions
        )
        assert p1_name != p2_name
        assert p1_name in characters
        assert p2_name in characters
        assert p1_desc == descriptions[p1_name]
        assert p2_desc == descriptions[p2_name]

    def test_many_selections_never_duplicate(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        for _ in range(50):
            p1, _, p2, _ = select_random_pair(characters, descriptions)
            assert p1 != p2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_character_pool.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement character_pool.py**

```python
# conversation_dataset_generator/character_pool.py
"""Character pool loading, validation, and random pair selection."""

import logging
import random

import yaml

logger = logging.getLogger(__name__)


def load_character_pool(path: str) -> list[str]:
    """Load character names from a YAML file.

    Expects a file with a top-level 'characters' key containing a list of names.
    Raises FileNotFoundError, ValueError, or yaml.YAMLError.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "characters" not in data:
        raise ValueError(
            f"Character pool YAML must contain a 'characters' key. "
            f"Found keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    characters = data["characters"]
    if not isinstance(characters, list) or len(characters) < 2:
        raise ValueError(
            f"Character pool must be a list with at least 2 characters. "
            f"Found {len(characters) if isinstance(characters, list) else 0}."
        )

    logger.info("Loaded %d characters from %s", len(characters), path)
    return characters


def load_description_pool(path: str) -> dict[str, str]:
    """Load character descriptions from a YAML file.

    Expects a file with a top-level 'descriptions' key containing a dict.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "descriptions" not in data:
        raise ValueError(
            f"Description pool YAML must contain a 'descriptions' key. "
            f"Found keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    descriptions = data["descriptions"]
    if not isinstance(descriptions, dict):
        raise ValueError(
            f"Descriptions must be a dictionary. Found: {type(descriptions)}"
        )

    logger.info("Loaded %d descriptions from %s", len(descriptions), path)
    return descriptions


def validate_pools(
    characters: list[str], descriptions: dict[str, str]
) -> None:
    """Validate that all characters have descriptions. Raises ValueError if not."""
    missing = [c for c in characters if c not in descriptions]
    if missing:
        raise ValueError(
            f"Characters missing descriptions: {missing}"
        )


def select_random_pair(
    characters: list[str], descriptions: dict[str, str]
) -> tuple[str, str, str, str]:
    """Select two random characters and return (name1, desc1, name2, desc2)."""
    selected = random.sample(characters, 2)
    p1, p2 = selected
    return p1, descriptions[p1], p2, descriptions[p2]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_character_pool.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/character_pool.py tests/test_character_pool.py
git commit -m "add character pool module with loading, validation, and random pairing"
```

---

### Task 5: Output Module (JSONL + Dataset Cards)

**Files:**
- Create: `conversation_dataset_generator/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_output.py
import json
import os
import pytest
from conversation_dataset_generator.output import write_jsonl, build_dataset_card


class TestWriteJsonl:
    def test_writes_correct_structure(self, tmp_path):
        conversations = [
            {
                "turns": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                ],
                "topic": "Greeting",
                "scenario": "Online chat",
                "style": "Casual",
                "include_points": "",
                "persona1_name": "Alice",
                "persona2_name": "Bob",
            }
        ]
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl(conversations, outfile)
        assert count == 2  # 2 turns

        with open(outfile, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

        row0 = json.loads(lines[0])
        assert row0["conversation_id"] == 0
        assert row0["turn_number"] == 0
        assert row0["role"] == "human"
        assert row0["speaker_name"] == "Alice"
        assert row0["content"] == "Hello"

        row1 = json.loads(lines[1])
        assert row1["conversation_id"] == 0
        assert row1["turn_number"] == 1
        assert row1["role"] == "gpt"
        assert row1["speaker_name"] == "Bob"

    def test_multiple_conversations(self, tmp_path):
        conversations = [
            {
                "turns": [{"from": "human", "value": "Hi"}],
                "topic": "T1",
                "scenario": "S1",
                "style": "St1",
                "include_points": "",
                "persona1_name": "A",
                "persona2_name": "B",
            },
            {
                "turns": [{"from": "gpt", "value": "Hey"}],
                "topic": "T2",
                "scenario": "S2",
                "style": "St2",
                "include_points": "",
                "persona1_name": "A",
                "persona2_name": "B",
            },
        ]
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl(conversations, outfile)
        assert count == 2

        with open(outfile, "r") as f:
            lines = f.readlines()
        row0 = json.loads(lines[0])
        row1 = json.loads(lines[1])
        assert row0["conversation_id"] == 0
        assert row1["conversation_id"] == 1

    def test_empty_conversations(self, tmp_path):
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl([], outfile)
        assert count == 0
        assert not os.path.exists(outfile)


class TestBuildDatasetCard:
    def test_manual_mode_card(self):
        card = build_dataset_card(
            mode="manual",
            num_requested=10,
            num_generated=8,
            total_turns=64,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            persona1="Alice",
            persona1_desc="Friendly",
            persona2="Bob",
            persona2_desc="Grumpy",
            topic="Weather",
            scenario="Bus stop",
            style="Casual",
        )
        assert "Alice" in card
        assert "Bob" in card
        assert "Manual" in card
        assert "Qwen/Qwen2.5-7B-Instruct" in card
        assert "---" in card  # YAML frontmatter

    def test_brief_mode_card(self):
        card = build_dataset_card(
            mode="brief",
            num_requested=10,
            num_generated=10,
            total_turns=100,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            persona1="Sherlock",
            persona1_desc="Detective",
            persona2="Watson",
            persona2_desc="Doctor",
            topic="Crime",
            scenario="Baker Street",
            style="Dramatic",
            creative_brief="Sherlock and Watson discuss a case",
        )
        assert "Creative Brief" in card
        assert "Sherlock and Watson discuss a case" in card

    def test_random_pairings_card(self):
        card = build_dataset_card(
            mode="random_pairings",
            num_requested=5,
            num_generated=5,
            total_turns=40,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            topic="Tech",
            scenario="Office",
            style="Professional",
            character_pool=["Alice", "Bob", "Charlie"],
            character_descriptions={
                "Alice": "Friendly",
                "Bob": "Grumpy",
                "Charlie": "Quiet",
            },
        )
        assert "Character Pool" in card
        assert "Alice" in card
        assert "Bob" in card
        assert "Charlie" in card

    def test_card_has_yaml_frontmatter(self):
        card = build_dataset_card(
            mode="manual",
            num_requested=1,
            num_generated=1,
            total_turns=2,
            model_id="test",
            persona1="A",
            persona1_desc="d",
            persona2="B",
            persona2_desc="d",
            topic="T",
            scenario="S",
            style="St",
        )
        assert card.startswith("---\n")
        assert "license:" in card
        assert "tags:" in card
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_output.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement output.py**

```python
# conversation_dataset_generator/output.py
"""JSONL serialization, HF Dataset creation, and dataset card templates."""

import json
import logging

logger = logging.getLogger(__name__)

YAML_FRONTMATTER = """---
license: mit
language:
- en
tags:
- conversational
- synthetic
- sharegpt
---"""


def write_jsonl(conversations: list[dict], output_path: str) -> int:
    """Write conversations to a JSONL file.

    Each conversation dict must have keys:
        turns, topic, scenario, style, include_points, persona1_name, persona2_name

    Returns the total number of turns written.
    """
    if not conversations:
        logger.warning("No conversations to write.")
        return 0

    total_turns = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for conv_id, conv in enumerate(conversations):
            for turn_num, turn in enumerate(conv["turns"]):
                role = turn["from"]
                if role == "human":
                    speaker = conv["persona1_name"]
                elif role == "gpt":
                    speaker = conv["persona2_name"]
                else:
                    speaker = "Unknown"

                row = {
                    "conversation_id": conv_id,
                    "turn_number": turn_num,
                    "role": role,
                    "speaker_name": speaker,
                    "topic": conv["topic"],
                    "scenario": conv["scenario"],
                    "style": conv["style"],
                    "include_points": conv.get("include_points", ""),
                    "content": turn.get("value", ""),
                }
                f.write(json.dumps(row) + "\n")
                total_turns += 1

    logger.info("Wrote %d turns to %s", total_turns, output_path)
    return total_turns


def build_dataset_card(
    mode: str,
    num_requested: int,
    num_generated: int,
    total_turns: int,
    model_id: str,
    persona1: str | None = None,
    persona1_desc: str | None = None,
    persona2: str | None = None,
    persona2_desc: str | None = None,
    topic: str | None = None,
    scenario: str | None = None,
    style: str | None = None,
    include_points: str | None = None,
    creative_brief: str | None = None,
    search_term1: str | None = None,
    search_term2: str | None = None,
    character_pool: list[str] | None = None,
    character_descriptions: dict[str, str] | None = None,
    variation_enabled: bool = False,
    repo_id: str | None = None,
) -> str:
    """Build a Markdown dataset card for HuggingFace Hub."""
    sections = [YAML_FRONTMATTER, ""]

    # Title
    if mode in ("random_pairings", "random_pairings_variation"):
        pool_size = len(character_pool) if character_pool else 0
        sections.append(
            f"# Character Pool Dataset ({pool_size} Characters) — "
            f"Conversation Dataset Generator"
        )
    else:
        sections.append(
            f"# {persona1 or '?'} & {persona2 or '?'}: {topic or '?'} — "
            f"Conversation Dataset Generator"
        )

    sections.append("")
    sections.append(
        "This dataset was generated using the "
        "[Conversation Dataset Generator]"
        "(https://github.com/cahlen/conversation-dataset-generator)."
    )

    # Generation Parameters
    sections.append("")
    sections.append("## Generation Parameters")
    sections.append("")

    # Mode description
    mode_desc = _mode_description(mode, creative_brief, search_term1, search_term2,
                                   persona1, persona2, variation_enabled)
    sections.append(f"* **Generation Mode:** {mode_desc}")
    sections.append(f"* **Conversations Requested:** {num_requested}")
    sections.append(f"* **Conversations Generated:** {num_generated}")
    sections.append(f"* **Total Turns:** {total_turns}")
    sections.append(f"* **Model:** `{model_id}`")

    if topic:
        sections.append(f"* **Topic:** `{topic}`")
    if scenario:
        sections.append(f"* **Scenario:** `{scenario}`")
    if style:
        sections.append(f"* **Style:** `{style}`")
    if include_points:
        sections.append(f"* **Include Points:** `{include_points}`")

    # Personas section (non-pool modes)
    if mode not in ("random_pairings", "random_pairings_variation"):
        if persona1:
            sections.append("")
            sections.append("## Personas")
            sections.append("")
            sections.append(f"**{persona1}** — `{persona1_desc}` → role: `human`")
            sections.append("")
            sections.append(f"**{persona2}** — `{persona2_desc}` → role: `gpt`")

    # Character pool section
    if mode in ("random_pairings", "random_pairings_variation") and character_pool:
        sections.append("")
        sections.append("## Character Pool")
        sections.append("")
        for char in character_pool:
            desc = (character_descriptions or {}).get(char, "No description")
            sections.append(f"**{char}** — `{desc}`")
            sections.append("")

    # Dataset format
    sections.append("")
    sections.append("## Dataset Format")
    sections.append("")
    sections.append("Each row contains:")
    sections.append("- `conversation_id` — unique conversation identifier")
    sections.append("- `turn_number` — sequential turn within the conversation")
    sections.append("- `role` — `human` or `gpt`")
    sections.append("- `speaker_name` — the character's name")
    sections.append("- `topic`, `scenario`, `style`, `include_points` — generation parameters")
    sections.append("- `content` — the dialogue text")

    # Usage
    if repo_id:
        sections.append("")
        sections.append("## Usage")
        sections.append("")
        sections.append("```python")
        sections.append("from datasets import load_dataset")
        sections.append("")
        sections.append(f'dataset = load_dataset("{repo_id}")')
        sections.append("print(dataset['train'][0])")
        sections.append("```")

    return "\n".join(sections)


def _mode_description(
    mode: str,
    creative_brief: str | None,
    search_term1: str | None,
    search_term2: str | None,
    persona1: str | None,
    persona2: str | None,
    variation_enabled: bool,
) -> str:
    """Generate a mode description string for the dataset card."""
    if mode == "brief":
        desc = f"Creative Brief (`--creative-brief`)"
        if creative_brief:
            desc += f"\n* **Original Brief:** `{creative_brief}`"
        if search_term1 or search_term2:
            desc += "\n* **Web Context Sources:**"
            if search_term1:
                desc += f" {persona1 or 'P1'} via `{search_term1}`"
            if search_term2:
                desc += f", {persona2 or 'P2'} via `{search_term2}`"
        return desc
    elif mode == "fixed_persona_variation":
        return "Fixed Persona with Variation (`--enable-variation`)"
    elif mode in ("random_pairings", "random_pairings_variation"):
        desc = "Random Pairings (`--random-pairings`)"
        if variation_enabled:
            desc += " with topic variation"
        return desc
    elif mode == "manual":
        return "Manual (No Variation)"
    return mode


def create_hf_dataset(output_path: str):
    """Load a JSONL file into a HuggingFace DatasetDict.

    Returns a DatasetDict with a 'train' split, or None on failure.
    """
    try:
        from datasets import (
            DatasetDict,
            Features,
            Value,
            load_dataset,
        )

        features = Features(
            {
                "conversation_id": Value("int64"),
                "turn_number": Value("int64"),
                "role": Value("string"),
                "speaker_name": Value("string"),
                "topic": Value("string"),
                "scenario": Value("string"),
                "style": Value("string"),
                "include_points": Value("string"),
                "content": Value("string"),
            }
        )
        ds = load_dataset("json", data_files=output_path, split="train", features=features)
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        return ds
    except Exception as e:
        logger.error("Failed to create HF dataset: %s", e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_output.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/output.py tests/test_output.py
git commit -m "add output module with JSONL writer and dataset card templates"
```

---

### Task 6: Web Search Module

**Files:**
- Create: `conversation_dataset_generator/web_search.py`

No tests for this module — it wraps an external API (DuckDuckGo). We test it indirectly via the generation module with mocks.

- [ ] **Step 1: Implement web_search.py**

```python
# conversation_dataset_generator/web_search.py
"""DuckDuckGo web search for persona context enrichment."""

import logging

logger = logging.getLogger(__name__)


def get_persona_context(persona_name: str, max_results: int = 3) -> str:
    """Search the web for background info on a persona.

    Returns concatenated text snippets, or a fallback string if search fails.
    """
    logger.info("Performing web search for: %s", persona_name)
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            query = f"Who is {persona_name}? Background, personality, notable traits."
            results = list(ddgs.text(query, max_results=max_results))

        if results:
            snippets = [f"- {r['body']}" for r in results if r.get("body")]
            if snippets:
                context = "\n".join(snippets)
                logger.info("Found %d snippets for %s", len(snippets), persona_name)
                return context

        logger.warning("No web search results for %s", persona_name)
    except Exception as e:
        logger.error("Error during web search for %s: %s", persona_name, e)

    return "No relevant web context found."
```

- [ ] **Step 2: Commit**

```bash
git add conversation_dataset_generator/web_search.py
git commit -m "add web search module for persona context enrichment"
```

---

### Task 7: Models Module

**Files:**
- Create: `conversation_dataset_generator/models.py`

No unit tests — this is pure GPU/model interaction. Tested via integration.

- [ ] **Step 1: Implement models.py**

```python
# conversation_dataset_generator/models.py
"""Model and tokenizer loading, pipeline creation."""

import logging
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_model_and_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = False,
) -> tuple:
    """Load tokenizer and model, return (pipeline, tokenizer).

    Args:
        model_id: HuggingFace model identifier.
        load_in_4bit: Whether to use 4-bit NF4 quantization.

    Returns:
        Tuple of (text-generation pipeline, tokenizer).
    """
    start = time.monotonic()

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        logger.info("Loading model with 4-bit quantization (NF4): %s", model_id)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info("Loading model with default precision: %s", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            device_map="auto",
            trust_remote_code=True,
        )

    logger.info("Creating text-generation pipeline...")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    elapsed = time.monotonic() - start
    logger.info(
        "Model loaded in %.2fs (4-bit: %s)", elapsed, load_in_4bit
    )

    return text_generator, tokenizer
```

- [ ] **Step 2: Commit**

```bash
git add conversation_dataset_generator/models.py
git commit -m "add models module for model loading and pipeline creation"
```

---

### Task 8: Generation Module

**Files:**
- Create: `conversation_dataset_generator/generation.py`
- Create: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generation.py
import pytest
from unittest.mock import MagicMock
from conversation_dataset_generator.generation import (
    generate_args_from_brief,
    generate_topic_variation,
    generate_conversation,
    extract_generated_text,
)


def make_mock_pipeline(response_text: str):
    """Create a mock pipeline that returns the given text."""
    mock = MagicMock()
    # Pipeline returns [{"generated_text": prompt + response}]
    def side_effect(prompt_text, **kwargs):
        return [{"generated_text": prompt_text + response_text}]
    mock.side_effect = side_effect
    return mock


def make_mock_tokenizer():
    """Create a mock tokenizer with apply_chat_template."""
    mock = MagicMock()
    mock.eos_token_id = 0
    mock.apply_chat_template.side_effect = lambda msgs, **kw: "PROMPT:"
    mock.encode.side_effect = lambda text: text.split()
    return mock


class TestExtractGeneratedText:
    def test_strips_prompt(self):
        result = extract_generated_text("PROMPT:Hello world", "PROMPT:")
        assert result == "Hello world"

    def test_full_output_when_prompt_missing(self):
        result = extract_generated_text("Hello world", "DIFFERENT_PROMPT:")
        assert result == "Hello world"

    def test_empty_generation(self):
        result = extract_generated_text("PROMPT:", "PROMPT:")
        assert result is None


class TestGenerateArgsFromBrief:
    def test_successful_generation(self):
        response = (
            '--persona1 "Sherlock"\n'
            '--persona1-desc "A brilliant detective"\n'
            '--persona2 "Watson"\n'
            '--persona2-desc "A loyal doctor"\n'
            '--topic "A mysterious case"\n'
            '--scenario "221B Baker Street"\n'
            '--style "Dramatic and suspenseful"'
        )
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()

        result = generate_args_from_brief("Sherlock and Watson", pipeline, tokenizer)
        assert result is not None
        assert result["persona1"] == "Sherlock"
        assert result["persona2"] == "Watson"

    def test_returns_none_on_garbage_output(self):
        pipeline = make_mock_pipeline("This is just random garbage text")
        tokenizer = make_mock_tokenizer()

        result = generate_args_from_brief("Test brief", pipeline, tokenizer)
        assert result is None

    def test_applies_defaults_for_missing_optional_fields(self):
        response = (
            '--persona1 "Sherlock"\n'
            '--persona1-desc "Detective"\n'
            '--persona2 "Watson"\n'
            '--persona2-desc "Doctor"\n'
            '--topic "Crime"\n'
            '--scenario "London"\n'
            '--style "Tense"'
        )
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()

        result = generate_args_from_brief("Test", pipeline, tokenizer)
        assert result is not None
        # include_points is optional, should not be present
        assert "include_points" not in result or result.get("include_points") is not None


class TestGenerateTopicVariation:
    def test_successful_variation(self):
        response = (
            '--topic "A new topic"\n'
            '--scenario "A new scenario"\n'
            '--style "A new style"'
        )
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()

        result = generate_topic_variation(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
            generator_pipeline=pipeline, tokenizer=tokenizer,
        )
        assert result is not None
        assert result["topic"] == "A new topic"
        assert result["scenario"] == "A new scenario"

    def test_returns_none_on_parse_failure(self):
        pipeline = make_mock_pipeline("Just random text, no args")
        tokenizer = make_mock_tokenizer()

        result = generate_topic_variation(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
            generator_pipeline=pipeline, tokenizer=tokenizer,
        )
        assert result is None


class TestGenerateConversation:
    def test_successful_generation(self):
        response = "Alice: Hello\nBob: Hi there\nAlice: How are you?"
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()

        turns = generate_conversation(
            topic="Greeting", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Online", style="Casual",
            generator_pipeline=pipeline, tokenizer=tokenizer,
            max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 3

    def test_returns_none_on_empty_output(self):
        pipeline = make_mock_pipeline("")
        tokenizer = make_mock_tokenizer()

        turns = generate_conversation(
            topic="T", persona1="A", persona2="B",
            persona1_desc="d1", persona2_desc="d2",
            scenario="S", style="St",
            generator_pipeline=pipeline, tokenizer=tokenizer,
            max_new_tokens=512,
        )
        assert turns is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement generation.py**

```python
# conversation_dataset_generator/generation.py
"""LLM call wrappers with retry logic for conversation generation."""

import logging
import time

from conversation_dataset_generator.parsing import (
    parse_arg_generation_output,
    parse_conversation_to_sharegpt,
    parse_variation_output,
)
from conversation_dataset_generator.prompts import (
    build_arg_generation_messages,
    build_conversation_messages,
    build_variation_messages,
)

logger = logging.getLogger(__name__)


def extract_generated_text(full_output: str, prompt_text: str) -> str | None:
    """Extract the generated portion of LLM output by removing the prompt prefix.

    Returns None if the generated text is empty.
    """
    if prompt_text in full_output:
        text = full_output[len(prompt_text):].strip()
    else:
        logger.warning("Prompt not found in output; using full text.")
        text = full_output.strip()

    return text if text else None


def _call_pipeline(
    generator_pipeline,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str | None:
    """Call the text-generation pipeline and extract the generated text.

    Returns the generated text string, or None on failure.
    """
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        logger.error("Failed to apply chat template: %s", e)
        return None

    try:
        outputs = generator_pipeline(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
    except Exception as e:
        logger.error("Pipeline call failed: %s", e)
        return None

    if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
        return extract_generated_text(outputs[0]["generated_text"], prompt_text)

    logger.warning("Unexpected pipeline output format.")
    return None


def generate_args_from_brief(
    brief: str,
    generator_pipeline,
    tokenizer,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Generate detailed conversation args from a creative brief.

    Uses the LLM to brainstorm personas, topic, scenario, etc.
    Retries on parse failure. Returns parsed dict or None.
    """
    # Optionally fetch web context
    persona1_context = None
    persona2_context = None

    if persona1_search_term or persona2_search_term:
        from conversation_dataset_generator.web_search import get_persona_context

        if persona1_search_term:
            persona1_context = get_persona_context(persona1_search_term)
        if persona2_search_term:
            persona2_context = get_persona_context(persona2_search_term)

    messages = build_arg_generation_messages(
        brief=brief,
        persona1_context=persona1_context,
        persona2_context=persona2_context,
        search_term1=persona1_search_term,
        search_term2=persona2_search_term,
    )

    for attempt in range(max_retries):
        logger.info("Arg generation attempt %d/%d", attempt + 1, max_retries)
        start = time.monotonic()
        text = _call_pipeline(
            generator_pipeline, tokenizer, messages,
            max_new_tokens=600, temperature=0.6,
        )
        elapsed = time.monotonic() - start

        if text is None:
            logger.warning("Empty response (attempt %d, %.2fs)", attempt + 1, elapsed)
            continue

        result = parse_arg_generation_output(text)
        if result is not None:
            logger.info("Args generated successfully (%.2fs)", elapsed)
            return result

        logger.warning("Parse failed (attempt %d, %.2fs)", attempt + 1, elapsed)
        if attempt < max_retries - 1:
            delay = 1 * (2 ** attempt)
            time.sleep(delay)

    logger.error("Arg generation failed after %d attempts.", max_retries)
    return None


def generate_args_from_brief_safe(
    brief: str,
    generator_pipeline,
    tokenizer,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
) -> dict | None:
    """Generate args from brief with fallback defaults for missing fields."""
    result = generate_args_from_brief(
        brief, generator_pipeline, tokenizer,
        persona1_search_term, persona2_search_term,
    )
    if result is None:
        return None

    # Persona names are required — no fallback
    if "persona1" not in result or "persona2" not in result:
        logger.error("Missing persona names. Cannot proceed.")
        return None

    # Apply defaults for missing optional descriptions
    defaults = {
        "persona1_desc": f"A character named {result.get('persona1')}. Speaks naturally.",
        "persona2_desc": f"A character named {result.get('persona2')}. Speaks naturally.",
        "topic": "A casual conversation",
        "scenario": "A neutral setting where the two personas meet",
        "style": "Natural, casual conversation",
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default
            logger.warning("Missing '%s', using default: '%s'", key, default)

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
    """Generate a topic/scenario/style variation using the LLM.

    Returns dict with 'topic', 'scenario', and optionally 'style', or None.
    """
    messages = build_variation_messages(
        persona1=persona1, persona1_desc=persona1_desc,
        persona2=persona2, persona2_desc=persona2_desc,
        initial_topic=initial_topic, initial_scenario=initial_scenario,
        initial_style=initial_style, original_brief=original_brief,
    )

    start = time.monotonic()
    text = _call_pipeline(
        generator_pipeline, tokenizer, messages,
        max_new_tokens=256, temperature=0.7,
    )
    elapsed = time.monotonic() - start

    if text is None:
        logger.warning("Empty variation response (%.2fs)", elapsed)
        return None

    result = parse_variation_output(text)
    if result is not None:
        # Use initial style as fallback if not in variation output
        if "style" not in result:
            result["style"] = initial_style
        logger.info("Variation generated (%.2fs)", elapsed)
    else:
        logger.warning("Variation parse failed (%.2fs)", elapsed)

    return result


def generate_conversation(
    topic: str,
    persona1: str,
    persona2: str,
    persona1_desc: str,
    persona2_desc: str,
    scenario: str,
    style: str,
    generator_pipeline,
    tokenizer,
    max_new_tokens: int = 2048,
    include_points: str | None = None,
    role_mapping: dict | None = None,
) -> list[dict] | None:
    """Generate a single conversation and return parsed ShareGPT turns.

    Returns list of turn dicts, or None on failure.
    """
    messages = build_conversation_messages(
        topic=topic, persona1=persona1, persona2=persona2,
        persona1_desc=persona1_desc, persona2_desc=persona2_desc,
        scenario=scenario, style=style, include_points=include_points,
    )

    start = time.monotonic()
    text = _call_pipeline(
        generator_pipeline, tokenizer, messages,
        max_new_tokens=max_new_tokens, temperature=0.75,
    )
    elapsed = time.monotonic() - start

    if text is None:
        logger.warning("Empty conversation response (%.2fs)", elapsed)
        return None

    # Validate output starts with a persona name
    if not (
        text.lstrip().startswith(f"{persona1}:")
        or text.lstrip().startswith(f"{persona2}:")
    ):
        logger.warning(
            "Conversation doesn't start with persona prefix. First 100 chars: %s",
            text[:100],
        )
        return None

    turns, _, _ = parse_conversation_to_sharegpt(
        text, persona1, persona2, role_mapping=role_mapping,
    )

    if turns:
        num_tokens = len(tokenizer.encode(text))
        tps = num_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            "Generated %d turns, %d tokens in %.2fs (%.1f tok/s)",
            len(turns), num_tokens, elapsed, tps,
        )

    return turns
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generation.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/generation.py tests/test_generation.py
git commit -m "add generation module with LLM wrappers, retry logic, and mocked tests"
```

---

### Task 9: Hub Module

**Files:**
- Create: `conversation_dataset_generator/hub.py`

No unit tests — wraps HuggingFace Hub API. Tested manually.

- [ ] **Step 1: Implement hub.py**

```python
# conversation_dataset_generator/hub.py
"""HuggingFace Hub upload functionality."""

import io
import logging
import time

logger = logging.getLogger(__name__)


def upload_to_hub(
    dataset_dict,
    repo_id: str,
    card_content: str | None = None,
    force: bool = False,
) -> bool:
    """Upload a DatasetDict to HuggingFace Hub.

    Args:
        dataset_dict: HuggingFace DatasetDict to upload.
        repo_id: Target repository ID (e.g., "username/dataset-name").
        card_content: Optional markdown content for README.md.
        force: Skip upload confirmation.

    Returns:
        True if upload succeeded, False otherwise.
    """
    from huggingface_hub import HfApi, HfFolder

    token = HfFolder.get_token()
    if not token:
        logger.error(
            "HuggingFace token not found. Run: huggingface-cli login"
        )
        return False

    if not force:
        try:
            confirm = input(
                f"Upload to {repo_id}? (yes/no): "
            )
            if confirm.lower() != "yes":
                logger.info("Upload cancelled.")
                return False
        except EOFError:
            logger.warning("No input available. Upload cancelled.")
            return False

    # Push dataset
    start = time.monotonic()
    try:
        dataset_dict.push_to_hub(repo_id, private=False)
        logger.info("Dataset pushed in %.2fs", time.monotonic() - start)
    except Exception as e:
        logger.error("Failed to push dataset: %s", e)
        return False

    # Upload README
    if card_content:
        try:
            api = HfApi(token=token)
            readme_bytes = card_content.encode("utf-8")
            api.upload_file(
                path_or_fileobj=io.BytesIO(readme_bytes),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            logger.info("README.md uploaded.")
        except Exception as e:
            logger.warning("Failed to upload README.md: %s", e)

    return True
```

- [ ] **Step 2: Commit**

```bash
git add conversation_dataset_generator/hub.py
git commit -m "add hub module for HuggingFace upload"
```

---

### Task 10: CLI Module (Orchestration)

**Files:**
- Create: `conversation_dataset_generator/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for argument parsing and mode detection**

```python
# tests/test_cli.py
import pytest
from conversation_dataset_generator.cli import build_parser, detect_mode


class TestBuildParser:
    def test_default_model(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"

    def test_default_max_tokens(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.max_new_tokens == 2048

    def test_no_delete_repo_flag(self):
        parser = build_parser()
        assert not hasattr(parser.parse_args(["--creative-brief", "test"]), "delete_repo")


class TestDetectMode:
    def test_brief_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        mode = detect_mode(args, parser)
        assert mode == "brief"

    def test_manual_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--topic", "T", "--persona1", "A", "--persona1-desc", "d1",
            "--persona2", "B", "--persona2-desc", "d2",
            "--scenario", "S", "--style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "manual"

    def test_fixed_persona_variation_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--enable-variation",
            "--fixed-persona1", "A", "--fixed-persona1-desc", "d1",
            "--fixed-persona2", "B", "--fixed-persona2-desc", "d2",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "fixed_persona_variation"

    def test_random_pairings_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings",
            "--character-pool", "chars.yaml",
            "--persona-desc-pool", "descs.yaml",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "random_pairings"

    def test_random_pairings_with_variation(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings", "--enable-variation",
            "--character-pool", "chars.yaml",
            "--persona-desc-pool", "descs.yaml",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "random_pairings_variation"

    def test_role_mapping_default(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.role_mapping is None  # None means default p1=human,p2=gpt

    def test_role_mapping_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test",
            "--role-mapping", "p1=gpt,p2=human",
        ])
        assert args.role_mapping == "p1=gpt,p2=human"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement cli.py**

```python
# conversation_dataset_generator/cli.py
"""Command-line interface and orchestration for conversation generation."""

import argparse
import logging
import sys
import time

from tqdm import tqdm

from conversation_dataset_generator.models import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


def parse_role_mapping(mapping_str: str | None) -> dict:
    """Parse role mapping string like 'p1=human,p2=gpt' into a dict."""
    if mapping_str is None:
        return {"p1": "human", "p2": "gpt"}

    result = {}
    for part in mapping_str.split(","):
        key, _, value = part.strip().partition("=")
        if key in ("p1", "p2") and value in ("human", "gpt"):
            result[key] = value

    if "p1" not in result or "p2" not in result:
        raise ValueError(
            f"Invalid role mapping: '{mapping_str}'. "
            "Expected format: 'p1=human,p2=gpt' or 'p1=gpt,p2=human'"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all generation modes."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversational data for LLM fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_ex = mode_group.add_mutually_exclusive_group()
    mode_ex.add_argument(
        "--creative-brief", type=str,
        help="High-level brief for automatic argument generation + topic variation.",
    )
    mode_ex.add_argument(
        "--random-pairings", action="store_true",
        help="Enable random pairing mode using character pools.",
    )

    # Manual mode args
    manual = parser.add_argument_group("Manual Mode")
    manual.add_argument("--topic", type=str)
    manual.add_argument("--persona1", type=str)
    manual.add_argument("--persona1-desc", type=str)
    manual.add_argument("--persona2", type=str)
    manual.add_argument("--persona2-desc", type=str)
    manual.add_argument("--scenario", type=str)
    manual.add_argument("--style", type=str)
    manual.add_argument("--include-points", type=str, default=None)

    # Fixed persona variation args
    fixed = parser.add_argument_group("Fixed Persona Variation Mode")
    fixed.add_argument("--fixed-persona1", type=str)
    fixed.add_argument("--fixed-persona1-desc", type=str)
    fixed.add_argument("--fixed-persona2", type=str)
    fixed.add_argument("--fixed-persona2-desc", type=str)
    fixed.add_argument("--initial-topic", type=str)
    fixed.add_argument("--initial-scenario", type=str)
    fixed.add_argument("--initial-style", type=str)
    fixed.add_argument("--enable-variation", action="store_true")

    # Random pairings args
    pool = parser.add_argument_group("Random Pairings Mode")
    pool.add_argument("--character-pool", type=str)
    pool.add_argument("--persona-desc-pool", type=str)

    # Brief context args
    brief_ctx = parser.add_argument_group("Creative Brief Web Context")
    brief_ctx.add_argument("--persona1-search-term", type=str, default=None)
    brief_ctx.add_argument("--persona2-search-term", type=str, default=None)

    # General args
    general = parser.add_argument_group("General")
    general.add_argument("--num-examples", type=int, default=3)
    general.add_argument("--output-file", type=str, default="generated_data.jsonl")
    general.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    general.add_argument("--max-new-tokens", type=int, default=2048)
    general.add_argument("--upload-to-hub", type=str, default=None, metavar="REPO_ID")
    general.add_argument("--load-in-4bit", action="store_true")
    general.add_argument("--force-upload", action="store_true")
    general.add_argument("--role-mapping", type=str, default=None,
                         help="Role mapping: 'p1=human,p2=gpt' or 'p1=gpt,p2=human'")

    return parser


def detect_mode(args, parser) -> str:
    """Determine the generation mode from parsed arguments."""
    if args.creative_brief:
        return "brief"

    if args.enable_variation and args.random_pairings:
        _require(args, ["character_pool", "persona_desc_pool",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "random_pairings_variation"

    if args.enable_variation:
        _require(args, ["fixed_persona1", "fixed_persona1_desc",
                        "fixed_persona2", "fixed_persona2_desc",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "fixed_persona_variation"

    if args.random_pairings:
        _require(args, ["character_pool", "persona_desc_pool",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "random_pairings"

    if args.persona1 and args.topic:
        _require(args, ["persona1", "persona1_desc", "persona2", "persona2_desc",
                        "topic", "scenario", "style"], parser)
        return "manual"

    parser.error(
        "Insufficient arguments. Provide --creative-brief, OR manual args, "
        "OR fixed persona args with --enable-variation, "
        "OR --random-pairings with pool files."
    )


def _require(args, keys: list[str], parser) -> None:
    """Check that all required keys are present in args."""
    missing = [f"--{k.replace('_', '-')}" for k in keys if getattr(args, k) is None]
    if missing:
        parser.error(f"Missing required arguments: {' '.join(missing)}")


def main():
    """Main entry point for the conversation dataset generator."""
    script_start = time.monotonic()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Reduce library noise
    for lib in ("huggingface_hub", "datasets", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    parser = build_parser()
    args = parser.parse_args()

    mode = detect_mode(args, parser)
    logger.info("Mode: %s", mode)

    role_mapping = parse_role_mapping(args.role_mapping)

    # --- Load model ---
    from conversation_dataset_generator.models import load_model_and_pipeline

    text_generator, tokenizer = load_model_and_pipeline(
        model_id=args.model_id, load_in_4bit=args.load_in_4bit,
    )

    # --- Determine base arguments per mode ---
    from conversation_dataset_generator.generation import (
        generate_args_from_brief_safe,
        generate_conversation,
        generate_topic_variation,
    )
    from conversation_dataset_generator.output import (
        build_dataset_card,
        create_hf_dataset,
        write_jsonl,
    )

    persona1 = None
    persona1_desc = None
    persona2 = None
    persona2_desc = None
    initial_topic = None
    initial_scenario = None
    initial_style = None
    initial_include_points = None
    variation_enabled = False
    character_pool = None
    character_descriptions = None

    if mode == "brief":
        variation_enabled = True
        generated = generate_args_from_brief_safe(
            args.creative_brief, text_generator, tokenizer,
            args.persona1_search_term, args.persona2_search_term,
        )
        if generated is None:
            logger.error("Failed to generate args from brief. Exiting.")
            sys.exit(1)

        persona1 = generated["persona1"]
        persona1_desc = generated["persona1_desc"]
        persona2 = generated["persona2"]
        persona2_desc = generated["persona2_desc"]
        initial_topic = generated["topic"]
        initial_scenario = generated["scenario"]
        initial_style = generated["style"]
        initial_include_points = generated.get("include_points")

    elif mode == "fixed_persona_variation":
        variation_enabled = True
        persona1 = args.fixed_persona1
        persona1_desc = args.fixed_persona1_desc
        persona2 = args.fixed_persona2
        persona2_desc = args.fixed_persona2_desc
        initial_topic = args.initial_topic
        initial_scenario = args.initial_scenario
        initial_style = args.initial_style
        initial_include_points = args.include_points

    elif mode in ("random_pairings", "random_pairings_variation"):
        from conversation_dataset_generator.character_pool import (
            load_character_pool,
            load_description_pool,
            select_random_pair,
            validate_pools,
        )
        import os

        char_path = args.character_pool
        desc_path = args.persona_desc_pool

        # Prepend character-config/ if not an absolute or explicit relative path
        if not os.path.isabs(char_path) and not char_path.startswith("character-config/"):
            char_path = os.path.join("character-config", char_path)
        if not os.path.isabs(desc_path) and not desc_path.startswith("character-config/"):
            desc_path = os.path.join("character-config", desc_path)

        character_pool = load_character_pool(char_path)
        character_descriptions = load_description_pool(desc_path)
        validate_pools(character_pool, character_descriptions)

        variation_enabled = mode == "random_pairings_variation"
        initial_topic = args.initial_topic
        initial_scenario = args.initial_scenario
        initial_style = args.initial_style
        initial_include_points = args.include_points

    elif mode == "manual":
        variation_enabled = False
        persona1 = args.persona1
        persona1_desc = args.persona1_desc
        persona2 = args.persona2
        persona2_desc = args.persona2_desc
        initial_topic = args.topic
        initial_scenario = args.scenario
        initial_style = args.style
        initial_include_points = args.include_points

    # --- Generation loop ---
    conversations = []
    total_llm_time = 0.0

    for i in tqdm(range(args.num_examples), desc="Generating", unit="example"):
        # Select random pair if needed
        if mode in ("random_pairings", "random_pairings_variation"):
            persona1, persona1_desc, persona2, persona2_desc = select_random_pair(
                character_pool, character_descriptions,
            )
            logger.info("Pair %d: %s & %s", i + 1, persona1, persona2)

        # Topic variation
        current_topic = initial_topic
        current_scenario = initial_scenario
        current_style = initial_style
        current_include_points = initial_include_points

        if variation_enabled:
            variation = generate_topic_variation(
                persona1=persona1, persona1_desc=persona1_desc,
                persona2=persona2, persona2_desc=persona2_desc,
                initial_topic=initial_topic, initial_scenario=initial_scenario,
                initial_style=initial_style,
                generator_pipeline=text_generator, tokenizer=tokenizer,
                original_brief=args.creative_brief if mode == "brief" else None,
            )
            if variation:
                current_topic = variation.get("topic", initial_topic)
                current_scenario = variation.get("scenario", initial_scenario)
                current_style = variation.get("style", initial_style)

        # Generate conversation
        start = time.monotonic()
        turns = generate_conversation(
            topic=current_topic, persona1=persona1, persona2=persona2,
            persona1_desc=persona1_desc, persona2_desc=persona2_desc,
            scenario=current_scenario, style=current_style,
            generator_pipeline=text_generator, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            include_points=current_include_points,
            role_mapping=role_mapping,
        )
        total_llm_time += time.monotonic() - start

        if turns:
            conversations.append({
                "turns": turns,
                "topic": current_topic,
                "scenario": current_scenario,
                "style": current_style,
                "include_points": current_include_points or "",
                "persona1_name": persona1,
                "persona2_name": persona2,
            })

    # --- Write output ---
    num_generated = len(conversations)
    logger.info("Generated %d/%d conversations", num_generated, args.num_examples)

    if not conversations:
        logger.error("No conversations generated. Exiting.")
        sys.exit(1)

    total_turns = write_jsonl(conversations, args.output_file)

    # --- Optional HF upload ---
    if args.upload_to_hub:
        ds = create_hf_dataset(args.output_file)
        if ds is not None:
            card = build_dataset_card(
                mode=mode,
                num_requested=args.num_examples,
                num_generated=num_generated,
                total_turns=total_turns,
                model_id=args.model_id,
                persona1=persona1, persona1_desc=persona1_desc,
                persona2=persona2, persona2_desc=persona2_desc,
                topic=current_topic if conversations else initial_topic,
                scenario=current_scenario if conversations else initial_scenario,
                style=current_style if conversations else initial_style,
                include_points=current_include_points,
                creative_brief=args.creative_brief if mode == "brief" else None,
                search_term1=args.persona1_search_term,
                search_term2=args.persona2_search_term,
                character_pool=character_pool,
                character_descriptions=character_descriptions,
                variation_enabled=variation_enabled,
                repo_id=args.upload_to_hub,
            )

            from conversation_dataset_generator.hub import upload_to_hub
            upload_to_hub(ds, args.upload_to_hub, card, force=args.force_upload)

    elapsed = time.monotonic() - script_start
    logger.info(
        "Done. %d conversations, %d turns, %.2fs total (%.2fs LLM time)",
        num_generated, total_turns, elapsed, total_llm_time,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/cli.py tests/test_cli.py
git commit -m "add CLI module with argument parsing, mode detection, and orchestration"
```

---

### Task 11: Wire Up Entry Point & Verify Backward Compat

**Files:**
- Modify: `generate.py`

- [ ] **Step 1: Replace generate.py with thin entry point**

Back up the old file first:

```bash
cp generate.py generate_old.py
```

Then replace `generate.py`:

```python
#!/usr/bin/env python
# coding=utf-8
"""Conversation Dataset Generator — entry point."""

from conversation_dataset_generator.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (42+ tests across 6 test files)

- [ ] **Step 3: Verify CLI help works**

Run: `python generate.py --help`
Expected: Shows help text with all argument groups, default model is `Qwen/Qwen2.5-7B-Instruct`, default max tokens is 2048, no `--delete-repo` flag

- [ ] **Step 4: Verify batch_generate.py still parses correctly**

Run: `python batch_generate.py --help`
Expected: Shows batch config help

- [ ] **Step 5: Commit**

```bash
git add generate.py
git commit -m "replace generate.py with thin entry point into package"
```

---

### Task 12: Integration Smoke Test

This is a manual verification task with actual model inference. Run only if a GPU is available.

- [ ] **Step 1: Test manual mode with a small example**

Run:
```bash
python generate.py \
  --topic "best pizza toppings" \
  --persona1 "Tony" --persona1-desc "A passionate Italian chef" \
  --persona2 "Dave" --persona2-desc "A stubborn pineapple-on-pizza enthusiast" \
  --scenario "kitchen argument" --style "heated but friendly debate" \
  --num-examples 2 --output-file test_manual.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```
Expected: Generates 2 conversations, writes `test_manual.jsonl`

- [ ] **Step 2: Verify JSONL output**

Run: `head -5 test_manual.jsonl | python -m json.tool`
Expected: Valid JSON with correct schema fields

- [ ] **Step 3: Test creative brief mode with variation**

Run:
```bash
python generate.py \
  --creative-brief "Sherlock Holmes and Dr Watson debate whether AI will replace detectives" \
  --num-examples 3 --output-file test_brief.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```
Expected: Generates 3 conversations with VARIED topics/scenarios (check the JSONL to confirm topics differ)

- [ ] **Step 4: Verify topic variation is actually working**

Run: `python -c "import json; lines = open('test_brief.jsonl').readlines(); topics = set(json.loads(l)['topic'] for l in lines); print(f'Unique topics: {len(topics)}'); [print(t) for t in topics]"`
Expected: More than 1 unique topic (variation is working!)

- [ ] **Step 5: Clean up test files**

```bash
rm -f test_manual.jsonl test_brief.jsonl
```

- [ ] **Step 6: Remove old generate.py backup**

```bash
rm -f generate_old.py
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "verify integration: all modes working, variation confirmed functional"
```

---

### Task 13: Final Cleanup

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Update CLAUDE.md**

Update the CLAUDE.md to reflect the new package structure, test commands, and architecture. Key changes:
- Package is now `conversation_dataset_generator/` with 9 modules
- Run tests: `pytest tests/ -v`
- Run single test: `pytest tests/test_parsing.py::TestParseVariationOutput -v`
- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Default max tokens: 2048
- Removed: `--delete-repo`, image search, pandas/peft/trl deps

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "update CLAUDE.md for new package structure and defaults"
```
