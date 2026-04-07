"""Output module for writing JSONL files and building Hugging Face dataset cards."""

import json
import os
from typing import Optional

YAML_FRONTMATTER = """---
license: mit
language:
- en
tags:
- conversational
- synthetic
- sharegpt
---"""


def write_jsonl(conversations: list, output_path: str) -> int:
    """Write conversations to a JSONL file.

    Each conversation dict must have: turns, topic, scenario, style,
    include_points, persona1_name, persona2_name.
    Each turn has "from" (human/gpt) and "value".

    Returns the total number of turns written.
    """
    if not conversations:
        return 0

    total_turns = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for conv_id, conv in enumerate(conversations):
            turns = conv.get("turns", [])
            topic = conv.get("topic", "")
            scenario = conv.get("scenario", "")
            style = conv.get("style", "")
            include_points = conv.get("include_points", "")
            persona1_name = conv.get("persona1_name", "")
            persona2_name = conv.get("persona2_name", "")

            for turn_num, turn in enumerate(turns):
                role = turn.get("from", "")
                content = turn.get("value", "")

                # Map role to speaker name: human -> persona1, gpt -> persona2
                if role == "human":
                    speaker_name = persona1_name
                else:
                    speaker_name = persona2_name

                row = {
                    "conversation_id": conv_id,
                    "turn_number": turn_num,
                    "role": role,
                    "speaker_name": speaker_name,
                    "topic": topic,
                    "scenario": scenario,
                    "style": style,
                    "include_points": include_points,
                    "content": content,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_turns += 1

    return total_turns


def _mode_description(
    mode: str,
    persona1: Optional[str] = None,
    persona1_desc: Optional[str] = None,
    persona2: Optional[str] = None,
    persona2_desc: Optional[str] = None,
    creative_brief: Optional[str] = None,
    character_pool: Optional[list] = None,
    character_descriptions: Optional[dict] = None,
) -> str:
    """Generate mode-specific section text for the dataset card."""
    if mode == "manual":
        lines = [
            "## Generation Mode: Manual",
            "",
            "Conversations were generated using manually specified personas.",
            "",
            "### Personas",
            f"- **{persona1}**: {persona1_desc}",
            f"- **{persona2}**: {persona2_desc}",
        ]
        return "\n".join(lines)
    elif mode == "brief":
        lines = [
            "## Generation Mode: Brief",
            "",
            "Conversations were generated from a creative brief.",
            "",
            "### Personas",
            f"- **{persona1}**: {persona1_desc}",
            f"- **{persona2}**: {persona2_desc}",
            "",
            "### Creative Brief",
            creative_brief or "",
        ]
        return "\n".join(lines)
    elif mode == "random_pairings":
        lines = [
            "## Generation Mode: Random Pairings",
            "",
            "Conversations were generated using random character pairings from a pool.",
            "",
            "### Character Pool",
        ]
        if character_pool and character_descriptions:
            for name in character_pool:
                desc = character_descriptions.get(name, "")
                lines.append(f"- **{name}**: {desc}")
        elif character_pool:
            for name in character_pool:
                lines.append(f"- {name}")
        return "\n".join(lines)
    else:
        return f"## Generation Mode: {mode}"


def build_dataset_card(
    mode: str,
    num_requested: int,
    num_generated: int,
    total_turns: int,
    model_id: str,
    topic: str = "",
    scenario: str = "",
    style: str = "",
    persona1: Optional[str] = None,
    persona1_desc: Optional[str] = None,
    persona2: Optional[str] = None,
    persona2_desc: Optional[str] = None,
    creative_brief: Optional[str] = None,
    character_pool: Optional[list] = None,
    character_descriptions: Optional[dict] = None,
    repo_id: Optional[str] = None,
) -> str:
    """Build a Hugging Face dataset card (README.md content).

    Returns the full card text including YAML frontmatter.
    """
    mode_label = {
        "manual": "Manual",
        "brief": "Brief",
        "random_pairings": "Random Pairings",
    }.get(mode, mode.capitalize())

    mode_section = _mode_description(
        mode=mode,
        persona1=persona1,
        persona1_desc=persona1_desc,
        persona2=persona2,
        persona2_desc=persona2_desc,
        creative_brief=creative_brief,
        character_pool=character_pool,
        character_descriptions=character_descriptions,
    )

    usage_section = ""
    if repo_id:
        usage_section = f"""
## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```
"""

    card = f"""{YAML_FRONTMATTER}

# Synthetic Conversation Dataset

A synthetically generated conversational dataset in ShareGPT format.

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| Mode | {mode_label} |
| Model | {model_id} |
| Requested Conversations | {num_requested} |
| Generated Conversations | {num_generated} |
| Total Turns | {total_turns} |
| Topic | {topic} |
| Scenario | {scenario} |
| Style | {style} |

{mode_section}

## Dataset Format

Each row in the JSONL file contains the following fields:

- `conversation_id`: Integer index of the conversation
- `turn_number`: Integer index of the turn within the conversation
- `role`: Speaker role (`human` or `gpt`)
- `speaker_name`: Name of the speaker
- `topic`: Conversation topic
- `scenario`: Conversation scenario/setting
- `style`: Conversation style
- `include_points`: Optional points to include in the conversation
- `content`: The spoken text for this turn
{usage_section}"""

    return card


def create_hf_dataset(output_path: str):
    """Load a JSONL file as a Hugging Face DatasetDict.

    Lazy-imports the datasets library. Returns None if not available or file missing.
    """
    try:
        from datasets import load_dataset, DatasetDict
    except ImportError:
        return None

    if not os.path.exists(output_path):
        return None

    try:
        ds = load_dataset("json", data_files={"train": output_path})
        return ds
    except Exception:
        return None
