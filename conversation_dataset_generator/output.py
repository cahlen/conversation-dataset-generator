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
    include_points. For legacy two-speaker support, also include
    persona1_name and persona2_name. For N-speaker support, each turn
    may carry its own "speaker_name" key.

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
            # Legacy fallback
            persona1_name = conv.get("persona1_name", "")
            persona2_name = conv.get("persona2_name", "")

            for turn_num, turn in enumerate(turns):
                role = turn.get("from", "")
                content = turn.get("value", "")
                speaker_name = turn.get("speaker_name", "")

                # Fallback to legacy persona mapping if no speaker_name on turn
                if not speaker_name:
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


def load_conversation_from_jsonl(path: str, conversation_id: int | None = None) -> dict | None:
    """Load a conversation from a JSONL file.

    If conversation_id is None, loads the last conversation.
    Returns dict with: conversation_id, turns, topic, scenario, style, include_points, speaker_names.
    Returns None if not found.
    """
    rows_by_conv = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row["conversation_id"]
            if cid not in rows_by_conv:
                rows_by_conv[cid] = []
            rows_by_conv[cid].append(row)

    if not rows_by_conv:
        return None

    if conversation_id is not None:
        if conversation_id not in rows_by_conv:
            return None
        target_id = conversation_id
    else:
        target_id = max(rows_by_conv.keys())

    rows = rows_by_conv[target_id]
    rows.sort(key=lambda r: r["turn_number"])
    first = rows[0]
    speaker_names = list(dict.fromkeys(r["speaker_name"] for r in rows))

    turns = [{"from": r["role"], "value": r["content"], "speaker_name": r["speaker_name"]} for r in rows]

    return {
        "conversation_id": target_id,
        "turns": turns,
        "topic": first.get("topic", ""),
        "scenario": first.get("scenario", ""),
        "style": first.get("style", ""),
        "include_points": first.get("include_points", ""),
        "speaker_names": speaker_names,
    }


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
