# Multi-Speaker + Continue Conversations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add N-persona conversations, conversation continuation from JSONL, train-speaker role mapping, and fix stale example YAMLs.

**Architecture:** Modify 6 existing modules (parsing, prompts, generation, output, character_pool, cli) and their tests. New `--persona` repeatable flag, `--personas` YAML file, `--train-speaker`, `--continue-from`, `--conversation-id`, `--group-size` flags. The core data type changes from two persona strings to a list of `(name, description)` tuples throughout the pipeline. Backward compat via internal conversion of legacy `--persona1`/`--persona2` flags.

**Tech Stack:** Python 3.10+, pytest, pyyaml

---

## File Map

### Files to modify:
- `conversation_dataset_generator/parsing.py` — N-speaker parser
- `conversation_dataset_generator/prompts.py` — N-speaker prompts + continuation prompt
- `conversation_dataset_generator/generation.py` — N-speaker generation + continuation
- `conversation_dataset_generator/output.py` — N-speaker JSONL + load conversation
- `conversation_dataset_generator/character_pool.py` — select_random_group
- `conversation_dataset_generator/cli.py` — new flags, role mapping, continue mode
- `tests/test_parsing.py` — N-speaker tests
- `tests/test_prompts.py` — N-speaker + continuation tests
- `tests/test_generation.py` — N-speaker + continuation tests
- `tests/test_output.py` — N-speaker JSONL + load conversation tests
- `tests/test_character_pool.py` — select_random_group tests
- `tests/test_cli.py` — new flag tests
- `examples/batch_rockstars_celebs.yaml` — remove model_id
- `examples/batch_mixed_modes.yaml` — remove model_id
- `examples/ai_course_curriculum.yaml` — remove model_id
- `README.md` — document new features

---

### Task 1: Fix Example YAMLs (Quick Win)

**Files:**
- Modify: `examples/batch_rockstars_celebs.yaml`
- Modify: `examples/batch_mixed_modes.yaml`
- Modify: `examples/ai_course_curriculum.yaml`

- [ ] **Step 1: Remove all model_id lines from example YAMLs**

Run:
```bash
sed -i '/model_id:/d' examples/batch_rockstars_celebs.yaml examples/batch_mixed_modes.yaml examples/ai_course_curriculum.yaml
```

- [ ] **Step 2: Verify no model_id references remain**

Run: `grep -r "model_id" examples/`
Expected: No output

- [ ] **Step 3: Verify YAML is still valid**

Run: `python -c "import yaml; [yaml.safe_load(open(f)) for f in ['examples/batch_rockstars_celebs.yaml','examples/batch_mixed_modes.yaml','examples/ai_course_curriculum.yaml']]; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add examples/
git commit -m "remove hardcoded model_id from example YAMLs — use default"
```

---

### Task 2: Update Parsing for N Speakers

**Files:**
- Modify: `conversation_dataset_generator/parsing.py`
- Modify: `tests/test_parsing.py`

- [ ] **Step 1: Write failing tests for N-speaker parsing**

Add to `tests/test_parsing.py`:

```python
class TestParseConversationMultiSpeaker:
    def test_three_speakers(self):
        text = "Alice: Hello\nBob: Hi\nCharlie: Hey everyone"
        turns, names = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
            role_mapping={"Alice": "human", "Bob": "gpt", "Charlie": "gpt"},
        )
        assert len(turns) == 3
        assert turns[0] == {"from": "human", "value": "Hello"}
        assert turns[1] == {"from": "gpt", "value": "Hi"}
        assert turns[2] == {"from": "gpt", "value": "Hey everyone"}
        assert names == ["Alice", "Bob", "Charlie"]

    def test_four_speakers(self):
        text = "A: one\nB: two\nC: three\nD: four"
        turns, names = parse_conversation_to_sharegpt(
            text, personas=["A", "B", "C", "D"],
            role_mapping={"A": "human", "B": "gpt", "C": "gpt", "D": "gpt"},
        )
        assert len(turns) == 4

    def test_default_role_mapping_multi(self):
        """First speaker human, rest gpt."""
        text = "Alice: Hello\nBob: Hi\nCharlie: Hey"
        turns, _ = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
        )
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "gpt"

    def test_legacy_two_args_still_work(self):
        """Backward compat: persona1/persona2 positional."""
        text = "Alice: Hello\nBob: Hi"
        turns, names = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert names == ["Alice", "Bob"]

    def test_train_speaker_mapping(self):
        """train_speaker makes one speaker gpt, rest human."""
        text = "Alice: Hi\nBob: Hello\nCharlie: Hey"
        mapping = {"Alice": "human", "Bob": "gpt", "Charlie": "human"}
        turns, _ = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
            role_mapping=mapping,
        )
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "human"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parsing.py::TestParseConversationMultiSpeaker -v`
Expected: FAIL — TypeError (wrong arguments)

- [ ] **Step 3: Rewrite parse_conversation_to_sharegpt for N speakers**

Replace the function in `conversation_dataset_generator/parsing.py`:

```python
def parse_conversation_to_sharegpt(
    conversation_text: str,
    persona1: str | None = None,
    persona2: str | None = None,
    role_mapping: dict | None = None,
    *,
    personas: list[str] | None = None,
) -> tuple[list[dict] | None, list[str] | None]:
    """Parse raw conversation text into ShareGPT turn structure.

    Supports both legacy 2-speaker and N-speaker modes.

    Legacy call: parse_conversation_to_sharegpt(text, "Alice", "Bob")
    N-speaker call: parse_conversation_to_sharegpt(text, personas=["Alice", "Bob", "Charlie"])

    Args:
        conversation_text: Raw text output from the LLM.
        persona1: Legacy — name of first speaker.
        persona2: Legacy — name of second speaker.
        role_mapping: Dict mapping speaker names to roles ("human"/"gpt").
            Legacy format {"p1": "human", "p2": "gpt"} also accepted.
            Default: first speaker → "human", rest → "gpt".
        personas: List of speaker names (N-speaker mode).

    Returns:
        Tuple of (turns_list, persona_names_list) or (None, None).
    """
    if not conversation_text or not conversation_text.strip():
        return None, None

    # Normalize to personas list
    if personas is None:
        if persona1 and persona2:
            personas = [persona1, persona2]
        else:
            return None, None

    # Normalize role_mapping to {name: role} format
    if role_mapping is None:
        # Default: first speaker is human, rest are gpt
        role_mapping = {}
        for i, name in enumerate(personas):
            role_mapping[name.lower()] = "human" if i == 0 else "gpt"
    elif "p1" in role_mapping or "p2" in role_mapping:
        # Legacy format conversion
        legacy = role_mapping
        role_mapping = {}
        for i, name in enumerate(personas):
            key = f"p{i + 1}"
            role_mapping[name.lower()] = legacy.get(key, "gpt")
    else:
        # Already in {name: role} format, normalize keys to lowercase
        role_mapping = {k.lower(): v for k, v in role_mapping.items()}

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

            role = role_mapping.get(speaker.lower(), "gpt")
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
```

- [ ] **Step 4: Update existing 2-speaker tests for new return type**

The return type changed from `(turns, persona1, persona2)` to `(turns, personas_list)`. Update all existing `TestParseConversationToSharegpt` tests. The key change is that the third return value is gone — now it returns a list.

Example changes:
```python
# Old:
turns, p1, p2 = parse_conversation_to_sharegpt(text, "Alice", "Bob")
assert p1 == "Alice"
assert p2 == "Bob"

# New:
turns, names = parse_conversation_to_sharegpt(text, "Alice", "Bob")
assert names == ["Alice", "Bob"]
```

Update the `test_custom_role_mapping` test to use the new `{name: role}` format OR keep the legacy `{p1: role}` format (both should work):
```python
def test_custom_role_mapping(self):
    text = "Alice: Hello\nBob: Hi"
    turns, _ = parse_conversation_to_sharegpt(
        text, "Alice", "Bob", role_mapping={"p1": "gpt", "p2": "human"}
    )
    assert turns[0]["from"] == "gpt"
    assert turns[1]["from"] == "human"

def test_custom_role_mapping_by_name(self):
    text = "Alice: Hello\nBob: Hi"
    turns, _ = parse_conversation_to_sharegpt(
        text, "Alice", "Bob", role_mapping={"Alice": "gpt", "Bob": "human"}
    )
    assert turns[0]["from"] == "gpt"
    assert turns[1]["from"] == "human"
```

- [ ] **Step 5: Run all parsing tests**

Run: `pytest tests/test_parsing.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 6: Commit**

```bash
git add conversation_dataset_generator/parsing.py tests/test_parsing.py
git commit -m "update parser for N-speaker support with backward compat"
```

---

### Task 3: Update Prompts for N Speakers + Continuation

**Files:**
- Modify: `conversation_dataset_generator/prompts.py`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_prompts.py`:

```python
class TestBuildConversationMessagesMulti:
    def test_three_speakers_in_system(self):
        msgs = build_conversation_messages(
            topic="Party planning",
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy"), ("Charlie", "Quiet")],
            scenario="Office", style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]
        assert "Charlie" in msgs[0]["content"]

    def test_legacy_two_speaker_still_works(self):
        msgs = build_conversation_messages(
            topic="Weather",
            persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Bus stop", style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]


class TestBuildContinuationMessages:
    def test_returns_two_messages(self):
        from conversation_dataset_generator.prompts import build_continuation_messages
        prior_turns = [
            {"from": "human", "value": "Hello", "speaker_name": "Alice"},
            {"from": "gpt", "value": "Hi there", "speaker_name": "Bob"},
        ]
        msgs = build_continuation_messages(
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy")],
            prior_turns=prior_turns,
            topic="Greeting", scenario="Online", style="Casual",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_prior_turns_in_prompt(self):
        from conversation_dataset_generator.prompts import build_continuation_messages
        prior_turns = [
            {"from": "human", "value": "What about quantum?", "speaker_name": "Alice"},
        ]
        msgs = build_continuation_messages(
            personas=[("Alice", "Scientist"), ("Bob", "Student")],
            prior_turns=prior_turns,
            topic="Physics", scenario="Lab", style="Educational",
        )
        assert "What about quantum?" in msgs[1]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py::TestBuildConversationMessagesMulti tests/test_prompts.py::TestBuildContinuationMessages -v`
Expected: FAIL

- [ ] **Step 3: Update build_conversation_messages for N speakers**

Replace in `conversation_dataset_generator/prompts.py`:

```python
def build_conversation_messages(
    topic: str,
    persona1: str | None = None,
    persona2: str | None = None,
    persona1_desc: str | None = None,
    persona2_desc: str | None = None,
    scenario: str = "",
    style: str = "",
    include_points: str | None = None,
    *,
    personas: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build system + user messages for generating a conversation.

    Supports both legacy 2-speaker and N-speaker modes.

    Legacy: build_conversation_messages(topic, persona1="A", persona2="B", ...)
    N-speaker: build_conversation_messages(topic, personas=[("A","desc"), ("B","desc"), ...])
    """
    # Normalize to personas list
    if personas is None:
        if persona1 and persona2:
            personas = [(persona1, persona1_desc or ""), (persona2, persona2_desc or "")]
        else:
            personas = []

    # Build character list
    char_lines = "\n".join(f"- {name}: {desc}" for name, desc in personas)
    name_list = [name for name, _ in personas]
    format_lines = "\n".join(f"{name}: <dialogue>" for name in name_list)

    system_content = (
        f"You are writing a realistic dialogue between {len(personas)} characters.\n\n"
        f"Characters:\n{char_lines}\n\n"
        f"Topic: {topic}\n"
        f"Scenario: {scenario}\n"
        f"Style: {style}\n\n"
        f"Format each turn as:\n{format_lines}\n\n"
        f"Write a natural, engaging conversation that reflects each character's "
        f"personality. Avoid repetition and keep it flowing naturally. "
        f"All characters should participate."
    )

    user_content = "Generate a conversation between the characters."
    if include_points:
        user_content += f"\n\nMake sure to include the following points: {include_points}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
```

- [ ] **Step 4: Add build_continuation_messages**

Add to `conversation_dataset_generator/prompts.py`:

```python
def build_continuation_messages(
    personas: list[tuple[str, str]],
    prior_turns: list[dict],
    topic: str,
    scenario: str,
    style: str,
) -> list[dict]:
    """Build messages for continuing an existing conversation.

    Args:
        personas: List of (name, description) tuples.
        prior_turns: Prior conversation turns with 'speaker_name' and 'value' keys.
        topic: Original conversation topic.
        scenario: Original scenario.
        style: Original style.

    Returns:
        List with system and user messages.
    """
    char_lines = "\n".join(f"- {name}: {desc}" for name, desc in personas)

    # Format prior conversation
    history_lines = []
    for turn in prior_turns:
        speaker = turn.get("speaker_name", "Unknown")
        value = turn.get("value", "")
        history_lines.append(f"{speaker}: {value}")
    history_text = "\n".join(history_lines)

    system_content = (
        f"You are continuing an existing conversation between these characters:\n\n"
        f"Characters:\n{char_lines}\n\n"
        f"Topic: {topic}\n"
        f"Scenario: {scenario}\n"
        f"Style: {style}\n\n"
        f"Maintain each character's established voice and continue naturally."
    )

    user_content = (
        f"Here is the conversation so far:\n\n{history_text}\n\n"
        f"Continue this conversation naturally. Pick up where it left off."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
```

- [ ] **Step 5: Run all prompt tests**

Run: `pytest tests/test_prompts.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add conversation_dataset_generator/prompts.py tests/test_prompts.py
git commit -m "update prompts for N-speaker and add continuation prompt"
```

---

### Task 4: Update Output for N Speakers + Load Conversation

**Files:**
- Modify: `conversation_dataset_generator/output.py`
- Modify: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_output.py`:

```python
class TestWriteJsonlMultiSpeaker:
    def test_three_speakers(self, tmp_path):
        conversations = [
            {
                "turns": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi"},
                    {"from": "gpt", "value": "Hey"},
                ],
                "topic": "Greet",
                "scenario": "Room",
                "style": "Casual",
                "include_points": "",
                "personas": [("Alice", "human"), ("Bob", "gpt"), ("Charlie", "gpt")],
                "speaker_names": {"human_0": "Alice", "gpt_0": "Bob", "gpt_1": "Charlie"},
            }
        ]
        # Won't work yet — need to update write_jsonl


class TestLoadConversationFromJsonl:
    def test_load_last_conversation(self, tmp_path):
        from conversation_dataset_generator.output import load_conversation_from_jsonl
        # Write test data
        outfile = str(tmp_path / "data.jsonl")
        with open(outfile, "w") as f:
            for row in [
                {"conversation_id": 0, "turn_number": 0, "role": "human", "speaker_name": "Alice", "topic": "T1", "scenario": "S1", "style": "St1", "include_points": "", "content": "Hello"},
                {"conversation_id": 0, "turn_number": 1, "role": "gpt", "speaker_name": "Bob", "topic": "T1", "scenario": "S1", "style": "St1", "include_points": "", "content": "Hi"},
                {"conversation_id": 1, "turn_number": 0, "role": "human", "speaker_name": "Alice", "topic": "T2", "scenario": "S2", "style": "St2", "include_points": "", "content": "Bye"},
            ]:
                f.write(json.dumps(row) + "\n")

        result = load_conversation_from_jsonl(outfile)
        assert result["conversation_id"] == 1
        assert len(result["turns"]) == 1
        assert result["turns"][0]["value"] == "Bye"
        assert result["topic"] == "T2"
        assert "Alice" in result["speaker_names"]

    def test_load_specific_conversation(self, tmp_path):
        from conversation_dataset_generator.output import load_conversation_from_jsonl
        outfile = str(tmp_path / "data.jsonl")
        with open(outfile, "w") as f:
            for row in [
                {"conversation_id": 0, "turn_number": 0, "role": "human", "speaker_name": "Alice", "topic": "T1", "scenario": "S1", "style": "St1", "include_points": "", "content": "Hello"},
                {"conversation_id": 1, "turn_number": 0, "role": "human", "speaker_name": "Bob", "topic": "T2", "scenario": "S2", "style": "St2", "include_points": "", "content": "Hey"},
            ]:
                f.write(json.dumps(row) + "\n")

        result = load_conversation_from_jsonl(outfile, conversation_id=0)
        assert result["conversation_id"] == 0
        assert result["turns"][0]["value"] == "Hello"

    def test_missing_conversation_id_returns_none(self, tmp_path):
        from conversation_dataset_generator.output import load_conversation_from_jsonl
        outfile = str(tmp_path / "data.jsonl")
        with open(outfile, "w") as f:
            f.write(json.dumps({"conversation_id": 0, "turn_number": 0, "role": "human", "speaker_name": "A", "topic": "T", "scenario": "S", "style": "St", "include_points": "", "content": "Hi"}) + "\n")

        result = load_conversation_from_jsonl(outfile, conversation_id=99)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_output.py::TestLoadConversationFromJsonl -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Update write_jsonl to use speaker_name from turns**

The current `write_jsonl` maps human→persona1_name and gpt→persona2_name. For N speakers, each turn already has a `"from"` role, but we need the speaker name stored per-turn. Update the conversation dict format:

Replace `write_jsonl` in `conversation_dataset_generator/output.py`:

```python
def write_jsonl(conversations: list, output_path: str) -> int:
    """Write conversations to a JSONL file.

    Each conversation dict must have:
        turns: list of {"from": role, "value": text, "speaker_name": name}
        topic, scenario, style, include_points

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

            # Legacy compat: if turns don't have speaker_name, use persona1/2_name
            persona1_name = conv.get("persona1_name", "")
            persona2_name = conv.get("persona2_name", "")

            for turn_num, turn in enumerate(turns):
                role = turn.get("from", "")
                content = turn.get("value", "")
                speaker_name = turn.get("speaker_name", "")

                # Fallback to legacy persona mapping
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
```

- [ ] **Step 4: Add load_conversation_from_jsonl**

Add to `conversation_dataset_generator/output.py`:

```python
def load_conversation_from_jsonl(
    path: str, conversation_id: int | None = None
) -> dict | None:
    """Load a conversation from a JSONL file.

    Args:
        path: Path to the JSONL file.
        conversation_id: Specific conversation to load. If None, loads the last one.

    Returns:
        Dict with keys: conversation_id, turns, topic, scenario, style,
        include_points, speaker_names. Or None if not found.
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

    # Select target conversation
    if conversation_id is not None:
        if conversation_id not in rows_by_conv:
            logger.warning("Conversation ID %d not found in %s", conversation_id, path)
            return None
        target_id = conversation_id
    else:
        target_id = max(rows_by_conv.keys())

    rows = rows_by_conv[target_id]
    rows.sort(key=lambda r: r["turn_number"])

    # Extract metadata from first row
    first = rows[0]
    speaker_names = list(dict.fromkeys(r["speaker_name"] for r in rows))

    turns = []
    for r in rows:
        turns.append({
            "from": r["role"],
            "value": r["content"],
            "speaker_name": r["speaker_name"],
        })

    return {
        "conversation_id": target_id,
        "turns": turns,
        "topic": first.get("topic", ""),
        "scenario": first.get("scenario", ""),
        "style": first.get("style", ""),
        "include_points": first.get("include_points", ""),
        "speaker_names": speaker_names,
    }
```

- [ ] **Step 5: Update existing write_jsonl tests for new turn format**

The existing tests in `TestWriteJsonl` use `persona1_name`/`persona2_name` in the conversation dict and turns without `speaker_name`. These should still work due to the legacy fallback. Run them to confirm.

- [ ] **Step 6: Run all output tests**

Run: `pytest tests/test_output.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 7: Commit**

```bash
git add conversation_dataset_generator/output.py tests/test_output.py
git commit -m "update output for N-speaker turns and add load_conversation_from_jsonl"
```

---

### Task 5: Update Character Pool for Group Selection

**Files:**
- Modify: `conversation_dataset_generator/character_pool.py`
- Modify: `tests/test_character_pool.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_character_pool.py`:

```python
from conversation_dataset_generator.character_pool import select_random_group


class TestSelectRandomGroup:
    def test_default_count_is_two(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions)
        assert len(group) == 2
        names = [name for name, _ in group]
        assert len(set(names)) == 2

    def test_count_three(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions, count=3)
        assert len(group) == 3
        names = [name for name, _ in group]
        assert len(set(names)) == 3

    def test_returns_name_desc_tuples(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions, count=2)
        for name, desc in group:
            assert name in characters
            assert desc == descriptions[name]

    def test_never_duplicates(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        for _ in range(50):
            group = select_random_group(characters, descriptions, count=2)
            names = [n for n, _ in group]
            assert len(set(names)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_character_pool.py::TestSelectRandomGroup -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Add select_random_group, keep select_random_pair as alias**

Add to `conversation_dataset_generator/character_pool.py`:

```python
def select_random_group(
    characters: list[str], descriptions: dict[str, str], count: int = 2
) -> list[tuple[str, str]]:
    """Select N random characters and return list of (name, desc) tuples."""
    selected = random.sample(characters, count)
    return [(name, descriptions[name]) for name in selected]
```

Keep `select_random_pair` as a backward-compat wrapper:

```python
def select_random_pair(
    characters: list[str], descriptions: dict[str, str]
) -> tuple[str, str, str, str]:
    """Select two random characters. Legacy compat — use select_random_group instead."""
    group = select_random_group(characters, descriptions, count=2)
    return group[0][0], group[0][1], group[1][0], group[1][1]
```

- [ ] **Step 4: Run all character pool tests**

Run: `pytest tests/test_character_pool.py -v`
Expected: All tests PASS (old + new)

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/character_pool.py tests/test_character_pool.py
git commit -m "add select_random_group for N-speaker character selection"
```

---

### Task 6: Update Generation for N Speakers + Continuation

**Files:**
- Modify: `conversation_dataset_generator/generation.py`
- Modify: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_generation.py`:

```python
class TestGenerateConversationMulti:
    def test_three_speakers(self):
        response = "Alice: Hello\nBob: Hi\nCharlie: Hey everyone"
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()
        turns = generate_conversation(
            topic="Greet",
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy"), ("Charlie", "Quiet")],
            scenario="Room", style="Casual",
            generator_pipeline=pipeline, tokenizer=tokenizer,
            max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 3

    def test_legacy_two_speaker(self):
        response = "Alice: Hello\nBob: Hi"
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()
        turns = generate_conversation(
            topic="Greet", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Room", style="Casual",
            generator_pipeline=pipeline, tokenizer=tokenizer,
            max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 2


class TestGenerateContinuation:
    def test_basic_continuation(self):
        from conversation_dataset_generator.generation import generate_continuation
        response = "Alice: Continuing now\nBob: Great"
        pipeline = make_mock_pipeline(response)
        tokenizer = make_mock_tokenizer()
        prior_turns = [
            {"from": "human", "value": "Hello", "speaker_name": "Alice"},
            {"from": "gpt", "value": "Hi", "speaker_name": "Bob"},
        ]
        turns = generate_continuation(
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy")],
            prior_turns=prior_turns,
            topic="Greet", scenario="Room", style="Casual",
            generator_pipeline=pipeline, tokenizer=tokenizer,
            max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation.py::TestGenerateConversationMulti tests/test_generation.py::TestGenerateContinuation -v`
Expected: FAIL

- [ ] **Step 3: Update generate_conversation for N speakers**

Replace in `conversation_dataset_generator/generation.py`:

```python
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

    Supports legacy 2-speaker and N-speaker modes.
    """
    # Normalize to personas list
    if personas is None:
        if persona1 and persona2:
            personas = [(persona1, persona1_desc or ""), (persona2, persona2_desc or "")]
        else:
            return None

    persona_names = [name for name, _ in personas]

    messages = build_conversation_messages(
        topic=topic, personas=personas,
        scenario=scenario, style=style, include_points=include_points,
    )

    text = _call_pipeline(
        generator_pipeline, tokenizer, messages,
        max_new_tokens=max_new_tokens,
    )

    if not text:
        logger.warning("generate_conversation: pipeline returned no text.")
        return None

    # Validate output starts with any persona name
    text_stripped = text.lstrip()
    if not any(text_stripped.startswith(f"{name}:") for name in persona_names):
        logger.warning(
            "Conversation doesn't start with a persona prefix. First 100 chars: %s",
            text[:100],
        )
        return None

    # Build role mapping if not provided
    if role_mapping is None:
        role_mapping = {}
        for i, name in enumerate(persona_names):
            role_mapping[name] = "human" if i == 0 else "gpt"

    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping,
    )

    if turns:
        # Add speaker_name to each turn
        name_pattern = re.compile(
            r"^\s*(" + "|".join(re.escape(n) for n in persona_names) + r")\s*:",
            re.IGNORECASE,
        )
        # Re-derive speaker names from the original text lines
        lines = text.strip().split("\n")
        turn_idx = 0
        for line in lines:
            line_s = line.strip()
            if not line_s:
                continue
            m = name_pattern.match(line_s)
            if m and turn_idx < len(turns):
                speaker = m.group(1).strip()
                # Find matching canonical name
                for name in persona_names:
                    if speaker.lower() == name.lower():
                        turns[turn_idx]["speaker_name"] = name
                        break
                turn_idx += 1

        num_tokens = len(tokenizer.encode(text))
        logger.info("Generated %d turns, %d tokens", len(turns), num_tokens)

    return turns
```

Add `import re` at the top of the file if not already present.

- [ ] **Step 4: Add generate_continuation**

Add to `conversation_dataset_generator/generation.py`:

```python
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
    """Continue an existing conversation.

    Args:
        personas: List of (name, description) tuples.
        prior_turns: Prior turns with 'speaker_name' and 'value' keys.
        topic, scenario, style: Conversation metadata.
        generator_pipeline: HuggingFace pipeline.
        tokenizer: Paired tokenizer.
        max_new_tokens: Max tokens to generate.
        role_mapping: Optional {name: role} dict.

    Returns:
        List of new turn dicts, or None on failure.
    """
    from conversation_dataset_generator.prompts import build_continuation_messages

    persona_names = [name for name, _ in personas]

    messages = build_continuation_messages(
        personas=personas, prior_turns=prior_turns,
        topic=topic, scenario=scenario, style=style,
    )

    text = _call_pipeline(
        generator_pipeline, tokenizer, messages,
        max_new_tokens=max_new_tokens,
    )

    if not text:
        logger.warning("generate_continuation: pipeline returned no text.")
        return None

    if role_mapping is None:
        role_mapping = {}
        for i, name in enumerate(persona_names):
            role_mapping[name] = "human" if i == 0 else "gpt"

    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping,
    )

    if turns:
        # Add speaker_name to turns (same logic as generate_conversation)
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

    return turns
```

- [ ] **Step 5: Update existing generation tests for new signature**

The existing `TestGenerateConversation` tests use `persona1`/`persona2` kwargs — these should still work via legacy compat. Run to confirm.

- [ ] **Step 6: Run all generation tests**

Run: `pytest tests/test_generation.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add conversation_dataset_generator/generation.py tests/test_generation.py
git commit -m "update generation for N-speaker and add continuation support"
```

---

### Task 7: Update CLI for All New Flags

**Files:**
- Modify: `conversation_dataset_generator/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for new flags**

Add to `tests/test_cli.py`:

```python
class TestNewFlags:
    def test_persona_flag_repeatable(self):
        parser = build_parser()
        args = parser.parse_args([
            "--persona", "Alice", "Friendly",
            "--persona", "Bob", "Grumpy",
            "--topic", "T", "--scenario", "S", "--style", "St",
        ])
        assert args.persona == [["Alice", "Friendly"], ["Bob", "Grumpy"]]

    def test_personas_file_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--personas", "chars.yaml",
            "--topic", "T", "--scenario", "S", "--style", "St",
        ])
        assert args.personas == "chars.yaml"

    def test_train_speaker_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test",
            "--train-speaker", "Captain America",
        ])
        assert args.train_speaker == "Captain America"

    def test_continue_from_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl"])
        assert args.continue_from == "data.jsonl"

    def test_conversation_id_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--continue-from", "data.jsonl",
            "--conversation-id", "5",
        ])
        assert args.conversation_id == 5

    def test_group_size_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings",
            "--character-pool", "c.yaml", "--persona-desc-pool", "d.yaml",
            "--initial-topic", "T", "--initial-scenario", "S", "--initial-style", "St",
            "--group-size", "3",
        ])
        assert args.group_size == 3

    def test_group_size_default_is_two(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.group_size == 2


class TestDetectModeContinue:
    def test_continue_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl"])
        mode = detect_mode(args, parser)
        assert mode == "continue"

    def test_continue_with_conversation_id(self):
        parser = build_parser()
        args = parser.parse_args([
            "--continue-from", "data.jsonl",
            "--conversation-id", "3",
        ])
        mode = detect_mode(args, parser)
        assert mode == "continue"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestNewFlags tests/test_cli.py::TestDetectModeContinue -v`
Expected: FAIL

- [ ] **Step 3: Add new flags to build_parser**

Add these argument groups to `build_parser()` in `conversation_dataset_generator/cli.py`:

In the "Mode Selection" group, add:
```python
    mode_group.add_argument(
        "--continue-from", type=str,
        help="Continue conversations from an existing JSONL file.",
    )
```

Add a new "Multi-Speaker" group after the Manual Mode group:
```python
    # Multi-speaker args
    multi = parser.add_argument_group("Multi-Speaker")
    multi.add_argument("--persona", nargs=2, action="append", metavar=("NAME", "DESC"),
                       help="Add a persona (repeatable). E.g. --persona 'Alice' 'Friendly'")
    multi.add_argument("--personas", type=str,
                       help="Path to YAML file with personas list.")
    multi.add_argument("--train-speaker", type=str, default=None,
                       help="Speaker name to assign 'gpt' role (rest become 'human').")
    multi.add_argument("--group-size", type=int, default=2,
                       help="Number of characters per conversation in random pairings mode.")
```

Add to the "Mode Selection" group (or General):
```python
    mode_group.add_argument(
        "--conversation-id", type=int, default=None,
        help="Specific conversation ID to continue (default: last).",
    )
```

- [ ] **Step 4: Update detect_mode for continue mode**

Add at the top of `detect_mode()`, before the brief check:

```python
    if args.continue_from:
        return "continue"
```

- [ ] **Step 5: Add build_role_mapping helper**

Add to `conversation_dataset_generator/cli.py`:

```python
def build_role_mapping(
    persona_names: list[str],
    train_speaker: str | None = None,
    role_mapping_str: str | None = None,
) -> dict:
    """Build a {speaker_name: role} dict from CLI flags.

    Priority: train_speaker > role_mapping_str > default (first=human, rest=gpt).
    """
    if train_speaker:
        mapping = {}
        for name in persona_names:
            mapping[name] = "gpt" if name == train_speaker else "human"
        return mapping

    if role_mapping_str:
        mapping = {}
        for part in role_mapping_str.split(","):
            key, _, value = part.strip().partition("=")
            key = key.strip()
            value = value.strip()
            if key and value:
                mapping[key] = value
        return mapping

    # Default: first speaker human, rest gpt
    mapping = {}
    for i, name in enumerate(persona_names):
        mapping[name] = "human" if i == 0 else "gpt"
    return mapping
```

- [ ] **Step 6: Add load_personas helper**

Add to `conversation_dataset_generator/cli.py`:

```python
def load_personas_from_yaml(path: str) -> list[tuple[str, str]]:
    """Load personas from a YAML file.

    Expected format:
        personas:
          - name: "Alice"
            description: "Friendly person"
          - name: "Bob"
            description: "Grumpy person"
    """
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "personas" not in data:
        raise ValueError(f"Personas YAML must contain a 'personas' key. Found: {list(data.keys()) if isinstance(data, dict) else type(data)}")

    personas = []
    for entry in data["personas"]:
        name = entry.get("name", "")
        desc = entry.get("description", "")
        if name:
            personas.append((name, desc))

    if len(personas) < 2:
        raise ValueError(f"Need at least 2 personas, found {len(personas)}.")

    return personas
```

- [ ] **Step 7: Update main() to handle all new modes**

This is the largest change. Update `main()` in `cli.py` to:

1. Normalize personas from any source (legacy `--persona1/2`, `--persona`, `--personas`, or from `--continue-from`)
2. Build role mapping using `build_role_mapping()`
3. Handle continue mode
4. Pass personas list through the pipeline instead of persona1/persona2

The key changes in the generation loop:
- `generate_conversation()` called with `personas=` kwarg
- Conversation dict stores turns with `speaker_name` per turn
- Continue mode calls `generate_continuation()` instead

Update the persona normalization at the top of mode handling:

```python
    # Normalize personas from any input method
    personas = None

    if args.persona:
        # --persona "Name" "Desc" repeated
        personas = [(name, desc) for name, desc in args.persona]
    elif args.personas:
        # --personas file.yaml
        personas = load_personas_from_yaml(args.personas)
    elif args.persona1 and args.persona2:
        # Legacy --persona1/--persona2
        personas = [
            (args.persona1, args.persona1_desc or ""),
            (args.persona2, args.persona2_desc or ""),
        ]
```

For continue mode:

```python
    if mode == "continue":
        from conversation_dataset_generator.output import load_conversation_from_jsonl

        conv_data = load_conversation_from_jsonl(
            args.continue_from, conversation_id=args.conversation_id
        )
        if conv_data is None:
            logger.error("Could not load conversation from %s", args.continue_from)
            sys.exit(1)

        # Use personas from loaded data if not specified on CLI
        if personas is None:
            personas = [(name, "") for name in conv_data["speaker_names"]]

        persona_names = [name for name, _ in personas]
        role_mapping = build_role_mapping(persona_names, args.train_speaker, args.role_mapping)

        continuation_turns = generate_continuation(
            personas=personas,
            prior_turns=conv_data["turns"],
            topic=conv_data["topic"],
            scenario=conv_data["scenario"],
            style=conv_data["style"],
            generator_pipeline=text_generator,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            role_mapping=role_mapping,
        )

        if continuation_turns:
            conversations.append({
                "turns": continuation_turns,
                "topic": conv_data["topic"],
                "scenario": conv_data["scenario"],
                "style": conv_data["style"],
                "include_points": conv_data.get("include_points", ""),
            })
```

For the standard generation loop, pass `personas=` and add speaker_name to turns:

```python
        turns = generate_conversation(
            topic=current_topic,
            personas=personas_for_this_conv,
            scenario=current_scenario,
            style=current_style,
            generator_pipeline=text_generator,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            include_points=current_include_points,
            role_mapping=role_mapping,
        )
```

- [ ] **Step 8: Run all CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: All tests PASS

- [ ] **Step 9: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add conversation_dataset_generator/cli.py tests/test_cli.py
git commit -m "add multi-speaker, continue-from, train-speaker, group-size CLI flags"
```

---

### Task 8: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add multi-speaker section to Modes**

After the "Random Pairings" section, add:

```markdown
### Multi-Speaker (3+ Personas)

Use `--persona` (repeatable) for inline definitions or `--personas` for a YAML file:

```bash
# Inline
python generate.py \
  --persona "Iron Man" "Genius billionaire with rapid-fire wit" \
  --persona "Captain America" "Principled, earnest, old-fashioned" \
  --persona "Thor" "Boisterous god with Shakespearean formality" \
  --topic "who pays for the pizza" --scenario "Avengers break room" --style "comedic argument" \
  --num-examples 10 --output-file avengers_pizza.jsonl

# From YAML file
python generate.py \
  --personas my_characters.yaml \
  --topic "planning a heist" --scenario "warehouse" --style "tense thriller" \
  --num-examples 5 --output-file heist.jsonl
```

### Continuing Conversations

Extend an existing conversation with more turns:

```bash
# Continue the last conversation in a file
python generate.py --continue-from conversations.jsonl --output-file more.jsonl

# Continue a specific conversation
python generate.py --continue-from conversations.jsonl --conversation-id 5 --output-file more.jsonl
```
```

- [ ] **Step 2: Add Role Mapping for Training section**

After the Output Format section, add:

```markdown
## Role Mapping for Training

The `role` field in the output determines how training frameworks interpret each turn:
- `"human"` = input/context (the model sees this)
- `"gpt"` = target (the model learns to generate this)

**Default:** First persona is `"human"`, all others are `"gpt"`.

**Train a specific character:** Use `--train-speaker` to make one character the `"gpt"` role:

```bash
# Train the model to BE Captain America
python generate.py \
  --persona "Iron Man" "Genius billionaire" \
  --persona "Captain America" "Principled leader" \
  --persona "Thor" "Boisterous god" \
  --train-speaker "Captain America" \
  --topic "mission planning" --scenario "war room" --style "serious" \
  --output-file cap_training.jsonl
```

In the output, Captain America's turns will have `"role": "gpt"` and everyone else will have `"role": "human"`. The `speaker_name` field always stores the actual character name regardless.

**Fine-grained control:** Use `--role-mapping` for custom assignments:

```bash
--role-mapping "Iron Man=human,Captain America=gpt,Thor=human"
```
```

- [ ] **Step 3: Update Argument Reference table**

Add new flags to the appropriate table sections:

Multi-Speaker table:
```markdown
### Multi-Speaker

| Flag | Description |
|---|---|
| `--persona NAME DESC` | Add a persona (repeatable) |
| `--personas FILE` | YAML file with personas list |
| `--train-speaker NAME` | Assign this speaker the "gpt" role |
| `--group-size N` | Characters per conversation in random pairings (default: 2) |
```

Continue Conversation table:
```markdown
### Continue Conversation

| Flag | Description |
|---|---|
| `--continue-from FILE` | Continue from an existing JSONL file |
| `--conversation-id N` | Specific conversation to continue (default: last) |
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "document multi-speaker, continuation, and role mapping in README"
```

---

### Task 9: Integration Test

- [ ] **Step 1: Test 3-speaker manual mode**

Run:
```bash
python generate.py \
  --persona "Iron Man" "Genius billionaire with rapid-fire wit and tech jargon" \
  --persona "Captain America" "Principled and earnest, old-fashioned politeness" \
  --persona "Thor" "Boisterous god with Shakespearean formality" \
  --topic "who should lead the next mission" --scenario "Avengers war room" --style "heated debate" \
  --num-examples 2 --output-file test_multi.jsonl \
  --model-id unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit
```
Expected: 2 conversations with 3 speakers

- [ ] **Step 2: Verify 3 speakers in output**

Run: `python -c "import json; names=set(json.loads(l)['speaker_name'] for l in open('test_multi.jsonl')); print(f'Speakers: {names}'); assert len(names)==3"`
Expected: 3 unique speaker names

- [ ] **Step 3: Test --train-speaker**

Run:
```bash
python generate.py \
  --persona "Iron Man" "Genius billionaire" \
  --persona "Captain America" "Principled leader" \
  --topic "tech vs tradition" --scenario "lab" --style "debate" \
  --train-speaker "Captain America" \
  --num-examples 1 --output-file test_train.jsonl \
  --model-id unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit
```

Run: `python -c "import json; [print(f'{json.loads(l)[\"speaker_name\"]}: {json.loads(l)[\"role\"]}') for l in open('test_train.jsonl')]"`
Expected: Captain America → gpt, Iron Man → human

- [ ] **Step 4: Test --continue-from**

Run:
```bash
python generate.py \
  --continue-from test_multi.jsonl \
  --num-examples 1 --output-file test_continued.jsonl \
  --model-id unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit
```
Expected: Generates continuation of the last conversation

- [ ] **Step 5: Clean up**

```bash
rm -f test_multi.jsonl test_train.jsonl test_continued.jsonl
```

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git commit --allow-empty -m "integration tests passed: multi-speaker, train-speaker, continue-from"
```
