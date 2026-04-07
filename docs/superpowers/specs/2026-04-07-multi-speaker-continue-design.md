# Multi-Speaker + Continue Conversations — Design Spec

## Overview

Add support for N-persona conversations (3+ speakers) and the ability to continue existing conversations from a JSONL file. Also fix stale model references in example YAML files.

Addresses GitHub issue #3 (Multi-persona Chat Generation).

## Multi-Speaker Personas

### CLI Interface (three ways, all coexist)

**Legacy (backward compat):**
```bash
--persona1 "Tony" --persona1-desc "Chef" --persona2 "Dave" --persona2-desc "Critic"
```
Internally converted to a persona list.

**Inline N-speaker:**
```bash
--persona "Iron Man" "Genius billionaire" --persona "Cap" "Earnest leader" --persona "Thor" "Boisterous god"
```
Each `--persona` takes exactly 2 values: name and description. Can be repeated N times.

**YAML file:**
```bash
--personas characters.yaml
```
File format:
```yaml
personas:
  - name: "Iron Man"
    description: "Genius billionaire with rapid-fire wit"
  - name: "Captain America"
    description: "Principled, earnest, old-fashioned"
  - name: "Thor"
    description: "Boisterous god, Shakespearean formality"
```

Minimum 2 personas required. `--persona1`/`--persona2`, `--persona`, and `--personas` are mutually exclusive.

### Role Mapping for Training

The `role` field in JSONL output determines how fine-tuning frameworks interpret each turn:
- `"human"` = context/input (the turn the model sees)
- `"gpt"` = target (the turn the model learns to generate)

**Default:** First persona → `"human"`, all others → `"gpt"`.

**`--train-speaker NAME`:** The named speaker becomes `"gpt"`, everyone else becomes `"human"`. Use this when training a model to be a specific character.

```bash
--train-speaker "Captain America"
```

**`--role-mapping MAP`:** Fine-grained control. Comma-separated `name=role` pairs.

```bash
--role-mapping "Iron Man=human,Captain America=gpt,Thor=human"
```

`--train-speaker` and `--role-mapping` are mutually exclusive.

The `speaker_name` field always stores the actual character name regardless of role assignment.

## Continue Existing Conversations

### CLI Flags

```bash
# Continue the last conversation
--continue-from data.jsonl

# Continue a specific conversation
--continue-from data.jsonl --conversation-id 5
```

### How It Works

1. Load the target JSONL file.
2. Extract the target conversation's turns (last conversation by default, or specific `--conversation-id`).
3. Read `speaker_name` values to identify personas. If `--persona`/`--personas` flags also provided, use those descriptions. Otherwise, instruct the LLM to infer personality from conversation history.
4. Inject prior turns into the prompt as "conversation so far".
5. LLM generates continuation turns.
6. Preserve original topic/scenario/style metadata.
7. Write new turns: append to same file or write to new `--output-file`.

### New Prompt Function

`build_continuation_messages(personas, prior_turns, topic, scenario, style)` — constructs a prompt with the conversation history and asks the LLM to continue naturally, maintaining each speaker's established voice.

## Module Changes

### parsing.py

- `parse_conversation_to_sharegpt(text, personas, role_mapping)` — `personas` changes from two separate strings to a list of persona names. `role_mapping` changes from `{"p1": "human", "p2": "gpt"}` to `{speaker_name: role}` dict. Matches any persona name in the list.
- No changes to `parse_variation_output` or `parse_arg_generation_output`.

### prompts.py

- `build_conversation_messages(topic, personas, scenario, style, include_points)` — `personas` is a list of `(name, description)` tuples. System prompt lists all N personas with descriptions.
- `build_continuation_messages(personas, prior_turns, topic, scenario, style)` — new function. Injects conversation history, asks LLM to continue.
- `build_arg_generation_messages` — prompt updated to instruct LLM to generate N `--persona` entries.

### generation.py

- `generate_conversation` — accept list of personas, pass to updated parser.
- `generate_continuation(personas, prior_turns, topic, scenario, style, generator_pipeline, tokenizer, max_new_tokens, role_mapping)` — new function. Uses `build_continuation_messages`, generates, parses.
- `generate_args_from_brief` — handle N persona output.

### character_pool.py

- `select_random_pair` → `select_random_group(characters, descriptions, count=2)`. Returns list of `(name, desc)` tuples.
- New `--group-size N` CLI flag for random pairings mode.

### output.py

- `write_jsonl` — role assignment uses `{speaker_name: role}` mapping dict.
- `build_dataset_card` — list all N personas.
- `load_conversation_from_jsonl(path, conversation_id=None)` — new function. Loads turns for a specific conversation (or the last one). Returns turns list + metadata (topic, scenario, style, speaker names).

### cli.py

- New flags: `--persona` (nargs=2, action=append), `--personas` (file path), `--train-speaker`, `--continue-from`, `--conversation-id`, `--group-size`
- Legacy `--persona1`/`--persona2` converted to persona list internally.
- New continue mode: detected when `--continue-from` is present.
- `build_role_mapping(personas, train_speaker, role_mapping_str)` — new helper that builds the `{name: role}` dict from whichever flag was used.

## Example YAML Fixes

Remove `model_id` lines from all example YAML files so they use whatever default the tool has:
- `examples/batch_rockstars_celebs.yaml`
- `examples/batch_mixed_modes.yaml`
- `examples/ai_course_curriculum.yaml`

## Backward Compatibility

- `--persona1`/`--persona2` flags still work, internally converted to persona list.
- Output JSONL schema unchanged (same fields).
- `role` field still uses "human"/"gpt" strings.
- Default role mapping (first=human, rest=gpt) matches old behavior for 2-speaker case.
- `select_random_group(count=2)` matches old `select_random_pair` behavior.

## Documentation

README updated with:
- Multi-speaker examples (inline and YAML)
- "Role Mapping for Training" subsection explaining human/gpt roles, defaults, `--train-speaker`, `--role-mapping`, concrete training scenario
- Continue conversation examples
- `--group-size` flag in random pairings section
- Updated argument reference table with all new flags
