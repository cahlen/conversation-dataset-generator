# Conversation Dataset Generator — Modernization Design

## Overview

Modular rewrite of the conversation dataset generator. Break the ~1800-line `generate.py` monolith into a proper Python package with testable modules, fix critical bugs (variation regex, role mapping), remove dead features, update defaults, and add comprehensive test coverage.

The CLI interface is preserved so `batch_generate.py` and existing YAML configs continue to work.

## Problem Statement

The current codebase has these issues:

1. **Variation is broken** — the regex parser for LLM variation output silently fails ~30-50% of the time due to `re.DOTALL` flag combined with trailing whitespace. Falls back to initial values with only a WARNING log, so all conversations in a dataset end up with the same topic/scenario/style.
2. **God script** — everything lives in one 1800-line file with ~500 lines of procedural `if __name__ == "__main__"` code.
3. **Zero tests** — no way to verify changes don't break things.
4. **Dead features** — `--delete-repo` and image search add complexity without value.
5. **Outdated defaults** — hardcoded to `meta-llama/Meta-Llama-3-8B-Instruct` with 768 max tokens.
6. **Anti-patterns** — placeholder sentinel values, duplicate data loading, imports inside loops, bare `except Exception` blocks, excessive debug logging.

## Package Structure

```
conversation_dataset_generator/
  __init__.py              # version, public API
  cli.py                   # argparse, mode detection, orchestration
  models.py                # model/tokenizer loading, pipeline creation, quantization config
  prompts.py               # all system prompts + prompt builders
  generation.py            # LLM call wrappers with retry, brief->args, topic variation
  parsing.py               # raw LLM text -> ShareGPT turns
  output.py                # JSONL serialization, HF Dataset creation, dataset card templates
  hub.py                   # HF Hub upload (optional, isolated)
  character_pool.py        # YAML pool loading, validation, random pair selection
  web_search.py            # DuckDuckGo persona context search

generate.py                # thin entry point: imports cli.main()
batch_generate.py          # unchanged subprocess orchestrator
tests/
  test_parsing.py
  test_prompts.py
  test_generation.py
  test_output.py
  test_character_pool.py
  test_cli.py
```

### Module Responsibilities

**cli.py** — Argument parsing, mode detection (brief/manual/fixed_persona_variation/random_pairings/random_pairings_variation), validation of required args per mode, conflict checking. Orchestrates the generation loop by calling into other modules. No LLM logic lives here.

**models.py** — `load_model_and_tokenizer(model_id, load_in_4bit)` returns a `(pipeline, tokenizer)` tuple. Encapsulates quantization config, device mapping, dtype selection. Single place to change model loading behavior.

**prompts.py** — Contains all system prompt constants (`ARG_GENERATION_SYSTEM_PROMPT`, `TOPIC_VARIATION_SYSTEM_PROMPT`, conversation system prompt). Contains prompt builder functions: `build_arg_generation_messages()`, `build_conversation_messages()`, `build_variation_messages()`. Pure functions, no LLM calls.

**generation.py** — `generate_args_from_brief()`, `generate_topic_variation()`, `generate_conversation()`. Each wraps an LLM pipeline call with retry logic and response extraction. Calls into `prompts.py` for message construction and `parsing.py` for output parsing.

**parsing.py** — `parse_conversation_to_sharegpt()`, `parse_arg_generation_output()`, `parse_variation_output()`. Pure regex-based parsers. The critical bug fixes live here.

**output.py** — `write_jsonl()`, `build_dataset_card()`, `create_hf_dataset()`. Dataset card templates are clean markdown with mode-specific sections. No image references.

**hub.py** — `upload_to_hub()`. Isolated so the rest of the tool works without HF auth. Handles token checking, dataset push, README upload.

**character_pool.py** — `load_character_pool()`, `load_description_pool()`, `validate_pools()`, `select_random_pair()`. Loads once, no duplicate reloading.

**web_search.py** — `get_persona_context(name)`. DuckDuckGo text search for persona enrichment in creative brief mode. Image search removed entirely.

## Bug Fixes

### Variation Regex (Critical)

**Current code (broken):**
```python
arg_pattern = re.compile(r'^--(topic|scenario|style)\s+"(.*?)"$', re.MULTILINE | re.DOTALL)
```

**Problems:**
- `re.DOTALL` makes `.` match newlines, causing greedy cross-line captures
- `$` anchor fails when lines have trailing whitespace
- Only double quotes supported; LLMs sometimes output single quotes
- When parsing fails, returns `None` silently, causing fallback to initial values

**Fix:**
- Remove `re.DOTALL` flag
- Strip each line before matching
- Support both quote styles
- Log actual match results on failure
- Apply same fix to `parse_arg_generation_output()` (same bug exists there)

### Role Mapping

**Current:** Hardcoded persona1->"human", persona2->"gpt". Semantically wrong when persona1 is an AI character.

**Fix:** Add optional `--role-mapping` CLI flag accepting `p1=human,p2=gpt` or `p1=gpt,p2=human` syntax. Default: `p1=human,p2=gpt` (preserves current behavior).

### Duplicate Character Pool Loading

**Current:** Pools loaded once during mode setup (lines 741-817), then reloaded again at the start of the generation loop (lines 1120-1148) "to ensure availability."

**Fix:** Load once in `character_pool.py`, pass the loaded data through. No reload.

### Placeholder Sentinel Values

**Current:** `base_persona1 = "PLACEHOLDER_WILL_BE_SELECTED_RANDOMLY"` used as sentinel in random pairing modes.

**Fix:** Use `None` and check for it explicitly. Random pairing modes set personas inside the generation loop, no sentinels needed.

## Feature Changes

### Removed
- `--delete-repo` flag and all deletion logic
- Image search (`get_persona_image_url`, `_image_url_cache`)
- Image URLs in dataset cards
- `pandas` dependency (unused)
- `peft` and `trl` dependencies (only referenced in card example code)

### Updated Defaults
- Default model: `Qwen/Qwen2.5-7B-Instruct` (from `meta-llama/Meta-Llama-3-8B-Instruct`)
- Default `--max-new-tokens`: 2048 (from 768)

### Improved Dataset Cards
- Clean markdown templates with proper YAML frontmatter
- Mode-specific sections describing generation parameters
- No image references
- Proper HF metadata (license, tags, language)

## Testing Strategy

### Pure Logic (no mocks)

**test_parsing.py** — Most important test file:
- Normal multi-turn conversation parsing
- Trailing whitespace, mixed quotes, empty turns
- Conversations that don't start with expected prefix
- Single-turn edge cases
- Case-insensitive speaker matching
- Variation output parsing (the fixed regex)
- Arg generation output parsing

**test_prompts.py:**
- All four modes produce valid message list structures
- Include-points injected correctly
- Persona/topic/scenario placeholders present in output

**test_output.py:**
- JSONL round-trip: write then read back
- Dataset card templates render for each mode
- Edge cases: empty conversations, special characters

**test_character_pool.py:**
- Valid pools load correctly
- Missing descriptions detected
- Pool with < 2 characters rejected
- Random pairs never duplicate

### Mocked LLM (test_generation.py)
- Mock pipeline returns canned output
- Brief->args parsing with good/malformed output
- Retry logic fires on failures
- Fallback defaults applied for missing fields

### CLI Tests (test_cli.py)
- Argument parsing and mode detection only
- Required args enforced per mode
- Conflicting args caught
- Default values applied

### Not Tested
- Actual LLM inference quality (non-deterministic)

## Backward Compatibility

### Preserved
- All CLI flags (except `--delete-repo`)
- `generate.py` remains the entry point
- `batch_generate.py` works unchanged
- Existing YAML batch configs remain valid
- Output JSONL schema unchanged

### Intentionally Breaking
- `--delete-repo` removed (unrecognized argument error)
- Image URLs no longer in dataset cards
- Default model changes (users who specified `--model-id` explicitly are unaffected)

## Dependencies

**requirements.txt:**
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

**Dev dependencies (separate or in requirements-dev.txt):**
```
pytest
```

**Removed:** `pandas`, `peft`, `trl`
