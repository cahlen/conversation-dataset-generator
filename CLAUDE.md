# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python tool that generates conversational dialogue datasets in ShareGPT format for LLM fine-tuning. Uses Hugging Face `transformers` pipelines to generate synthetic conversations between two personas, with optional upload to Hugging Face Hub.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for pytest
```

Requires a GPU with CUDA for model inference. Supports 4-bit quantization via `--load-in-4bit`.

## Running

**Single generation:**
```bash
python generate.py --creative-brief "brief description" --num-examples 5 --output-file output.jsonl
python generate.py --topic "topic" --persona1 "Name" --persona1-desc "desc" --persona2 "Name" --persona2-desc "desc" --scenario "scenario" --style "style" --output-file output.jsonl
```

**Batch generation** (runs multiple generate.py invocations from a YAML config):
```bash
python batch_generate.py examples/batch_mixed_modes.yaml
```

## Testing

```bash
pytest tests/ -v                                          # all tests
pytest tests/test_parsing.py -v                           # one module
pytest tests/test_parsing.py::TestParseVariationOutput -v # one class
pytest tests/test_parsing.py::TestParseVariationOutput::test_trailing_whitespace_on_lines -v  # one test
```

79 tests across 6 test files. No GPU required for tests — LLM calls are mocked.

## Architecture

### Package: `conversation_dataset_generator/`

| Module | Responsibility |
|---|---|
| `cli.py` | Argparse, mode detection, orchestration loop |
| `models.py` | Model/tokenizer loading, pipeline creation, quantization config |
| `prompts.py` | System prompt constants + message builder functions |
| `generation.py` | LLM call wrappers with retry logic (brief→args, variation, conversation) |
| `parsing.py` | Regex parsers: raw LLM text → ShareGPT turns, variation output, arg output |
| `output.py` | JSONL serialization, dataset card templates, HF Dataset creation |
| `hub.py` | HuggingFace Hub upload (isolated, optional) |
| `character_pool.py` | YAML pool loading, validation, random pair selection |
| `web_search.py` | DuckDuckGo persona context search for creative brief mode |

`generate.py` is a thin entry point that calls `cli.main()`.

### Five generation modes

1. **Creative Brief** (`--creative-brief`): LLM brainstorms personas/topic/scenario, generates variations per example
2. **Manual** (`--topic`, `--persona1`, etc.): All parameters specified, no variation
3. **Fixed Persona + Variation** (`--enable-variation` + `--fixed-persona*` + `--initial-*`): Fixed personas, LLM-varied topics
4. **Random Pairings** (`--random-pairings`): Random character pairs from YAML pools
5. **Random Pairings + Variation** (`--random-pairings --enable-variation`): Random pairs with topic variation

### Key defaults

- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Default max tokens: 2048
- Role mapping: persona1→"human", persona2→"gpt" (configurable via `--role-mapping`)

### character-config/

YAML files with character pools. Each domain (avengers, got, southpark, tech) has:
- `*_characters.yaml` — list of names under `characters` key
- `*_descriptions.yaml` — name-to-description mapping under `descriptions` key

### Output Format

JSONL with fields: conversation_id, turn_number, role, speaker_name, topic, scenario, style, include_points, content. Output files are gitignored (`*.jsonl`).

## Key Design Decisions

- Single model used for all LLM tasks (brief expansion, variation, conversation) via `--model-id`
- Web search (DuckDuckGo) enriches persona descriptions in creative brief mode
- Conversation parser uses regex to extract speaker turns from free-form LLM output
- `batch_generate.py` uses subprocess execution (not function imports) to isolate each run
- Variation regex uses `re.MULTILINE` only (no `re.DOTALL`) to correctly parse per-line key-value output
