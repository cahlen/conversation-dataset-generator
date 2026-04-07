# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python tool that generates synthetic conversational dialogue datasets in ShareGPT format for LLM fine-tuning. Supports 2+ speakers, topic variation, conversation continuation, and optional HuggingFace Hub upload.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for pytest
```

Requires Python 3.10+, NVIDIA GPU with CUDA. Supports 4-bit quantization via `--load-in-4bit`. Docker also available (see README).

## Running

```bash
# Creative brief (auto-generates personas + variation)
python generate.py --creative-brief "Sherlock and Watson debate AI" --num-examples 5 --output-file out.jsonl

# Manual 2-speaker
python generate.py --persona1 "Alice" --persona1-desc "desc" --persona2 "Bob" --persona2-desc "desc" --topic "T" --scenario "S" --style "St" --output-file out.jsonl

# Multi-speaker (3+ personas)
python generate.py --persona "Iron Man" "Genius billionaire" --persona "Cap" "Earnest leader" --persona "Thor" "Boisterous god" --topic "T" --scenario "S" --style "St" --output-file out.jsonl

# From personas YAML file
python generate.py --personas examples/avengers_personas.yaml --topic "T" --scenario "S" --style "St" --output-file out.jsonl

# Continue existing conversation
python generate.py --continue-from data.jsonl --output-file more.jsonl

# Batch generation
python batch_generate.py examples/batch_mixed_modes.yaml

# Evaluate generated data
python evaluate.py conversations.jsonl
python evaluate.py conversations.jsonl --no-embeddings    # skip embedding metrics
```

## Testing

```bash
pytest tests/ -v                                          # all 146 tests
pytest tests/test_parsing.py -v                           # one module
pytest tests/test_parsing.py::TestParseVariationOutput -v # one class
```

146 tests across 7 test files. No GPU required — LLM calls and embeddings are mocked.

## Architecture

### Package: `conversation_dataset_generator/`

| Module | Responsibility |
|---|---|
| `cli.py` | Argparse, mode detection, orchestration loop |
| `models.py` | Model/tokenizer loading, pipeline creation, quantization config |
| `prompts.py` | System prompt constants + message builders (conversation, variation, continuation) |
| `generation.py` | LLM call wrappers with retry logic (brief→args, variation, conversation, continuation) |
| `parsing.py` | Regex parsers: raw LLM text → ShareGPT turns (N-speaker), variation output, arg output |
| `output.py` | JSONL serialization, dataset card templates, load conversation from JSONL |
| `hub.py` | HuggingFace Hub upload (isolated, optional) |
| `character_pool.py` | YAML pool loading, validation, random group selection |
| `web_search.py` | DuckDuckGo persona context search for creative brief mode |
| `evaluation.py` | Intrinsic quality metrics (distinct-N, coherence, speaker distinctiveness) |

`generate.py` is a thin entry point that calls `cli.main()`.

### Generation Modes

1. **Creative Brief** (`--creative-brief`): LLM brainstorms personas/topic/scenario, varies per example
2. **Manual** (`--topic` + `--persona1`/`--persona2` or `--persona`/`--personas`): User specifies all params
3. **Fixed Persona + Variation** (`--enable-variation` + `--fixed-persona*` + `--initial-*`): Fixed personas, varied topics
4. **Random Pairings** (`--random-pairings`): Random character groups from YAML pools, optional `--group-size N`
5. **Continue** (`--continue-from`): Extend existing conversation from JSONL file

### Key Details

- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Default max tokens: 4096
- N-speaker support: `--persona` (repeatable) or `--personas` YAML file
- Role mapping: `--train-speaker "Name"` (that speaker = gpt, rest = human) or `--role-mapping "Name1=human,Name2=gpt"`
- Personas are `list[tuple[str, str]]` (name, description) throughout the pipeline
- Parser handles both one-arg-per-line and inline LLM output formats
- Variation regex uses non-anchored pattern (no `re.DOTALL`) to match `--key "value"` anywhere in text

### Output Format

JSONL with fields: conversation_id, turn_number, role, speaker_name, topic, scenario, style, include_points, content. Each turn carries its own speaker_name. Output files are gitignored (`*.jsonl`).
