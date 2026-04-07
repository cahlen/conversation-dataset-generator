# README Revamp + Docker Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the 952-line README to ~200-250 lines of clean markdown, add Docker support with multi-CUDA build arg.

**Architecture:** Four independent files: README.md (rewrite), Dockerfile, docker-compose.yml, .dockerignore. README references Docker usage. No application code changes.

**Tech Stack:** Markdown, Docker, NVIDIA CUDA base images

---

## File Map

### Files to create:
- `Dockerfile` — multi-CUDA build with `CUDA_VERSION` arg
- `docker-compose.yml` — GPU-enabled convenience wrapper
- `.dockerignore` — keep image small

### Files to modify:
- `README.md` — full rewrite

---

### Task 1: Docker Support Files

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Create .dockerignore**

```
.git
__pycache__
*.pyc
*.jsonl
*.log
docs/
tests/
.venv
venv
.cache
.env
generate_old.py
character-config/
examples/
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY conversation_dataset_generator/ conversation_dataset_generator/
COPY generate.py .
COPY character-config/ character-config/

ENTRYPOINT ["python3", "generate.py"]
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
services:
  cdg:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ./output:/app/output
```

- [ ] **Step 4: Verify Dockerfile syntax**

Run: `docker build --check . 2>&1 || echo "docker build --check not available, skipping"`

- [ ] **Step 5: Commit**

```bash
git add .dockerignore Dockerfile docker-compose.yml
git commit -m "add Docker support with multi-CUDA build arg"
```

---

### Task 2: Rewrite README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README to understand what exists**

Read: `README.md` (952 lines)

Note the reference-style link definitions at the bottom (lines 929-952) — we'll keep a minimal set for the license shield only.

- [ ] **Step 2: Replace README.md with the new content**

```markdown
# Conversation Dataset Generator

Generate synthetic conversational datasets in ShareGPT format for LLM fine-tuning. Define personas, topics, and styles — or provide a creative brief and let the LLM figure it out.

[![MIT License](https://img.shields.io/github/license/cahlen/conversation-dataset-generator.svg)](LICENSE)

## Quick Start (pip)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python generate.py \
  --creative-brief "Sherlock Holmes and Watson debate whether AI will replace detectives" \
  --num-examples 5 --output-file conversations.jsonl
```

Requires Python 3.10+ and an NVIDIA GPU with CUDA.

## Quick Start (Docker)

```bash
# Build (default CUDA 12.x — works on 30xx/40xx/50xx)
docker build -t cdg .

# Or build for CUDA 13.x (RTX 50xx with latest drivers)
docker build --build-arg CUDA_VERSION=13.0.0 -t cdg .

# Run
docker run --gpus all -v $(pwd)/output:/app/output cdg \
  --creative-brief "Two scientists argue about time travel" \
  --output-file output/data.jsonl
```

Or with docker compose:

```bash
docker compose run cdg \
  --creative-brief "Two scientists argue about time travel" \
  --output-file output/data.jsonl
```

## Modes

### Manual

Specify everything directly. No variation — every conversation uses the same parameters.

```bash
python generate.py \
  --topic "best pizza toppings" \
  --persona1 "Tony" --persona1-desc "A passionate Italian chef" \
  --persona2 "Dave" --persona2-desc "A pineapple-on-pizza enthusiast" \
  --scenario "kitchen argument" --style "heated but friendly debate" \
  --num-examples 10 --output-file pizza_debate.jsonl
```

### Creative Brief

Provide a high-level brief. The LLM generates personas, topic, scenario, and style, then varies the topic/scenario for each conversation.

```bash
python generate.py \
  --creative-brief "A grumpy cat and an overly enthusiastic golden retriever share a sunbeam" \
  --num-examples 20 --output-file cat_dog.jsonl
```

Optionally enrich personas with web search context:

```bash
python generate.py \
  --creative-brief "Linus Torvalds and Tim Cook debate open source" \
  --persona1-search-term "Linus Torvalds" \
  --persona2-search-term "Tim Cook Apple CEO" \
  --num-examples 10 --output-file tech_debate.jsonl
```

### Fixed Persona + Variation

Fix the personas but let the LLM vary the topic and scenario each time.

```bash
python generate.py \
  --enable-variation \
  --fixed-persona1 "Iron Man" --fixed-persona1-desc "Genius billionaire with rapid-fire wit" \
  --fixed-persona2 "Captain America" --fixed-persona2-desc "Principled, earnest, old-fashioned" \
  --initial-topic "team leadership" --initial-scenario "Avengers HQ" --initial-style "friendly disagreement" \
  --num-examples 50 --output-file avengers.jsonl
```

### Random Pairings

Randomly pair characters from YAML pool files for each conversation.

```bash
python generate.py \
  --random-pairings \
  --character-pool avengers_characters.yaml \
  --persona-desc-pool avengers_descriptions.yaml \
  --initial-topic "planning a party" --initial-scenario "break room" --initial-style "casual banter" \
  --num-examples 100 --output-file avengers_random.jsonl
```

Add `--enable-variation` to also vary topics per conversation.

### Batch Generation

Run multiple generation jobs from a YAML config:

```bash
python batch_generate.py examples/batch_mixed_modes.yaml
```

See `examples/` for sample batch configs.

## Argument Reference

### Mode Selection

| Flag | Description |
|---|---|
| `--creative-brief TEXT` | Creative brief for automatic parameter generation |
| `--enable-variation` | Vary topic/scenario between conversations |
| `--random-pairings` | Random character pairs from pool files |

### Manual Mode

| Flag | Description |
|---|---|
| `--topic TEXT` | Conversation topic |
| `--persona1 TEXT` | First speaker name |
| `--persona1-desc TEXT` | First speaker description |
| `--persona2 TEXT` | Second speaker name |
| `--persona2-desc TEXT` | Second speaker description |
| `--scenario TEXT` | Setting/context |
| `--style TEXT` | Dialogue style/tone |
| `--include-points TEXT` | Comma-separated keywords to include |

### Fixed Persona Variation

| Flag | Description |
|---|---|
| `--fixed-persona1 TEXT` | Fixed first speaker name |
| `--fixed-persona1-desc TEXT` | Fixed first speaker description |
| `--fixed-persona2 TEXT` | Fixed second speaker name |
| `--fixed-persona2-desc TEXT` | Fixed second speaker description |
| `--initial-topic TEXT` | Seed topic for variation |
| `--initial-scenario TEXT` | Seed scenario for variation |
| `--initial-style TEXT` | Seed style for variation |

### Random Pairings

| Flag | Description |
|---|---|
| `--character-pool FILE` | YAML file with character names |
| `--persona-desc-pool FILE` | YAML file with character descriptions |

### Web Search (Creative Brief)

| Flag | Description |
|---|---|
| `--persona1-search-term TEXT` | Web search term for persona 1 context |
| `--persona2-search-term TEXT` | Web search term for persona 2 context |

### General

| Flag | Default | Description |
|---|---|---|
| `--num-examples N` | 3 | Number of conversations to generate |
| `--output-file PATH` | `generated_data.jsonl` | Output file path |
| `--model-id ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model for generation |
| `--max-new-tokens N` | 4096 | Max tokens per generation |
| `--load-in-4bit` | off | Enable 4-bit quantization (requires bitsandbytes) |
| `--upload-to-hub REPO` | — | Upload dataset to HuggingFace Hub |
| `--force-upload` | off | Skip upload confirmation |
| `--role-mapping MAP` | `p1=human,p2=gpt` | Map personas to ShareGPT roles |

## Output Format

Each line in the JSONL output is one conversation turn:

```json
{
  "conversation_id": 0,
  "turn_number": 0,
  "role": "human",
  "speaker_name": "Tony",
  "topic": "best pizza toppings",
  "scenario": "kitchen argument",
  "style": "heated but friendly debate",
  "include_points": "",
  "content": "So, you're telling me pineapple on pizza is the ultimate topping?"
}
```

## For Contributors

### Package Structure

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing, mode detection, orchestration |
| `models.py` | Model/tokenizer loading, pipeline creation |
| `prompts.py` | System prompts and message builders |
| `generation.py` | LLM call wrappers with retry logic |
| `parsing.py` | Regex parsers for LLM output |
| `output.py` | JSONL writing and dataset card templates |
| `hub.py` | HuggingFace Hub upload |
| `character_pool.py` | YAML pool loading and random pairing |
| `web_search.py` | DuckDuckGo persona context search |

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                    # all 81 tests
pytest tests/test_parsing.py -v     # one module
```

No GPU required for tests — LLM calls are mocked.

## License

MIT. See [LICENSE](LICENSE).
```

- [ ] **Step 3: Verify the README renders correctly**

Run: `wc -l README.md`
Expected: ~200-250 lines

Run: `head -5 README.md`
Expected: Starts with `# Conversation Dataset Generator`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "rewrite README: lean practical docs, Docker support, all 5 modes documented"
```

---

### Task 3: Verify Docker Build

- [ ] **Step 1: Build the Docker image**

Run: `docker build -t cdg . 2>&1 | tail -5`
Expected: Successfully built

- [ ] **Step 2: Verify the image runs**

Run: `docker run --rm cdg --help 2>&1 | head -10`
Expected: Shows generate.py help text with `--creative-brief`, `--model-id`, etc.

- [ ] **Step 3: Commit any fixes if needed**

If the build or help test failed, fix the Dockerfile and recommit.
