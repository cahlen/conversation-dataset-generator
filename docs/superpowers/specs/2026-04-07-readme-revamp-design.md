# README Revamp + Docker Support — Design Spec

## Overview

Rewrite the 952-line README to ~200-250 lines of clean, practical markdown. Add Dockerfile with multi-CUDA support, docker-compose.yml, and .dockerignore. Target audience: developers generating training data, with a brief section for contributors.

## README Structure

Target: ~200-250 lines. No HTML cruft, no badge-per-dependency, no "back to top" links, no collapsible sections.

```
# Conversation Dataset Generator
  One-liner + brief paragraph (what it does, who it's for)

## Quick Start (pip)
  3 commands: venv, pip install, run generate.py

## Quick Start (Docker)
  2 commands: docker build, docker run --gpus all

## Modes
  All 5 modes, each with:
  - 3-5 line description
  - One example command
  Modes: Manual, Creative Brief, Fixed Persona + Variation,
         Random Pairings, Random Pairings + Variation

## Argument Reference
  Clean table with columns: flag, description, default
  Grouped by: mode selection, manual mode, fixed persona, random pairings,
              brief context, general

## Output Format
  One JSON sample row + field descriptions

## Web Search Enrichment
  Brief explanation of --persona1-search-term / --persona2-search-term
  One example command

## For Contributors
  Package structure table (module → responsibility)
  How to run tests (3 commands)
  Link to design docs

## License
  One line + link to LICENSE file
```

## What Gets Cut

- All HTML (`<p align="right">`, `<details>`, `<summary>`, `<br>`, `<div>`)
- Shield badges for individual dependencies
- LoRA training tutorial (~100 lines of inline Python)
- Collapsible example sections (~400 lines)
- Pandas references (dependency removed)
- Image search / delete-repo mentions (features removed)
- Stale roadmap
- "Back to top" links
- Acknowledgments section (excessive for a tool README)
- "Built With" badge wall

## What Gets Updated

- Default model: `Qwen/Qwen2.5-7B-Instruct` (was Meta-Llama-3-8B-Instruct)
- Python version: 3.10+ (was 3.8+)
- All 5 modes documented (was 2)
- Package structure (was monolith)
- Dependencies list (removed pandas, peft, trl)
- Test commands included
- Docker usage documented

## What Gets Added

- Docker quick start section
- `--role-mapping` flag documented
- Contributors section with test commands
- `--max-new-tokens` default documented as 4096

## Dockerfile

Single Dockerfile with `CUDA_VERSION` build arg, defaulting to 12.8.0.

```
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

- Install Python 3.10, pip
- Copy requirements.txt, install deps
- Copy package source
- Entrypoint: python generate.py
```

Build and run:
```bash
# Default CUDA 12.x (broadest compat: 30xx, 40xx, 50xx)
docker build -t cdg .
docker run --gpus all cdg --creative-brief "Sherlock vs Watson"

# CUDA 13.x for RTX 50xx with latest drivers
docker build --build-arg CUDA_VERSION=13.0.0 -t cdg .
```

Output files bind-mounted:
```bash
docker run --gpus all -v $(pwd)/output:/app/output cdg \
  --creative-brief "test" --output-file output/data.jsonl
```

## docker-compose.yml

For convenience — handles `--gpus` flag automatically:

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

Usage: `docker compose run cdg --creative-brief "..." --output-file output/data.jsonl`

## .dockerignore

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
generate_old.py
```

## Files to Create/Modify

- Modify: `README.md` — full rewrite
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
