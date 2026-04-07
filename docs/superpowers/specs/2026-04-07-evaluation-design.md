# Evaluation Module — Design Spec

## Overview

Add a standalone `evaluate.py` script that computes intrinsic quality metrics on generated JSONL datasets. CPU-only, no GPU required. Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for embedding-based metrics.

## Usage

```bash
python evaluate.py conversations.jsonl
python evaluate.py conversations.jsonl --format json
python evaluate.py conversations.jsonl --verbose
```

## Metrics

### Dataset Summary
- Conversation count
- Total turns
- Avg turns per conversation
- Speaker count
- Speaker turn distribution (% of turns per speaker)

### Diversity
- **Distinct-N** (N=1,2,3) — distinct unigrams/bigrams/trigrams as fraction of total. Standard dialogue evaluation metric from Li et al. (2016). Higher = more lexically diverse.
- **Topic diversity** — mean pairwise cosine distance between conversation topic embeddings. 0 = all identical topics, 1 = completely unrelated.
- **Vocabulary richness** — type-token ratio (unique tokens / total tokens).

### Coherence
- **Turn-to-turn similarity** — mean cosine similarity between consecutive turn embeddings within each conversation. Target range: 0.3-0.6. Very low = incoherent, very high = repetitive.
- **Self-repetition rate** — fraction of turns with cosine similarity > 0.9 to any earlier turn in the same conversation.

### Speaker Quality
- **Speaker distinctiveness** — mean pairwise cosine distance between per-speaker embedding centroids. Higher = speakers have more distinct voices/vocabularies.

## Output Formats

### Human-readable (default)

```
=== CDG Evaluation Report ===

Dataset: conversations.jsonl
Conversations: 100 | Turns: 1,247 | Avg turns: 12.5

Speakers (3):
  Iron Man:          34.2% of turns
  Captain America:   33.1% of turns
  Thor:              32.7% of turns

Diversity:
  Distinct-1: 0.42 | Distinct-2: 0.81 | Distinct-3: 0.91
  Topic diversity: 0.72 (0=identical, 1=unrelated)
  Vocabulary richness (TTR): 0.68

Coherence:
  Turn-to-turn similarity: 0.47 (target: 0.3-0.6)
  Self-repetition rate: 2.1%

Speaker Distinctiveness:
  Avg pairwise distance: 0.38 (higher = more distinct voices)
```

### JSON (`--format json`)

Same data as a flat JSON object for programmatic use.

### Verbose (`--verbose`)

Adds per-conversation breakdown: turn count, speakers, topic, coherence score.

## Module Structure

### New files
- `conversation_dataset_generator/evaluation.py` — all metric computation
- `tests/test_evaluation.py` — unit tests
- `evaluate.py` — thin entry point

### evaluation.py Functions

- `load_jsonl(path)` — load JSONL into list of conversation dicts (grouped by conversation_id)
- `compute_dataset_summary(conversations)` — counts, averages, speaker distribution
- `compute_distinct_n(conversations, n)` — distinct n-gram ratio
- `compute_topic_diversity(conversations, model)` — embedding-based topic distance
- `compute_vocabulary_richness(conversations)` — type-token ratio
- `compute_turn_coherence(conversations, model)` — sequential turn similarity
- `compute_self_repetition(conversations, model)` — near-duplicate turn detection
- `compute_speaker_distinctiveness(conversations, model)` — per-speaker embedding centroids
- `run_evaluation(path, model_name)` — orchestrates all metrics, returns results dict
- `format_report(results)` — human-readable report string
- `format_json(results)` — JSON output

### Embedding Model

Uses `sentence-transformers/all-MiniLM-L6-v2`:
- 22M parameters, runs fast on CPU
- Already cached in user's HF hub cache
- Loaded once, passed to all embedding-based functions

## Dependencies

Add to `requirements.txt`:
```
sentence-transformers
```

## Testing Strategy

Pure unit tests with small synthetic data — no real embeddings needed for most tests:
- `test_load_jsonl` — verify grouping by conversation_id
- `test_compute_dataset_summary` — counts and distributions
- `test_compute_distinct_n` — known inputs with calculable outputs
- `test_compute_vocabulary_richness` — known TTR
- Mock embedding model for embedding-based tests (return fixed vectors)

## README Section

Add "Evaluation" section to README after "Role Mapping for Training":
- What it does
- Example command + sample output
- Brief metric descriptions
- Note that benchmark comparisons will be added after evaluation runs on real datasets

## Backward Compatibility

No changes to existing modules. New files only.
