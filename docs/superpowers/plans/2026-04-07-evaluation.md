# Evaluation Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `evaluate.py` that computes intrinsic quality metrics (diversity, coherence, speaker distinctiveness) on generated JSONL datasets, CPU-only.

**Architecture:** One new module `conversation_dataset_generator/evaluation.py` with pure metric functions + embedding-based metrics via sentence-transformers. Thin `evaluate.py` entry point. README section documenting usage and metrics.

**Tech Stack:** Python 3.10+, sentence-transformers, numpy, pytest

---

## File Map

### New files:
- `conversation_dataset_generator/evaluation.py` — all metric computation
- `tests/test_evaluation.py` — unit tests
- `evaluate.py` — thin entry point

### Files to modify:
- `requirements.txt` — add sentence-transformers
- `README.md` — add Evaluation section
- `CLAUDE.md` — mention evaluate.py

---

### Task 1: Dependencies + Pure Metric Functions

**Files:**
- Modify: `requirements.txt`
- Create: `conversation_dataset_generator/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Add sentence-transformers to requirements.txt**

Add `sentence-transformers` to the end of `requirements.txt`.

- [ ] **Step 2: Install it**

Run: `python -m pip install sentence-transformers`

- [ ] **Step 3: Write failing tests for pure metrics (no embeddings)**

```python
# tests/test_evaluation.py
import pytest
from conversation_dataset_generator.evaluation import (
    load_jsonl,
    compute_dataset_summary,
    compute_distinct_n,
    compute_vocabulary_richness,
)


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a sample JSONL file with 2 conversations."""
    import json
    path = tmp_path / "test.jsonl"
    rows = [
        {"conversation_id": 0, "turn_number": 0, "role": "human", "speaker_name": "Alice", "topic": "Weather", "scenario": "Park", "style": "Casual", "include_points": "", "content": "Nice day today"},
        {"conversation_id": 0, "turn_number": 1, "role": "gpt", "speaker_name": "Bob", "topic": "Weather", "scenario": "Park", "style": "Casual", "include_points": "", "content": "Yes it is beautiful outside"},
        {"conversation_id": 0, "turn_number": 2, "role": "human", "speaker_name": "Alice", "topic": "Weather", "scenario": "Park", "style": "Casual", "include_points": "", "content": "Should we go for a walk"},
        {"conversation_id": 1, "turn_number": 0, "role": "human", "speaker_name": "Alice", "topic": "Food", "scenario": "Kitchen", "style": "Casual", "include_points": "", "content": "What should we cook tonight"},
        {"conversation_id": 1, "turn_number": 1, "role": "gpt", "speaker_name": "Bob", "topic": "Food", "scenario": "Kitchen", "style": "Casual", "include_points": "", "content": "How about pasta with fresh tomatoes"},
    ]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


class TestLoadJsonl:
    def test_groups_by_conversation(self, sample_jsonl):
        convs = load_jsonl(sample_jsonl)
        assert len(convs) == 2
        assert len(convs[0]["turns"]) == 3
        assert len(convs[1]["turns"]) == 2

    def test_preserves_metadata(self, sample_jsonl):
        convs = load_jsonl(sample_jsonl)
        assert convs[0]["topic"] == "Weather"
        assert convs[1]["topic"] == "Food"
        assert convs[0]["speakers"] == ["Alice", "Bob"]


class TestComputeDatasetSummary:
    def test_counts(self, sample_jsonl):
        convs = load_jsonl(sample_jsonl)
        summary = compute_dataset_summary(convs)
        assert summary["num_conversations"] == 2
        assert summary["total_turns"] == 5
        assert summary["avg_turns"] == 2.5
        assert summary["num_speakers"] == 2
        assert "Alice" in summary["speaker_distribution"]
        assert "Bob" in summary["speaker_distribution"]

    def test_speaker_distribution_sums_to_one(self, sample_jsonl):
        convs = load_jsonl(sample_jsonl)
        summary = compute_dataset_summary(convs)
        total = sum(summary["speaker_distribution"].values())
        assert abs(total - 1.0) < 0.01


class TestComputeDistinctN:
    def test_distinct_1(self):
        # "the cat sat on the mat" has 5 unique unigrams out of 6 total
        convs = [{"turns": [{"content": "the cat sat on the mat"}]}]
        score = compute_distinct_n(convs, n=1)
        assert abs(score - 5/6) < 0.01

    def test_distinct_2(self):
        # "a b c a b" bigrams: (a,b), (b,c), (c,a), (a,b) = 3 unique / 4 total
        convs = [{"turns": [{"content": "a b c a b"}]}]
        score = compute_distinct_n(convs, n=2)
        assert abs(score - 3/4) < 0.01

    def test_empty_returns_zero(self):
        score = compute_distinct_n([], n=1)
        assert score == 0.0


class TestComputeVocabularyRichness:
    def test_all_unique(self):
        convs = [{"turns": [{"content": "alpha beta gamma delta"}]}]
        ttr = compute_vocabulary_richness(convs)
        assert ttr == 1.0

    def test_all_same(self):
        convs = [{"turns": [{"content": "the the the the"}]}]
        ttr = compute_vocabulary_richness(convs)
        assert ttr == 0.25

    def test_empty_returns_zero(self):
        ttr = compute_vocabulary_richness([])
        assert ttr == 0.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 5: Implement pure metric functions**

```python
# conversation_dataset_generator/evaluation.py
"""Intrinsic quality metrics for generated conversation datasets."""

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL and group rows into conversations.

    Returns list of dicts, each with:
        conversation_id, topic, scenario, style, speakers, turns
    Each turn has: from, value/content, speaker_name
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

    conversations = []
    for cid in sorted(rows_by_conv.keys()):
        rows = sorted(rows_by_conv[cid], key=lambda r: r["turn_number"])
        first = rows[0]
        speakers = list(dict.fromkeys(r["speaker_name"] for r in rows))
        turns = [{"content": r["content"], "speaker_name": r["speaker_name"], "from": r["role"]} for r in rows]
        conversations.append({
            "conversation_id": cid,
            "topic": first.get("topic", ""),
            "scenario": first.get("scenario", ""),
            "style": first.get("style", ""),
            "speakers": speakers,
            "turns": turns,
        })

    return conversations


def compute_dataset_summary(conversations: list[dict]) -> dict:
    """Compute basic dataset statistics."""
    if not conversations:
        return {"num_conversations": 0, "total_turns": 0, "avg_turns": 0.0,
                "num_speakers": 0, "speaker_distribution": {}}

    total_turns = sum(len(c["turns"]) for c in conversations)
    all_speakers = Counter()
    for c in conversations:
        for t in c["turns"]:
            all_speakers[t["speaker_name"]] += 1

    distribution = {name: count / total_turns for name, count in all_speakers.items()}

    return {
        "num_conversations": len(conversations),
        "total_turns": total_turns,
        "avg_turns": total_turns / len(conversations),
        "num_speakers": len(all_speakers),
        "speaker_distribution": distribution,
    }


def _get_ngrams(text: str, n: int) -> list[tuple]:
    """Extract n-grams from text."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_distinct_n(conversations: list[dict], n: int) -> float:
    """Compute distinct-N: unique n-grams / total n-grams."""
    all_ngrams = []
    for c in conversations:
        for t in c["turns"]:
            all_ngrams.extend(_get_ngrams(t["content"], n))

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def compute_vocabulary_richness(conversations: list[dict]) -> float:
    """Compute type-token ratio (unique tokens / total tokens)."""
    all_tokens = []
    for c in conversations:
        for t in c["turns"]:
            all_tokens.extend(t["content"].lower().split())

    if not all_tokens:
        return 0.0

    return len(set(all_tokens)) / len(all_tokens)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 10 tests PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt conversation_dataset_generator/evaluation.py tests/test_evaluation.py
git commit -m "add evaluation module with pure metric functions (distinct-N, TTR, summary)"
```

---

### Task 2: Embedding-Based Metrics

**Files:**
- Modify: `conversation_dataset_generator/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests with mocked embeddings**

Add to `tests/test_evaluation.py`:

```python
import numpy as np
from unittest.mock import MagicMock
from conversation_dataset_generator.evaluation import (
    compute_topic_diversity,
    compute_turn_coherence,
    compute_self_repetition,
    compute_speaker_distinctiveness,
)


def make_mock_model():
    """Mock sentence-transformers model that returns fixed embeddings."""
    model = MagicMock()
    call_count = [0]
    def encode_side_effect(texts, **kwargs):
        results = []
        for t in texts:
            # Deterministic but different per text
            np.random.seed(hash(t) % 2**31)
            results.append(np.random.randn(384).astype(np.float32))
        return np.array(results)
    model.encode.side_effect = encode_side_effect
    return model


class TestComputeTopicDiversity:
    def test_identical_topics_low_diversity(self):
        model = MagicMock()
        # Return identical embeddings for identical topics
        model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        convs = [
            {"topic": "Weather", "turns": []},
            {"topic": "Weather", "turns": []},
        ]
        score = compute_topic_diversity(convs, model)
        assert score < 0.01  # Nearly zero distance

    def test_different_topics_higher_diversity(self):
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        convs = [
            {"topic": "Weather", "turns": []},
            {"topic": "Quantum physics", "turns": []},
        ]
        score = compute_topic_diversity(convs, model)
        assert score > 0.5

    def test_single_conversation_returns_zero(self):
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        convs = [{"topic": "Weather", "turns": []}]
        score = compute_topic_diversity(convs, model)
        assert score == 0.0


class TestComputeTurnCoherence:
    def test_returns_float(self):
        model = make_mock_model()
        convs = [{"turns": [
            {"content": "Hello there", "speaker_name": "A"},
            {"content": "Hi how are you", "speaker_name": "B"},
            {"content": "I am fine thanks", "speaker_name": "A"},
        ]}]
        score = compute_turn_coherence(convs, model)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_turn_returns_zero(self):
        model = make_mock_model()
        convs = [{"turns": [{"content": "Hello", "speaker_name": "A"}]}]
        score = compute_turn_coherence(convs, model)
        assert score == 0.0


class TestComputeSelfRepetition:
    def test_no_repetition(self):
        model = make_mock_model()
        convs = [{"turns": [
            {"content": "The weather is nice today", "speaker_name": "A"},
            {"content": "I prefer rainy days personally", "speaker_name": "B"},
            {"content": "Quantum physics is fascinating", "speaker_name": "A"},
        ]}]
        rate = compute_self_repetition(convs, model)
        assert isinstance(rate, float)

    def test_empty_returns_zero(self):
        rate = compute_self_repetition([], MagicMock())
        assert rate == 0.0


class TestComputeSpeakerDistinctiveness:
    def test_returns_float(self):
        model = make_mock_model()
        convs = [{"turns": [
            {"content": "Verily I say unto thee", "speaker_name": "Thor"},
            {"content": "JARVIS run diagnostics", "speaker_name": "Iron Man"},
            {"content": "We need a plan soldier", "speaker_name": "Cap"},
        ]}]
        score = compute_speaker_distinctiveness(convs, model)
        assert isinstance(score, float)

    def test_single_speaker_returns_zero(self):
        model = make_mock_model()
        convs = [{"turns": [
            {"content": "Hello", "speaker_name": "A"},
            {"content": "World", "speaker_name": "A"},
        ]}]
        score = compute_speaker_distinctiveness(convs, model)
        assert score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation.py::TestComputeTopicDiversity -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement embedding-based metrics**

Add to `conversation_dataset_generator/evaluation.py`:

```python
import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    """Compute mean pairwise cosine distance (1 - similarity) between embeddings."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - _cosine_similarity(embeddings[i], embeddings[j])
            count += 1
    return total / count


def compute_topic_diversity(conversations: list[dict], model) -> float:
    """Mean pairwise cosine distance between topic embeddings. 0=identical, 1=unrelated."""
    topics = [c["topic"] for c in conversations if c.get("topic")]
    if len(topics) < 2:
        return 0.0
    embeddings = model.encode(topics, show_progress_bar=False)
    return _mean_pairwise_cosine_distance(embeddings)


def compute_turn_coherence(conversations: list[dict], model) -> float:
    """Mean cosine similarity between consecutive turns. Target: 0.3-0.6."""
    similarities = []
    for c in conversations:
        turns = c["turns"]
        if len(turns) < 2:
            continue
        texts = [t["content"] for t in turns]
        embeddings = model.encode(texts, show_progress_bar=False)
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def compute_self_repetition(conversations: list[dict], model, threshold: float = 0.9) -> float:
    """Fraction of turns with cosine similarity > threshold to any earlier turn."""
    if not conversations:
        return 0.0
    total_turns = 0
    repeated_turns = 0
    for c in conversations:
        turns = c["turns"]
        if len(turns) < 2:
            total_turns += len(turns)
            continue
        texts = [t["content"] for t in turns]
        embeddings = model.encode(texts, show_progress_bar=False)
        for i in range(len(embeddings)):
            total_turns += 1
            for j in range(i):
                if _cosine_similarity(embeddings[i], embeddings[j]) > threshold:
                    repeated_turns += 1
                    break

    if total_turns == 0:
        return 0.0
    return repeated_turns / total_turns


def compute_speaker_distinctiveness(conversations: list[dict], model) -> float:
    """Mean pairwise cosine distance between per-speaker embedding centroids."""
    speaker_texts = {}
    for c in conversations:
        for t in c["turns"]:
            name = t["speaker_name"]
            if name not in speaker_texts:
                speaker_texts[name] = []
            speaker_texts[name].append(t["content"])

    if len(speaker_texts) < 2:
        return 0.0

    centroids = []
    for name, texts in speaker_texts.items():
        embeddings = model.encode(texts, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)
        centroids.append(centroid)

    return _mean_pairwise_cosine_distance(np.array(centroids))
```

- [ ] **Step 4: Run all evaluation tests**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 20 tests PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/evaluation.py tests/test_evaluation.py
git commit -m "add embedding-based evaluation metrics (topic diversity, coherence, speaker distinctiveness)"
```

---

### Task 3: Report Formatting + Entry Point

**Files:**
- Modify: `conversation_dataset_generator/evaluation.py`
- Create: `evaluate.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for run_evaluation and format_report**

Add to `tests/test_evaluation.py`:

```python
from conversation_dataset_generator.evaluation import run_evaluation, format_report


class TestRunEvaluation:
    def test_returns_all_metrics(self, sample_jsonl):
        results = run_evaluation(sample_jsonl, model_name=None)  # None = skip embedding metrics
        assert "num_conversations" in results
        assert "total_turns" in results
        assert "distinct_1" in results
        assert "distinct_2" in results
        assert "distinct_3" in results
        assert "vocabulary_richness" in results

    def test_with_mock_model(self, sample_jsonl):
        results = run_evaluation(sample_jsonl, model_name=None, model=make_mock_model())
        assert "topic_diversity" in results
        assert "turn_coherence" in results
        assert "self_repetition_rate" in results
        assert "speaker_distinctiveness" in results


class TestFormatReport:
    def test_contains_section_headers(self):
        results = {
            "path": "test.jsonl",
            "num_conversations": 10, "total_turns": 100, "avg_turns": 10.0,
            "num_speakers": 2, "speaker_distribution": {"Alice": 0.5, "Bob": 0.5},
            "distinct_1": 0.42, "distinct_2": 0.81, "distinct_3": 0.91,
            "vocabulary_richness": 0.68,
            "topic_diversity": 0.72,
            "turn_coherence": 0.47,
            "self_repetition_rate": 0.02,
            "speaker_distinctiveness": 0.38,
        }
        report = format_report(results)
        assert "CDG Evaluation Report" in report
        assert "Diversity" in report
        assert "Coherence" in report
        assert "Speaker" in report
        assert "0.42" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluation.py::TestRunEvaluation tests/test_evaluation.py::TestFormatReport -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement run_evaluation and format_report**

Add to `conversation_dataset_generator/evaluation.py`:

```python
def run_evaluation(
    path: str,
    model_name: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    model=None,
) -> dict:
    """Run all evaluation metrics on a JSONL file.

    Args:
        path: Path to JSONL file.
        model_name: Sentence-transformers model to load. None to skip embedding metrics.
        model: Pre-loaded model (for testing). Overrides model_name.

    Returns:
        Dict with all metric results.
    """
    conversations = load_jsonl(path)
    summary = compute_dataset_summary(conversations)

    results = {
        "path": path,
        **summary,
        "distinct_1": compute_distinct_n(conversations, 1),
        "distinct_2": compute_distinct_n(conversations, 2),
        "distinct_3": compute_distinct_n(conversations, 3),
        "vocabulary_richness": compute_vocabulary_richness(conversations),
    }

    # Embedding-based metrics (optional)
    if model is None and model_name is not None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", model_name)
            model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. Skipping embedding metrics.")
        except Exception as e:
            logger.warning("Failed to load embedding model: %s", e)

    if model is not None:
        results["topic_diversity"] = compute_topic_diversity(conversations, model)
        results["turn_coherence"] = compute_turn_coherence(conversations, model)
        results["self_repetition_rate"] = compute_self_repetition(conversations, model)
        results["speaker_distinctiveness"] = compute_speaker_distinctiveness(conversations, model)

    return results


def format_report(results: dict) -> str:
    """Format evaluation results as a human-readable report."""
    lines = []
    lines.append("=== CDG Evaluation Report ===")
    lines.append("")
    lines.append(f"Dataset: {results.get('path', 'unknown')}")
    lines.append(
        f"Conversations: {results['num_conversations']} | "
        f"Turns: {results['total_turns']} | "
        f"Avg turns: {results['avg_turns']:.1f}"
    )
    lines.append("")

    # Speakers
    dist = results.get("speaker_distribution", {})
    lines.append(f"Speakers ({results['num_speakers']}):")
    for name, frac in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:25s} {frac:.1%} of turns")
    lines.append("")

    # Diversity
    lines.append("Diversity:")
    lines.append(
        f"  Distinct-1: {results['distinct_1']:.2f} | "
        f"Distinct-2: {results['distinct_2']:.2f} | "
        f"Distinct-3: {results['distinct_3']:.2f}"
    )
    if "topic_diversity" in results:
        lines.append(f"  Topic diversity: {results['topic_diversity']:.2f} (0=identical, 1=unrelated)")
    lines.append(f"  Vocabulary richness (TTR): {results['vocabulary_richness']:.2f}")
    lines.append("")

    # Coherence
    if "turn_coherence" in results:
        lines.append("Coherence:")
        lines.append(f"  Turn-to-turn similarity: {results['turn_coherence']:.2f} (target: 0.3-0.6)")
        lines.append(f"  Self-repetition rate: {results['self_repetition_rate']:.1%}")
        lines.append("")

    # Speaker distinctiveness
    if "speaker_distinctiveness" in results:
        lines.append("Speaker Distinctiveness:")
        lines.append(f"  Avg pairwise distance: {results['speaker_distinctiveness']:.2f} (higher = more distinct voices)")
        lines.append("")

    return "\n".join(lines)


def format_json(results: dict) -> str:
    """Format evaluation results as JSON."""
    return json.dumps(results, indent=2)
```

- [ ] **Step 4: Create evaluate.py entry point**

```python
#!/usr/bin/env python
# coding=utf-8
"""Evaluate generated conversation datasets."""

import argparse
import logging
import sys

from conversation_dataset_generator.evaluation import run_evaluation, format_report, format_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated conversation dataset.")
    parser.add_argument("input", help="Path to JSONL file to evaluate.")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-conversation breakdown.")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding-based metrics (faster, no model download).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_name = None if args.no_embeddings else "sentence-transformers/all-MiniLM-L6-v2"
    results = run_evaluation(args.input, model_name=model_name)

    if args.format == "json":
        print(format_json(results))
    else:
        print(format_report(results))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run all evaluation tests**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 24 tests PASS

- [ ] **Step 6: Verify entry point works**

Run: `python evaluate.py --help`
Expected: Shows help with `input`, `--format`, `--verbose`, `--no-embeddings`

- [ ] **Step 7: Commit**

```bash
git add evaluate.py conversation_dataset_generator/evaluation.py tests/test_evaluation.py
git commit -m "add evaluation entry point with report formatting"
```

---

### Task 4: Update README + CLAUDE.md + Integration Test

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add Evaluation section to README**

Add after the "Role Mapping for Training" section:

```markdown
## Evaluation

Measure the quality of generated datasets with intrinsic metrics:

```bash
python evaluate.py conversations.jsonl
```

Output:
```
=== CDG Evaluation Report ===

Dataset: conversations.jsonl
Conversations: 100 | Turns: 1,247 | Avg turns: 12.5

Speakers (3):
  Iron Man                  34.2% of turns
  Captain America           33.1% of turns
  Thor                      32.7% of turns

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

**Metrics explained:**
- **Distinct-N** — fraction of unique n-grams. Higher = more lexically diverse.
- **Topic diversity** — embedding distance between conversation topics. 0 = all identical, 1 = completely varied.
- **Turn coherence** — how well consecutive turns relate. Sweet spot: 0.3-0.6.
- **Self-repetition** — fraction of near-duplicate turns within conversations.
- **Speaker distinctiveness** — how different each speaker's language is from others.

Options:
```bash
python evaluate.py data.jsonl --format json     # machine-readable
python evaluate.py data.jsonl --no-embeddings   # skip embedding metrics (faster)
```
```

- [ ] **Step 2: Update CLAUDE.md**

Add `evaluate.py` to the Running section. Update test count. Mention `evaluation.py` module in the package table.

- [ ] **Step 3: Generate test data and run evaluate.py on it**

Run:
```bash
python generate.py \
  --persona "Iron Man" "Genius billionaire" \
  --persona "Captain America" "Principled leader" \
  --persona "Thor" "Boisterous god" \
  --topic "mission planning" --scenario "war room" --style "debate" \
  --num-examples 5 --output-file test_eval_data.jsonl \
  --model-id unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit
```

Then:
```bash
python evaluate.py test_eval_data.jsonl
```

Expected: Full evaluation report with all metrics.

Clean up: `rm test_eval_data.jsonl`

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (121 existing + ~24 new = ~145)

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "document evaluation module in README and CLAUDE.md"
```
