"""Intrinsic quality metrics for generated conversation datasets."""

import json
import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
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
    tokens = text.lower().split()
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_distinct_n(conversations: list[dict], n: int) -> float:
    all_ngrams = []
    for c in conversations:
        for t in c["turns"]:
            all_ngrams.extend(_get_ngrams(t["content"], n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_vocabulary_richness(conversations: list[dict]) -> float:
    all_tokens = []
    for c in conversations:
        for t in c["turns"]:
            all_tokens.extend(t["content"].lower().split())
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
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
    topics = [c["topic"] for c in conversations if c.get("topic")]
    if len(topics) < 2:
        return 0.0
    embeddings = model.encode(topics, show_progress_bar=False)
    return _mean_pairwise_cosine_distance(embeddings)


def compute_turn_coherence(conversations: list[dict], model) -> float:
    similarities = []
    for c in conversations:
        turns = c["turns"]
        if len(turns) < 2:
            continue
        texts = [t["content"] for t in turns]
        embeddings = model.encode(texts, show_progress_bar=False)
        for i in range(len(embeddings) - 1):
            sim = max(0.0, _cosine_similarity(embeddings[i], embeddings[i + 1]))
            similarities.append(sim)
    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def compute_self_repetition(conversations: list[dict], model, threshold: float = 0.9) -> float:
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


def compute_vendi_score(conversations: list[dict], model) -> float:
    """Effective number of distinct conversations.

    VS = exp(-Σ p_i log p_i) where p_i are eigenvalues of the L2-normalized
    cosine-similarity Gram matrix divided by N. Range is [1, N]: 1 = all
    items identical, N = all items mutually orthogonal. Reference:
    Friedman & Dieng, "The Vendi Score" (2023). Embeds each conversation
    by concatenating all of its turn contents.
    """
    if not conversations:
        return 0.0
    if len(conversations) < 2:
        return 1.0

    texts = [
        " ".join(t["content"] for t in c["turns"])
        for c in conversations
    ]
    embeddings = np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    n = len(normalized)
    K = normalized @ normalized.T
    K_norm = K / n

    eigenvalues = np.linalg.eigvalsh(K_norm)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    nonzero = eigenvalues[eigenvalues > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(np.exp(entropy))


def is_near_duplicate(
    new_embedding: np.ndarray,
    prior_embeddings: list,
    threshold: float = 0.95,
) -> bool:
    """True if max cosine similarity between new and any prior exceeds threshold."""
    if not prior_embeddings:
        return False
    for prior in prior_embeddings:
        if _cosine_similarity(new_embedding, prior) > threshold:
            return True
    return False


def compute_speaker_distinctiveness(conversations: list[dict], model) -> float:
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


def run_evaluation(
    path: str,
    model_name: str | None = "sentence-transformers/all-MiniLM-L6-v2",
    model=None,
) -> dict:
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
        results["vendi_score"] = compute_vendi_score(conversations, model)

    return results


def format_report(results: dict) -> str:
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

    dist = results.get("speaker_distribution", {})
    lines.append(f"Speakers ({results['num_speakers']}):")
    for name, frac in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"  {name:25s} {frac:.1%} of turns")
    lines.append("")

    lines.append("Diversity:")
    lines.append(
        f"  Distinct-1: {results['distinct_1']:.2f} | "
        f"Distinct-2: {results['distinct_2']:.2f} | "
        f"Distinct-3: {results['distinct_3']:.2f}"
    )
    if "topic_diversity" in results:
        lines.append(f"  Topic diversity: {results['topic_diversity']:.2f} (0=identical, 1=unrelated)")
    lines.append(f"  Vocabulary richness (TTR): {results['vocabulary_richness']:.2f}")
    if "vendi_score" in results:
        n = results["num_conversations"]
        lines.append(
            f"  Vendi Score: {results['vendi_score']:.2f} / {n} "
            f"(effective distinct conversations; closer to N = more diverse)"
        )
    lines.append("")

    if "turn_coherence" in results:
        lines.append("Coherence:")
        lines.append(f"  Turn-to-turn similarity: {results['turn_coherence']:.2f} (target: 0.3-0.6)")
        lines.append(f"  Self-repetition rate: {results['self_repetition_rate']:.1%}")
        lines.append("")

    if "speaker_distinctiveness" in results:
        lines.append("Speaker Distinctiveness:")
        lines.append(f"  Avg pairwise distance: {results['speaker_distinctiveness']:.2f} (higher = more distinct voices)")
        lines.append("")

    return "\n".join(lines)


def format_json(results: dict) -> str:
    return json.dumps(results, indent=2)
