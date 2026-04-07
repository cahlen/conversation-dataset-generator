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
