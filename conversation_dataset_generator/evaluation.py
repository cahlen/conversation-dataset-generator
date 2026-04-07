"""Intrinsic quality metrics for generated conversation datasets."""

import json
import logging
from collections import Counter

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
