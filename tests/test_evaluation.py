import pytest
import json
from conversation_dataset_generator.evaluation import (
    load_jsonl,
    compute_dataset_summary,
    compute_distinct_n,
    compute_vocabulary_richness,
)


@pytest.fixture
def sample_jsonl(tmp_path):
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
        convs = [{"turns": [{"content": "the cat sat on the mat"}]}]
        score = compute_distinct_n(convs, n=1)
        assert abs(score - 5/6) < 0.01

    def test_distinct_2(self):
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
