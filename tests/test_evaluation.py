import pytest
import json
import numpy as np
from unittest.mock import MagicMock
from conversation_dataset_generator.evaluation import (
    load_jsonl,
    compute_dataset_summary,
    compute_distinct_n,
    compute_vocabulary_richness,
    compute_topic_diversity,
    compute_turn_coherence,
    compute_self_repetition,
    compute_speaker_distinctiveness,
    run_evaluation,
    format_report,
    format_json,
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


def make_mock_model():
    model = MagicMock()
    def encode_side_effect(texts, **kwargs):
        results = []
        for t in texts:
            np.random.seed(hash(t) % 2**31)
            results.append(np.random.randn(384).astype(np.float32))
        return np.array(results)
    model.encode.side_effect = encode_side_effect
    return model


class TestComputeTopicDiversity:
    def test_identical_topics_low_diversity(self):
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        convs = [{"topic": "Weather", "turns": []}, {"topic": "Weather", "turns": []}]
        score = compute_topic_diversity(convs, model)
        assert score < 0.01

    def test_different_topics_higher_diversity(self):
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        convs = [{"topic": "Weather", "turns": []}, {"topic": "Quantum", "turns": []}]
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


class TestRunEvaluation:
    def test_returns_all_pure_metrics(self, sample_jsonl):
        results = run_evaluation(sample_jsonl, model_name=None)
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

    def test_no_embedding_metrics_when_no_model(self, sample_jsonl):
        results = run_evaluation(sample_jsonl, model_name=None)
        assert "topic_diversity" not in results


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

    def test_works_without_embedding_metrics(self):
        results = {
            "path": "test.jsonl",
            "num_conversations": 5, "total_turns": 20, "avg_turns": 4.0,
            "num_speakers": 2, "speaker_distribution": {"A": 0.5, "B": 0.5},
            "distinct_1": 0.5, "distinct_2": 0.8, "distinct_3": 0.9,
            "vocabulary_richness": 0.7,
        }
        report = format_report(results)
        assert "CDG Evaluation Report" in report
        assert "Coherence" not in report  # No embedding metrics


class TestFormatJson:
    def test_valid_json(self):
        import json as json_mod
        results = {"path": "test.jsonl", "num_conversations": 5, "distinct_1": 0.42}
        output = format_json(results)
        parsed = json_mod.loads(output)
        assert parsed["num_conversations"] == 5


class TestComputeVendiScore:
    """Vendi Score: effective number of distinct conversations.

    VS = exp(H(eigenvalues / n)) where eigenvalues come from the L2-normalized
    cosine-similarity Gram matrix. Range is [1, n]: 1 means all items are
    identical (one effective example), n means all items are mutually
    orthogonal (n effective examples).
    """

    def test_identical_items_give_score_one(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        identical_emb = np.array([1.0, 0.0, 0.0])
        model = MagicMock()
        model.encode.return_value = np.array([identical_emb, identical_emb, identical_emb])
        convs = [
            {"turns": [{"content": "x"}]},
            {"turns": [{"content": "x"}]},
            {"turns": [{"content": "x"}]},
        ]
        score = compute_vendi_score(convs, model)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_items_give_score_n(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        model = MagicMock()
        model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        convs = [{"turns": [{"content": f"x{i}"}]} for i in range(3)]
        score = compute_vendi_score(convs, model)
        assert score == pytest.approx(3.0, abs=1e-4)

    def test_returns_one_for_single_conversation(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        model = MagicMock()
        convs = [{"turns": [{"content": "only"}]}]
        score = compute_vendi_score(convs, model)
        assert score == pytest.approx(1.0)

    def test_returns_zero_for_empty_input(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        model = MagicMock()
        score = compute_vendi_score([], model)
        assert score == 0.0

    def test_two_identical_two_orthogonal_is_between_one_and_n(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        model = MagicMock()
        model.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # duplicate of first
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        convs = [{"turns": [{"content": f"x{i}"}]} for i in range(4)]
        score = compute_vendi_score(convs, model)
        # 4 items but one is a duplicate — effective diversity is between 3 and 4
        assert 2.5 < score < 3.5

    def test_concatenates_all_turns_for_embedding(self):
        from conversation_dataset_generator.evaluation import compute_vendi_score
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        convs = [
            {"turns": [{"content": "hello"}, {"content": "world"}]},
            {"turns": [{"content": "foo"}, {"content": "bar"}]},
        ]
        compute_vendi_score(convs, model)
        # The model should have been called with concatenated turn texts
        call_arg = model.encode.call_args[0][0]
        assert any("hello" in t and "world" in t for t in call_arg)
        assert any("foo" in t and "bar" in t for t in call_arg)


class TestIsNearDuplicate:
    """Cosine-similarity duplicate detection helper used by the generation loop."""

    def test_returns_false_when_no_priors(self):
        from conversation_dataset_generator.evaluation import is_near_duplicate
        new = np.array([1.0, 0.0, 0.0])
        assert is_near_duplicate(new, [], threshold=0.95) is False

    def test_returns_true_when_above_threshold(self):
        from conversation_dataset_generator.evaluation import is_near_duplicate
        new = np.array([1.0, 0.0, 0.0])
        priors = [np.array([0.99, 0.01, 0.0])]
        assert is_near_duplicate(new, priors, threshold=0.95) is True

    def test_returns_false_when_below_threshold(self):
        from conversation_dataset_generator.evaluation import is_near_duplicate
        new = np.array([1.0, 0.0, 0.0])
        priors = [np.array([0.5, 0.5, 0.0])]
        assert is_near_duplicate(new, priors, threshold=0.95) is False

    def test_uses_max_similarity_across_priors(self):
        from conversation_dataset_generator.evaluation import is_near_duplicate
        new = np.array([1.0, 0.0, 0.0])
        priors = [
            np.array([0.0, 1.0, 0.0]),  # orthogonal — sim 0
            np.array([0.99, 0.01, 0.0]),  # near duplicate — sim ~1
        ]
        assert is_near_duplicate(new, priors, threshold=0.95) is True

    def test_handles_zero_vectors(self):
        from conversation_dataset_generator.evaluation import is_near_duplicate
        new = np.array([0.0, 0.0, 0.0])
        priors = [np.array([1.0, 0.0, 0.0])]
        assert is_near_duplicate(new, priors, threshold=0.95) is False
