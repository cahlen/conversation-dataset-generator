"""Tests for the Gradio webapp's pure handler + helpers.

The Gradio UI binding itself is declarative — we only test the handler and
the formatting helpers it relies on.
"""

import json
import os
from unittest.mock import MagicMock


def _patch_build_backend(monkeypatch, return_value):
    monkeypatch.setattr("webapp.build_backend", lambda **kw: return_value)


def _two_turns(a="Alice", b="Bob"):
    return [
        {"from": "human", "value": "Hello there", "speaker_name": a},
        {"from": "gpt",   "value": "Hi back",     "speaker_name": b},
    ]


class TestGenerateHandlerSingle:
    """N=1 still produces the conversation-as-markdown preview."""

    def test_returns_status_metrics_preview_file(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = "Alice: Hello there\nBob: Hi back"
        _patch_build_backend(monkeypatch, backend)

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A friendly engineer",
            persona2="Bob", persona2_desc="A curious student",
            topic="weather", scenario="cafe", style="Casual",
            max_new_tokens=512,
            num_examples=1, enable_variation=False, dedup_threshold=0,
        )
        assert "Alice" in preview
        assert "Bob" in preview
        assert "Hello there" in preview
        assert "1" in status  # count appears
        assert file_path is not None and os.path.exists(file_path)

    def test_returns_error_when_generation_fails(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = None
        _patch_build_backend(monkeypatch, backend)

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="", load_in_4bit=False,
            persona1="A", persona1_desc="a", persona2="B", persona2_desc="b",
            topic="T", scenario="S", style="St",
            max_new_tokens=512,
            num_examples=1, enable_variation=False, dedup_threshold=0,
        )
        assert "fail" in status.lower() or "error" in status.lower()
        assert file_path is None

    def test_returns_error_when_build_backend_raises(self, monkeypatch):
        from webapp import generate_handler

        def boom(**kw):
            raise RuntimeError("missing api_base_url")
        monkeypatch.setattr("webapp.build_backend", boom)

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="", api_key="", load_in_4bit=False,
            persona1="A", persona1_desc="a", persona2="B", persona2_desc="b",
            topic="T", scenario="S", style="St",
            max_new_tokens=512,
            num_examples=1, enable_variation=False, dedup_threshold=0,
        )
        assert "missing api_base_url" in status or "error" in status.lower()
        assert file_path is None

    def test_passes_form_values_through_to_build_backend(self, monkeypatch):
        from webapp import generate_handler

        captured = {}
        def fake_build(**kw):
            captured.update(kw)
            backend = MagicMock()
            backend.complete.return_value = "Alice: hi\nBob: hello"
            return backend
        monkeypatch.setattr("webapp.build_backend", fake_build)

        generate_handler(
            backend_kind="openai", model_id="llama3.2:1b",
            api_base_url="http://localhost:11434/v1", api_key="my-key",
            load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=512,
            num_examples=1, enable_variation=False, dedup_threshold=0,
        )
        assert captured["kind"] == "openai"
        assert captured["model_id"] == "llama3.2:1b"
        assert captured["api_base_url"] == "http://localhost:11434/v1"
        assert captured["api_key"] == "my-key"


class TestGenerateHandlerBulk:
    """N>1 produces multiple conversations + metrics + JSONL."""

    def test_writes_n_conversations_to_jsonl(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = "Alice: hi {n}\nBob: hello"
        _patch_build_backend(monkeypatch, backend)

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=3, enable_variation=False, dedup_threshold=0,
        )
        assert file_path is not None and os.path.exists(file_path)
        with open(file_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        conv_ids = sorted({r["conversation_id"] for r in rows})
        assert conv_ids == [0, 1, 2]
        assert "3" in status

    def test_metrics_card_present_when_n_gt_one(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = "Alice: hi\nBob: hello"
        _patch_build_backend(monkeypatch, backend)
        # short-circuit eval to avoid loading sentence-transformers in tests
        monkeypatch.setattr(
            "webapp._compute_metrics",
            lambda path: {"distinct_1": 0.42, "vendi_score": 1.7, "num_conversations": 2},
        )

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="a", persona2="Bob", persona2_desc="b",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=2, enable_variation=False, dedup_threshold=0,
        )
        assert "Distinct" in metrics or "distinct" in metrics.lower()
        assert "Vendi" in metrics or "vendi" in metrics.lower()

    def test_dedup_drops_near_duplicates(self, monkeypatch):
        from webapp import generate_handler

        # Mock generate_conversation directly so we can return identical turns
        identical_turns = _two_turns()
        monkeypatch.setattr(
            "webapp.generate_conversation",
            lambda **kw: identical_turns,
        )
        backend = MagicMock()
        _patch_build_backend(monkeypatch, backend)

        # Mock embedding model: every conversation gets the SAME embedding
        # so dedup should drop everything after the first.
        import numpy as np
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        monkeypatch.setattr("webapp._load_dedup_model", lambda thr: fake_model)
        monkeypatch.setattr(
            "webapp._compute_metrics",
            lambda path: {"num_conversations": 1, "distinct_1": 0.0},
        )

        status, metrics, preview, file_path = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="A", persona1_desc="a", persona2="B", persona2_desc="b",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=5, enable_variation=False, dedup_threshold=0.9,
        )
        # Only 1 conversation should make it to the JSONL
        with open(file_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        conv_ids = {r["conversation_id"] for r in rows}
        assert len(conv_ids) == 1
        assert "drop" in status.lower() or "duplicate" in status.lower()


class TestFormatHelpers:
    def test_format_preview_caps_at_limit(self):
        from webapp import _format_preview
        convs = [
            {"turns": _two_turns(f"X{i}", f"Y{i}"), "topic": f"t{i}"}
            for i in range(10)
        ]
        out = _format_preview(convs, limit=3)
        assert "X0" in out
        assert "X2" in out
        assert "X3" not in out  # cut off

    def test_format_preview_handles_empty(self):
        from webapp import _format_preview
        assert _format_preview([], limit=3) == ""

    def test_format_metrics_renders_known_keys(self):
        from webapp import _format_metrics_card
        md = _format_metrics_card({
            "num_conversations": 5, "total_turns": 60, "avg_turns": 12.0,
            "distinct_1": 0.5, "distinct_2": 0.8, "distinct_3": 0.95,
            "vocabulary_richness": 0.6, "vendi_score": 4.2,
            "topic_diversity": 0.65,
        }, dedup_drops=2)
        low = md.lower()
        assert "distinct" in low
        assert "vendi" in low
        assert "5" in md  # num_conversations
        assert "drop" in low
