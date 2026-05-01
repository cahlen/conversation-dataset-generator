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

        status, metrics, preview, file_path, _state = generate_handler(
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

        status, metrics, preview, file_path, _state = generate_handler(
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

        status, metrics, preview, file_path, _state = generate_handler(
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

        status, metrics, preview, file_path, _state = generate_handler(
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

        status, metrics, preview, file_path, _state = generate_handler(
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

        status, metrics, preview, file_path, _state = generate_handler(
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

    def test_parse_extra_personas_empty(self):
        from webapp import _parse_extra_personas
        assert _parse_extra_personas("") == []
        assert _parse_extra_personas("   ") == []

    def test_parse_extra_personas_two_lines(self):
        from webapp import _parse_extra_personas
        text = "Carol | A skeptical scientist\nDave | An eager intern"
        assert _parse_extra_personas(text) == [
            ("Carol", "A skeptical scientist"),
            ("Dave", "An eager intern"),
        ]

    def test_parse_extra_personas_skips_malformed(self):
        from webapp import _parse_extra_personas
        text = "Carol | A scientist\nbadline_no_pipe\n  \nDave | An intern"
        assert _parse_extra_personas(text) == [
            ("Carol", "A scientist"),
            ("Dave", "An intern"),
        ]

    def test_parse_extra_personas_strips_whitespace(self):
        from webapp import _parse_extra_personas
        text = "  Carol  |   A scientist  "
        assert _parse_extra_personas(text) == [("Carol", "A scientist")]


class TestGenerateHandlerNSpeaker:
    def test_extra_personas_pass_through_as_n_speaker(self, monkeypatch):
        from webapp import generate_handler

        captured = {}
        def fake_generate(**kwargs):
            captured["personas"] = kwargs.get("personas")
            captured["persona1"] = kwargs.get("persona1")
            return [
                {"from": "human", "value": "hi", "speaker_name": "Alice"},
                {"from": "gpt", "value": "hello", "speaker_name": "Bob"},
                {"from": "gpt", "value": "hey", "speaker_name": "Carol"},
            ]
        monkeypatch.setattr("webapp.generate_conversation", fake_generate)
        backend = MagicMock()
        backend.complete.return_value = "anything"
        _patch_build_backend(monkeypatch, backend)

        status, metrics, preview, file_path, _state = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=1, enable_variation=False, dedup_threshold=0,
            extra_personas="Carol | A skeptical scientist",
        )
        # When extra personas exist, the N-speaker path uses personas= kwarg
        assert captured["personas"] == [
            ("Alice", "A"),
            ("Bob", "B"),
            ("Carol", "A skeptical scientist"),
        ]


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


class TestAutoFix:
    """auto_fix_handler dispatches a fix per failing metric."""

    def _base_kwargs(self):
        return dict(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A engineer",
            persona2="Bob", persona2_desc="A student",
            extra_personas="",
            topic="weather", scenario="a cafe", style="Casual",
            max_new_tokens=1024, enable_variation=False, dedup_threshold=0.0,
        )

    def test_no_metrics_means_nothing_to_fix(self):
        from webapp import auto_fix_handler
        result = auto_fix_handler(**self._base_kwargs(), metrics_state={})
        assert len(result) == 9
        status = result[-1].lower()
        assert "metric" in status or "generate" in status or "first" in status

    def test_low_distinct2_toggles_variation_on(self, monkeypatch):
        from webapp import auto_fix_handler
        backend = MagicMock()
        backend.complete.return_value = ""
        _patch_build_backend(monkeypatch, backend)
        metrics = {"distinct_2": 0.40, "num_conversations": 5}
        result = auto_fix_handler(**self._base_kwargs(), metrics_state=metrics)
        assert result[7] is True

    def test_high_self_repetition_lowers_max_tokens(self, monkeypatch):
        from webapp import auto_fix_handler
        backend = MagicMock()
        backend.complete.return_value = ""
        _patch_build_backend(monkeypatch, backend)
        metrics = {"self_repetition_rate": 0.20, "num_conversations": 5}
        result = auto_fix_handler(**self._base_kwargs(), metrics_state=metrics)
        assert result[6] < 1024

    def test_low_speaker_distinctness_rewrites_personas(self, monkeypatch):
        from webapp import auto_fix_handler
        backend = MagicMock()
        backend.complete.return_value = "Alice => Sharp clinical voice.\nBob => Verbose emotional voice."
        _patch_build_backend(monkeypatch, backend)
        metrics = {"speaker_distinctiveness": 0.15, "num_conversations": 5}
        result = auto_fix_handler(**self._base_kwargs(), metrics_state=metrics)
        assert "clinical" in result[0].lower()
        assert "verbose" in result[1].lower() or "emotional" in result[1].lower()

    def test_low_topic_diversity_rewrites_topic(self, monkeypatch):
        from webapp import auto_fix_handler
        backend = MagicMock()
        backend.complete.side_effect = ["A broad survey of seasonal patterns and human reactions to weather"]
        _patch_build_backend(monkeypatch, backend)
        metrics = {"topic_diversity": 0.30, "num_conversations": 5}
        result = auto_fix_handler(**self._base_kwargs(), metrics_state=metrics)
        assert result[3] != "weather"
        assert len(result[3]) > 10

    def test_status_summarizes_applied_fixes(self, monkeypatch):
        from webapp import auto_fix_handler
        backend = MagicMock()
        backend.complete.return_value = "Alice => x\nBob => y"
        _patch_build_backend(monkeypatch, backend)
        metrics = {"speaker_distinctiveness": 0.15, "distinct_2": 0.40, "num_conversations": 5}
        result = auto_fix_handler(**self._base_kwargs(), metrics_state=metrics)
        status = result[-1].lower()
        assert "persona" in status or "variation" in status or "fix" in status


class TestFixPersonas:
    def test_parse_rewritten_personas_picks_matching_names(self):
        from webapp import _parse_rewritten_personas
        text = (
            "Alice => A grizzled chief engineer with cynical humor and clipped sentences.\n"
            "Bob => A buoyant intern who interrupts with metaphors from skateboarding."
        )
        result = _parse_rewritten_personas(text, ["Alice", "Bob"])
        assert "Alice" in result
        assert "Bob" in result
        assert "grizzled" in result["Alice"]
        assert "skateboarding" in result["Bob"]

    def test_parse_rewritten_personas_skips_non_matching_names(self):
        from webapp import _parse_rewritten_personas
        text = "Carol => A sarcastic critic.\nAlice => Updated."
        result = _parse_rewritten_personas(text, ["Alice", "Bob"])
        assert result == {"Alice": "Updated."}

    def test_parse_rewritten_personas_handles_bullet_prefixes(self):
        from webapp import _parse_rewritten_personas
        text = "- Alice => Sharp and terse.\n• Bob => Verbose and warm."
        result = _parse_rewritten_personas(text, ["Alice", "Bob"])
        assert result["Alice"] == "Sharp and terse."
        assert result["Bob"] == "Verbose and warm."

    def test_fix_personas_handler_rewrites_descriptions(self, monkeypatch):
        from webapp import fix_personas_handler

        backend = MagicMock()
        backend.complete.return_value = (
            "Alice => A wry, no-nonsense senior with caustic wit.\n"
            "Bob => An over-caffeinated rookie who quotes movies."
        )
        _patch_build_backend(monkeypatch, backend)

        new_p1d, new_p2d, new_extras, status = fix_personas_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A friendly engineer",
            persona2="Bob", persona2_desc="A curious student",
            extra_personas="",
        )
        assert "wry" in new_p1d
        assert "rookie" in new_p2d
        assert "rewrote" in status.lower() or "done" in status.lower()

    def test_fix_personas_handler_rewrites_extras(self, monkeypatch):
        from webapp import fix_personas_handler

        backend = MagicMock()
        backend.complete.return_value = (
            "Alice => Sharp.\nBob => Soft.\nCarol => Manic."
        )
        _patch_build_backend(monkeypatch, backend)

        _, _, new_extras, _ = fix_personas_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="x",
            persona2="Bob", persona2_desc="y",
            extra_personas="Carol | Original",
        )
        assert "Carol | Manic" in new_extras

    def test_fix_personas_handler_returns_originals_on_backend_failure(self, monkeypatch):
        from webapp import fix_personas_handler
        def boom(**kw):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("webapp.build_backend", boom)

        new_p1d, new_p2d, new_extras, status = fix_personas_handler(
            backend_kind="openai", model_id="m",
            api_base_url="", api_key="", load_in_4bit=False,
            persona1="Alice", persona1_desc="ORIGINAL_P1",
            persona2="Bob", persona2_desc="ORIGINAL_P2",
            extra_personas="Carol | ORIGINAL_C",
        )
        assert new_p1d == "ORIGINAL_P1"
        assert new_p2d == "ORIGINAL_P2"
        assert new_extras == "Carol | ORIGINAL_C"
        assert "error" in status.lower() or "fail" in status.lower()


class TestBrainstormHandler:
    """Creative-brief mode: paste a brief, LLM populates persona/scene fields."""

    def test_populates_form_fields(self, monkeypatch):
        from webapp import brainstorm_handler

        backend = MagicMock()
        backend.complete.return_value = (
            '--persona1 "Iron Man"\n--persona1-desc "Genius billionaire"\n'
            '--persona2 "Captain America"\n--persona2-desc "Earnest leader"\n'
            '--topic "team dynamics"\n--scenario "the conference room"\n'
            '--style "Tense"'
        )
        _patch_build_backend(monkeypatch, backend)

        result = brainstorm_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            brief="Two avengers debate strategy",
        )
        # 8 outputs: persona1, persona1_desc, persona2, persona2_desc, topic, scenario, style, status
        assert len(result) == 8
        assert result[0] == "Iron Man"
        assert "billionaire" in result[1]
        assert result[2] == "Captain America"
        assert result[4] == "team dynamics"
        assert "conference" in result[5]

    def test_empty_brief_returns_error(self):
        from webapp import brainstorm_handler
        result = brainstorm_handler(
            backend_kind="openai", model_id="m",
            api_base_url="x", api_key="", load_in_4bit=False,
            brief="",
        )
        status = result[-1].lower()
        assert "brief" in status or "empty" in status

    def test_backend_failure_returns_error(self, monkeypatch):
        from webapp import brainstorm_handler
        def boom(**kw):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("webapp.build_backend", boom)
        result = brainstorm_handler(
            backend_kind="openai", model_id="m",
            api_base_url="x", api_key="", load_in_4bit=False,
            brief="A real brief",
        )
        status = result[-1].lower()
        assert "error" in status or "fail" in status or "refused" in status


class TestTrainSpeaker:
    """train_speaker dropdown should produce role_mapping for fine-tuning."""

    def test_auto_train_speaker_means_no_explicit_mapping(self, monkeypatch):
        from webapp import generate_handler

        captured = {}
        def fake_gen(**kw):
            captured.update(kw)
            return [{"from": "human", "value": "hi", "speaker_name": "Alice"}]
        monkeypatch.setattr("webapp.generate_conversation", fake_gen)
        backend = MagicMock(); backend.complete.return_value = "x"
        _patch_build_backend(monkeypatch, backend)

        generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=1, enable_variation=False, dedup_threshold=0,
            train_speaker="auto",
        )
        # When auto, role_mapping should be None or absent
        assert captured.get("role_mapping") in (None, {})

    def test_named_train_speaker_builds_mapping(self, monkeypatch):
        from webapp import generate_handler

        captured = {}
        def fake_gen(**kw):
            captured.update(kw)
            return [{"from": "gpt", "value": "hi", "speaker_name": "Bob"}]
        monkeypatch.setattr("webapp.generate_conversation", fake_gen)
        backend = MagicMock(); backend.complete.return_value = "x"
        _patch_build_backend(monkeypatch, backend)

        generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=1, enable_variation=False, dedup_threshold=0,
            train_speaker="Bob",
        )
        rm = captured.get("role_mapping") or {}
        assert rm.get("Bob") == "gpt"
        assert rm.get("Alice") == "human"


class TestIncludePoints:
    def test_include_points_passed_through(self, monkeypatch):
        from webapp import generate_handler

        captured = {}
        def fake_gen(**kw):
            captured.update(kw)
            return [{"from": "human", "value": "hi", "speaker_name": "Alice"}]
        monkeypatch.setattr("webapp.generate_conversation", fake_gen)
        backend = MagicMock(); backend.complete.return_value = "x"
        _patch_build_backend(monkeypatch, backend)

        generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=256,
            num_examples=1, enable_variation=False, dedup_threshold=0,
            include_points="rain, sun, umbrellas",
        )
        assert captured.get("include_points") == "rain, sun, umbrellas"


class TestPresets:
    def test_presets_include_custom_and_at_least_two_examples(self):
        from webapp import PRESETS
        names = list(PRESETS.keys())
        # First entry should be the no-op "Custom" option
        assert names[0].lower().startswith("—") or "custom" in names[0].lower()
        # Plus at least 2 real presets
        assert len(names) >= 3

    def test_apply_preset_returns_form_values(self):
        from webapp import _apply_preset, PRESETS
        # Pick the first non-custom preset
        real_preset = [k for k in PRESETS if PRESETS[k]][0]
        result = _apply_preset(real_preset)
        # 8 outputs: persona1, persona1_desc, persona2, persona2_desc, extras, topic, scenario, style
        assert len(result) == 8
        # First field should be a non-empty persona1 name
        p = PRESETS[real_preset]
        assert result[0] == p["persona1"]
        assert result[5] == p["topic"]

    def test_apply_preset_avengers_has_three_speakers(self):
        from webapp import PRESETS
        # The Avengers preset should include extras (3rd speaker)
        avengers = next((v for k, v in PRESETS.items() if v and "veng" in k.lower()), None)
        assert avengers is not None, "expected an Avengers preset"
        assert avengers.get("extra_personas", "").strip() != ""


class TestRecommendations:
    def test_low_speaker_distinctness_recommends_contrastive_descriptions(self):
        from webapp import _recommendations
        recs = _recommendations({"speaker_distinctiveness": 0.10, "num_conversations": 5})
        assert any("contrast" in r.lower() or "persona" in r.lower() for r in recs)

    def test_low_distinct_2_recommends_variation(self):
        from webapp import _recommendations
        recs = _recommendations({"distinct_2": 0.40, "num_conversations": 5})
        assert any("variation" in r.lower() or "vary" in r.lower() for r in recs)

    def test_high_self_repetition_recommends_shortening(self):
        from webapp import _recommendations
        recs = _recommendations({"self_repetition_rate": 0.20, "num_conversations": 5})
        assert any("max" in r.lower() or "repeat" in r.lower() for r in recs)

    def test_coherence_too_high_recommends_temperature(self):
        from webapp import _recommendations
        recs = _recommendations({"turn_coherence": 0.85, "num_conversations": 5})
        assert any("temperature" in r.lower() or "robotic" in r.lower() for r in recs)

    def test_no_recommendations_when_all_targets_met(self):
        from webapp import _recommendations
        recs = _recommendations({
            "speaker_distinctiveness": 0.45, "distinct_2": 0.85,
            "topic_diversity": 0.70, "self_repetition_rate": 0.02,
            "turn_coherence": 0.45, "num_conversations": 5,
        })
        assert recs == []
