"""Tests for the Gradio webapp's pure handler function.

The Gradio UI binding itself is declarative — we only test the handler that
receives form values, builds a backend, runs generation, and formats output.
"""

from unittest.mock import MagicMock


def _patch_build_backend(monkeypatch, return_value):
    """Make webapp.build_backend return the given value without touching the real one."""
    monkeypatch.setattr("webapp.build_backend", lambda **kw: return_value)


class TestGenerateHandler:
    def test_returns_formatted_conversation_on_success(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = "Alice: Hello there\nBob: Hi back"
        _patch_build_backend(monkeypatch, backend)

        result = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="k", load_in_4bit=False,
            persona1="Alice", persona1_desc="A friendly engineer",
            persona2="Bob", persona2_desc="A curious student",
            topic="weather", scenario="cafe", style="Casual",
            max_new_tokens=512,
        )
        assert "Alice" in result
        assert "Bob" in result
        assert "Hello there" in result
        assert "Hi back" in result

    def test_returns_error_message_when_generation_fails(self, monkeypatch):
        from webapp import generate_handler

        backend = MagicMock()
        backend.complete.return_value = None
        _patch_build_backend(monkeypatch, backend)

        result = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="http://x/v1", api_key="", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=512,
        )
        assert "fail" in result.lower() or "error" in result.lower()

    def test_returns_error_when_build_backend_raises(self, monkeypatch):
        from webapp import generate_handler

        def boom(**kw):
            raise RuntimeError("missing api_base_url")
        monkeypatch.setattr("webapp.build_backend", boom)

        result = generate_handler(
            backend_kind="openai", model_id="m",
            api_base_url="", api_key="", load_in_4bit=False,
            persona1="Alice", persona1_desc="A",
            persona2="Bob", persona2_desc="B",
            topic="T", scenario="S", style="St",
            max_new_tokens=512,
        )
        assert "missing api_base_url" in result or "fail" in result.lower()

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
        )
        assert captured["kind"] == "openai"
        assert captured["model_id"] == "llama3.2:1b"
        assert captured["api_base_url"] == "http://localhost:11434/v1"
        assert captured["api_key"] == "my-key"
