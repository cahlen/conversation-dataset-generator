"""Tests for the ChatBackend protocol and its implementations."""

import pytest
from unittest.mock import MagicMock


class TestImports:
    def test_chat_backend_protocol_importable(self):
        from conversation_dataset_generator.backend import ChatBackend
        assert ChatBackend is not None


def _make_hf_mocks(prompt_prefix: str = "PROMPT:", generated: str = "Hello"):
    """Build a (pipeline, tokenizer) pair that round-trips through HFBackend."""
    pipeline = MagicMock()
    pipeline.side_effect = lambda p, **kw: [{"generated_text": p + generated}]
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = prompt_prefix
    tokenizer.eos_token_id = 0
    return pipeline, tokenizer


class TestHFBackend:
    def test_returns_generated_text_with_prompt_stripped(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline, tokenizer = _make_hf_mocks(generated="Hello world")
        backend = HFBackend(pipeline, tokenizer)
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_returns_none_on_empty_generation(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline = MagicMock()
        pipeline.side_effect = lambda p, **kw: [{"generated_text": p}]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "PROMPT:"
        tokenizer.eos_token_id = 0
        backend = HFBackend(pipeline, tokenizer)
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_when_pipeline_raises(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "PROMPT:"
        tokenizer.eos_token_id = 0
        backend = HFBackend(pipeline, tokenizer)
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_when_apply_chat_template_raises(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.side_effect = RuntimeError("template error")
        tokenizer.eos_token_id = 0
        backend = HFBackend(pipeline, tokenizer)
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None
        pipeline.assert_not_called()

    def test_passes_sampling_params_to_pipeline(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline, tokenizer = _make_hf_mocks(generated="ok")
        tokenizer.eos_token_id = 42
        backend = HFBackend(pipeline, tokenizer)
        backend.complete(
            [{"role": "user", "content": "x"}],
            max_new_tokens=999,
            temperature=0.5,
            top_p=0.7,
        )
        kwargs = pipeline.call_args.kwargs
        assert kwargs["max_new_tokens"] == 999
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.7
        assert kwargs["eos_token_id"] == 42

    def test_returns_none_on_unexpected_pipeline_output(self):
        from conversation_dataset_generator.backend import HFBackend
        pipeline = MagicMock(return_value=None)
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "PROMPT:"
        tokenizer.eos_token_id = 0
        backend = HFBackend(pipeline, tokenizer)
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None
