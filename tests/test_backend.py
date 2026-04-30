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


def _make_openai_client(content: str | None = "Hello world"):
    """Build a mock openai.OpenAI-shaped client whose chat.completions.create
    returns a single choice with the given content."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    client.chat.completions.create.return_value = completion
    return client


class TestOpenAIBackend:
    def test_returns_content_on_success(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = _make_openai_client(content="Hello world")
        backend = OpenAIBackend(
            model_id="qwen-7b",
            base_url="http://localhost:1234/v1",
            api_key="test",
            client=client,
        )
        result = backend.complete([{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_passes_messages_and_model(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = _make_openai_client()
        backend = OpenAIBackend(
            model_id="qwen-7b", base_url="x", api_key="y", client=client,
        )
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        backend.complete(msgs)
        call = client.chat.completions.create.call_args
        assert call.kwargs["model"] == "qwen-7b"
        assert call.kwargs["messages"] == msgs

    def test_translates_max_new_tokens_to_max_tokens(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = _make_openai_client()
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        backend.complete(
            [{"role": "user", "content": "x"}],
            max_new_tokens=777, temperature=0.3, top_p=0.6,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 777
        assert kwargs["temperature"] == 0.3
        assert kwargs["top_p"] == 0.6
        assert "max_new_tokens" not in kwargs

    def test_returns_none_on_empty_content(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = _make_openai_client(content="")
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_on_none_content(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = _make_openai_client(content=None)
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_on_connection_error(self):
        import openai
        from conversation_dataset_generator.backend import OpenAIBackend
        client = MagicMock()
        client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_on_generic_exception(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None

    def test_returns_none_on_no_choices(self):
        from conversation_dataset_generator.backend import OpenAIBackend
        client = MagicMock()
        completion = MagicMock()
        completion.choices = []
        client.chat.completions.create.return_value = completion
        backend = OpenAIBackend(
            model_id="m", base_url="x", api_key="y", client=client,
        )
        result = backend.complete([{"role": "user", "content": "x"}])
        assert result is None


class TestMakeBackend:
    def test_hf_kind(self):
        from conversation_dataset_generator.backend import make_backend, HFBackend
        pipeline = MagicMock()
        tokenizer = MagicMock()
        backend = make_backend("hf", pipeline=pipeline, tokenizer=tokenizer)
        assert isinstance(backend, HFBackend)

    def test_openai_kind(self):
        from conversation_dataset_generator.backend import make_backend, OpenAIBackend
        client = _make_openai_client()
        backend = make_backend(
            "openai",
            model_id="m",
            base_url="http://localhost:1234/v1",
            api_key="k",
            client=client,
        )
        assert isinstance(backend, OpenAIBackend)

    def test_unknown_kind_raises(self):
        from conversation_dataset_generator.backend import make_backend
        with pytest.raises(ValueError, match="unknown backend"):
            make_backend("nope")
