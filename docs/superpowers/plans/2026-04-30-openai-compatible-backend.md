# OpenAI-Compatible Inference Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users plug `generate.py` into any OpenAI-compatible inference server (LM Studio, Ollama, vLLM, the real OpenAI API) so the tool works without local CUDA + transformers + bitsandbytes.

**Architecture:** Introduce a `ChatBackend` protocol with one method, `complete(messages, max_new_tokens, temperature, top_p) -> str | None`. Two implementations: `HFBackend` wraps the existing `(pipeline, tokenizer)` pair (preserving today's behavior); `OpenAIBackend` wraps the `openai.OpenAI` client with a custom `base_url`. Refactor the four generation entry points (`generate_args_from_brief`, `generate_topic_variation`, `generate_conversation`, `generate_continuation`) so they accept a `backend` rather than a `(generator_pipeline, tokenizer)` pair. Add `--backend`, `--api-base-url`, `--api-key` CLI flags. The internal helper `_call_pipeline` and the public helper `extract_generated_text` move into the new module — the abstraction boundary already exists in spirit (everything downstream consumes `messages` lists), we just give it a name.

**Tech Stack:** Python 3.10+, `openai>=1.0` (new dep), `pytest`, `unittest.mock`. No torch/transformers code changes — those stay isolated to `models.py` and `HFBackend`.

**Resolved decisions (locked in for this plan):**
- Clean cut on the signature change (no dual-mode API), since the codebase is internal and the test suite is small enough to update in one task.
- `ChatBackend.complete()` returns `None` on any failure (network error, parse error, empty response). Mirrors today's `_call_pipeline` contract — the existing retry logic at `generate_args_from_brief` already handles `None` returns.
- Protocol uses HF terminology (`max_new_tokens`) to minimize call-site churn. `OpenAIBackend` translates internally to `max_tokens`.
- Default `--backend` is `hf` to preserve today's UX. `--api-base-url` defaults to LM Studio (`http://localhost:1234/v1`). `--api-key` falls back to env `OPENAI_API_KEY`, then to `"not-needed"` (so local servers that ignore the key just work).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `conversation_dataset_generator/backend.py` | **Create** | `ChatBackend` Protocol, `HFBackend`, `OpenAIBackend`, `make_backend()`, `_extract_generated_text()` helper |
| `tests/test_backend.py` | **Create** | All tests for the new module |
| `conversation_dataset_generator/generation.py` | Modify | Drop `_call_pipeline` and `extract_generated_text`; the four generators take `backend` instead of `(generator_pipeline, tokenizer)` |
| `conversation_dataset_generator/cli.py` | Modify | New flags; `build_backend_from_args(args)` helper; `main()` wires backend through |
| `tests/test_generation.py` | Modify | Replace `make_mock_pipeline`/`make_mock_tokenizer` with `make_mock_backend`; update all call sites |
| `tests/test_cli.py` | Modify | Add tests for `--backend`, `--api-base-url`, `--api-key`, `build_backend_from_args` |
| `requirements.txt` | Modify | Add `openai>=1.0` |
| `README.md` | Modify | Document new flags + LM Studio/Ollama recipes |
| `CLAUDE.md` | Modify | Add `backend.py` to the architecture table |

---

## Task 1: Bootstrap — add openai dep, create empty backend module

**Files:**
- Modify: `requirements.txt`
- Create: `conversation_dataset_generator/backend.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backend.py`:

```python
"""Tests for the ChatBackend protocol and its implementations."""

import pytest
from unittest.mock import MagicMock


class TestImports:
    def test_chat_backend_protocol_importable(self):
        from conversation_dataset_generator.backend import ChatBackend
        assert ChatBackend is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend.py::TestImports::test_chat_backend_protocol_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation_dataset_generator.backend'`

- [ ] **Step 3: Add openai dep**

Append to `requirements.txt`:

```
openai>=1.0
```

Run: `pip install -r requirements.txt`

- [ ] **Step 4: Create the backend module**

Write `conversation_dataset_generator/backend.py`:

```python
"""Chat backends: HF pipeline and OpenAI-compatible HTTP."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ChatBackend(Protocol):
    """A chat-completion backend.

    Implementations take a list of role/content message dicts and return the
    generated assistant text (just the new tokens, not the prompt). On any
    failure — network error, parse error, empty response — return None.
    """

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_backend.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt conversation_dataset_generator/backend.py tests/test_backend.py
git commit -m "scaffold ChatBackend protocol and add openai dependency"
```

---

## Task 2: HFBackend — wraps the existing pipeline + tokenizer

**Files:**
- Modify: `conversation_dataset_generator/backend.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend.py::TestHFBackend -v`
Expected: FAIL with `ImportError: cannot import name 'HFBackend'` (all six tests)

- [ ] **Step 3: Implement HFBackend**

Append to `conversation_dataset_generator/backend.py`:

```python
def _extract_generated_text(full_output: str, prompt_text: str) -> str | None:
    """Strip the prompt prefix from a pipeline's output. Return None if empty."""
    if full_output.startswith(prompt_text):
        generated = full_output[len(prompt_text):]
    else:
        generated = full_output
    generated = generated.strip()
    return generated if generated else None


class HFBackend:
    """ChatBackend backed by a transformers text-generation pipeline + tokenizer."""

    def __init__(self, pipeline, tokenizer):
        self._pipeline = pipeline
        self._tokenizer = tokenizer

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as exc:
            logger.error("apply_chat_template failed: %s", exc)
            return None

        try:
            outputs = self._pipeline(
                prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception as exc:
            logger.error("HF pipeline call failed: %s", exc)
            return None

        if not outputs or not isinstance(outputs, list):
            logger.warning("Pipeline returned unexpected output: %r", outputs)
            return None

        raw = outputs[0].get("generated_text", "")
        return _extract_generated_text(raw, prompt_text)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend.py -v`
Expected: PASS (all HFBackend tests + the import test)

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/backend.py tests/test_backend.py
git commit -m "add HFBackend wrapping transformers pipeline"
```

---

## Task 3: OpenAIBackend — wraps the openai SDK against any compatible URL

**Files:**
- Modify: `conversation_dataset_generator/backend.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend.py::TestOpenAIBackend -v`
Expected: FAIL with `ImportError: cannot import name 'OpenAIBackend'`

- [ ] **Step 3: Implement OpenAIBackend**

Append to `conversation_dataset_generator/backend.py`:

```python
class OpenAIBackend:
    """ChatBackend backed by an OpenAI-compatible HTTP API.

    Works with the real OpenAI API as well as LM Studio, Ollama (`/v1` shim),
    vLLM, TGI, OpenRouter — anything that speaks `chat.completions`.

    Pass ``client`` to inject a pre-built or mocked client (used in tests);
    otherwise the constructor creates one from ``base_url`` + ``api_key``.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        *,
        client=None,
    ):
        self.model_id = model_id
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def complete(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str | None:
        try:
            completion = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            logger.error("OpenAI chat.completions call failed: %s", exc)
            return None

        if not completion.choices:
            logger.warning("OpenAI response had no choices.")
            return None

        content = completion.choices[0].message.content
        if not content:
            return None
        content = content.strip()
        return content if content else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend.py -v`
Expected: PASS (all backend tests so far)

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/backend.py tests/test_backend.py
git commit -m "add OpenAIBackend for OpenAI-compatible HTTP servers"
```

---

## Task 4: `make_backend` factory

**Files:**
- Modify: `conversation_dataset_generator/backend.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backend.py::TestMakeBackend -v`
Expected: FAIL with `ImportError: cannot import name 'make_backend'`

- [ ] **Step 3: Implement the factory**

Append to `conversation_dataset_generator/backend.py`:

```python
def make_backend(kind: str, **kwargs) -> ChatBackend:
    """Construct a ChatBackend by kind name.

    For kind="hf": pass pipeline=, tokenizer=.
    For kind="openai": pass model_id=, base_url=, api_key=, optionally client=.
    """
    if kind == "hf":
        return HFBackend(kwargs["pipeline"], kwargs["tokenizer"])
    if kind == "openai":
        return OpenAIBackend(
            model_id=kwargs["model_id"],
            base_url=kwargs["base_url"],
            api_key=kwargs.get("api_key"),
            client=kwargs.get("client"),
        )
    raise ValueError(f"unknown backend: {kind!r} (expected 'hf' or 'openai')")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/backend.py tests/test_backend.py
git commit -m "add make_backend factory"
```

---

## Task 5: Refactor `generation.py` — generators take `backend`, not `(pipeline, tokenizer)`

This is the cut-over. The old `_call_pipeline` and `extract_generated_text` are deleted from `generation.py` (their logic now lives in `backend.py`). The four entry-point signatures change from `(generator_pipeline, tokenizer)` → `(backend)`. Tests in `test_generation.py` are rewritten to mock the backend instead of the pipeline+tokenizer pair.

**Files:**
- Modify: `conversation_dataset_generator/generation.py`
- Modify: `tests/test_generation.py`

- [ ] **Step 1: Update tests to mock the backend instead**

Overwrite `tests/test_generation.py` with:

```python
import pytest
from unittest.mock import MagicMock
from conversation_dataset_generator.generation import (
    generate_args_from_brief,
    generate_topic_variation,
    generate_conversation,
    generate_continuation,
)


def make_mock_backend(response_text: str | None):
    """Return a mock ChatBackend whose .complete() returns response_text."""
    backend = MagicMock()
    backend.complete.return_value = response_text
    return backend


class TestGenerateArgsFromBrief:
    def test_successful_generation(self):
        response = (
            '--persona1 "Sherlock"\n'
            '--persona1-desc "A brilliant detective"\n'
            '--persona2 "Watson"\n'
            '--persona2-desc "A loyal doctor"\n'
            '--topic "A mysterious case"\n'
            '--scenario "221B Baker Street"\n'
            '--style "Dramatic and suspenseful"'
        )
        backend = make_mock_backend(response)
        result = generate_args_from_brief("Sherlock and Watson", backend)
        assert result is not None
        assert result["persona1"] == "Sherlock"
        assert result["persona2"] == "Watson"

    def test_returns_none_on_garbage_output(self):
        backend = make_mock_backend("This is just random garbage text")
        result = generate_args_from_brief("Test brief", backend, max_retries=1)
        assert result is None

    def test_applies_defaults_for_missing_optional_fields(self):
        response = (
            '--persona1 "Sherlock"\n'
            '--persona1-desc "Detective"\n'
            '--persona2 "Watson"\n'
            '--persona2-desc "Doctor"\n'
            '--topic "Crime"\n'
            '--scenario "London"\n'
            '--style "Tense"'
        )
        backend = make_mock_backend(response)
        result = generate_args_from_brief("Test", backend)
        assert result is not None


class TestGenerateTopicVariation:
    def test_successful_variation(self):
        response = (
            '--topic "A new topic"\n'
            '--scenario "A new scenario"\n'
            '--style "A new style"'
        )
        backend = make_mock_backend(response)
        result = generate_topic_variation(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
            backend=backend,
        )
        assert result is not None
        assert result["topic"] == "A new topic"
        assert result["scenario"] == "A new scenario"

    def test_returns_none_on_parse_failure(self):
        backend = make_mock_backend("Just random text, no args")
        result = generate_topic_variation(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
            backend=backend,
        )
        assert result is None


class TestGenerateConversation:
    def test_successful_generation(self):
        backend = make_mock_backend("Alice: Hello\nBob: Hi there\nAlice: How are you?")
        turns = generate_conversation(
            topic="Greeting", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Online", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 3

    def test_returns_none_on_empty_output(self):
        backend = make_mock_backend(None)
        turns = generate_conversation(
            topic="T", persona1="A", persona2="B",
            persona1_desc="d1", persona2_desc="d2",
            scenario="S", style="St",
            backend=backend, max_new_tokens=512,
        )
        assert turns is None

    def test_speaker_name_added_to_turns(self):
        backend = make_mock_backend("Alice: Hello\nBob: Hi there\nAlice: How are you?")
        turns = generate_conversation(
            topic="Greeting", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Online", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        assert turns[0]["speaker_name"] == "Alice"
        assert turns[1]["speaker_name"] == "Bob"
        assert turns[2]["speaker_name"] == "Alice"


class TestGenerateConversationMulti:
    def test_three_speakers(self):
        backend = make_mock_backend(
            "Alice: Hello\nBob: Hi\nCharlie: Hey everyone"
        )
        turns = generate_conversation(
            topic="Greet",
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy"), ("Charlie", "Quiet")],
            scenario="Room", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 3
        assert turns[0]["speaker_name"] == "Alice"
        assert turns[1]["speaker_name"] == "Bob"
        assert turns[2]["speaker_name"] == "Charlie"

    def test_legacy_two_speaker(self):
        backend = make_mock_backend("Alice: Hello\nBob: Hi")
        turns = generate_conversation(
            topic="Greet", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Room", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 2

    def test_three_speakers_speaker_names(self):
        backend = make_mock_backend(
            "Alice: Hello\nBob: Hi\nCharlie: Hey everyone"
        )
        turns = generate_conversation(
            topic="Greet",
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy"), ("Charlie", "Quiet")],
            scenario="Room", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        for turn in turns:
            assert "speaker_name" in turn


class TestGenerateContinuation:
    def test_basic_continuation(self):
        backend = make_mock_backend("Alice: Continuing now\nBob: Great")
        prior_turns = [
            {"from": "human", "value": "Hello", "speaker_name": "Alice"},
            {"from": "gpt", "value": "Hi", "speaker_name": "Bob"},
        ]
        turns = generate_continuation(
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy")],
            prior_turns=prior_turns,
            topic="Greet", scenario="Room", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is not None
        assert len(turns) == 2
        assert turns[0]["speaker_name"] == "Alice"
        assert turns[1]["speaker_name"] == "Bob"

    def test_continuation_returns_none_on_empty(self):
        backend = make_mock_backend(None)
        prior_turns = [
            {"from": "human", "value": "Hello", "speaker_name": "Alice"},
        ]
        turns = generate_continuation(
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy")],
            prior_turns=prior_turns,
            topic="Greet", scenario="Room", style="Casual",
            backend=backend, max_new_tokens=512,
        )
        assert turns is None
```

(Note: `TestExtractGeneratedText` is removed — that helper has moved to `backend.py` and is exercised through `HFBackend` tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL — every test errors out with `TypeError` because the entry points still expect `generator_pipeline=`/`tokenizer=`, not `backend=`.

- [ ] **Step 3: Refactor `generation.py`**

Overwrite `conversation_dataset_generator/generation.py`:

```python
"""LLM generation wrappers with retry logic for conversation dataset generation."""

from __future__ import annotations

import logging
import re
import time

from conversation_dataset_generator.parsing import (
    parse_arg_generation_output,
    parse_conversation_to_sharegpt,
    parse_variation_output,
)
from conversation_dataset_generator.prompts import (
    build_arg_generation_messages,
    build_continuation_messages,
    build_conversation_messages,
    build_variation_messages,
)

logger = logging.getLogger(__name__)

_ARG_DEFAULTS = {
    "persona1_desc": "A character in the conversation",
    "persona2_desc": "A character in the conversation",
    "topic": "General discussion",
    "scenario": "An unspecified setting",
    "style": "Casual",
}


def generate_args_from_brief(
    brief: str,
    backend,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Generate conversation arguments from a creative brief."""
    persona1_context: str | None = None
    persona2_context: str | None = None

    if persona1_search_term or persona2_search_term:
        import time as _time
        from conversation_dataset_generator.web_search import get_persona_context

    if persona1_search_term:
        try:
            persona1_context = get_persona_context(persona1_search_term)
        except Exception as exc:
            logger.warning("Web search for persona1 failed: %s", exc)

    if persona1_search_term and persona2_search_term:
        _time.sleep(1.5)

    if persona2_search_term:
        try:
            persona2_context = get_persona_context(persona2_search_term)
        except Exception as exc:
            logger.warning("Web search for persona2 failed: %s", exc)

    messages = build_arg_generation_messages(
        brief,
        persona1_context=persona1_context,
        persona2_context=persona2_context,
        search_term1=persona1_search_term,
        search_term2=persona2_search_term,
    )

    delay = 1
    for attempt in range(max_retries):
        text = backend.complete(messages)
        if text:
            result = parse_arg_generation_output(text)
            if result is not None:
                return result
            logger.warning(
                "Attempt %d/%d: failed to parse arg generation output.",
                attempt + 1, max_retries,
            )
        else:
            logger.warning(
                "Attempt %d/%d: backend returned no text.",
                attempt + 1, max_retries,
            )
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2

    logger.error("generate_args_from_brief exhausted %d retries.", max_retries)
    return None


def generate_args_from_brief_safe(
    brief: str,
    backend,
    persona1_search_term: str | None = None,
    persona2_search_term: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """Wrapper that fills missing optional fields with defaults."""
    result = generate_args_from_brief(
        brief, backend,
        persona1_search_term=persona1_search_term,
        persona2_search_term=persona2_search_term,
        max_retries=max_retries,
    )
    if result is None:
        return None
    for key, default in _ARG_DEFAULTS.items():
        if key not in result or not result[key]:
            result[key] = default
            logger.info("Applied default for missing key '%s': %r", key, default)
    return result


def generate_topic_variation(
    persona1: str, persona1_desc: str,
    persona2: str, persona2_desc: str,
    initial_topic: str, initial_scenario: str, initial_style: str,
    backend,
    original_brief: str | None = None,
) -> dict | None:
    """Generate a topic/scenario/style variation for existing personas."""
    messages = build_variation_messages(
        persona1=persona1, persona1_desc=persona1_desc,
        persona2=persona2, persona2_desc=persona2_desc,
        initial_topic=initial_topic, initial_scenario=initial_scenario,
        initial_style=initial_style, original_brief=original_brief,
    )
    text = backend.complete(messages)
    if not text:
        logger.warning("generate_topic_variation: backend returned no text.")
        return None
    result = parse_variation_output(text)
    if result is None:
        logger.warning("generate_topic_variation: failed to parse variation output.")
        return None
    if "style" not in result or not result["style"]:
        result["style"] = initial_style
    return result


def _add_speaker_names(turns, text, persona_names):
    name_pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(n) for n in persona_names) + r")\s*:",
        re.IGNORECASE,
    )
    lines = text.strip().split("\n")
    turn_idx = 0
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        m = name_pattern.match(line_s)
        if m and turn_idx < len(turns):
            speaker = m.group(1).strip()
            for name in persona_names:
                if speaker.lower() == name.lower():
                    turns[turn_idx]["speaker_name"] = name
                    break
            turn_idx += 1


def generate_conversation(
    topic: str,
    persona1: str | None = None,
    persona2: str | None = None,
    persona1_desc: str | None = None,
    persona2_desc: str | None = None,
    scenario: str = "",
    style: str = "",
    backend=None,
    max_new_tokens: int = 2048,
    include_points: str | None = None,
    role_mapping: dict | None = None,
    *,
    personas: list[tuple[str, str]] | None = None,
) -> list[dict] | None:
    """Generate a conversation and parse it into ShareGPT turn format."""
    if personas is None:
        personas = [
            (persona1, persona1_desc or ""),
            (persona2, persona2_desc or ""),
        ]
    persona_names = [name for name, _ in personas]

    messages = build_conversation_messages(
        topic=topic, personas=personas,
        scenario=scenario, style=style, include_points=include_points,
    )
    text = backend.complete(messages, max_new_tokens=max_new_tokens)
    if not text:
        logger.warning("generate_conversation: backend returned no text.")
        return None

    stripped = text.strip()
    if not any(
        stripped.lower().startswith(f"{name.lower()}:") for name in persona_names
    ):
        logger.warning(
            "generate_conversation: output does not start with a persona prefix. "
            "Got: %r", stripped[:80],
        )

    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping
    )
    if turns is None:
        logger.warning("generate_conversation: failed to parse conversation output.")
        return None

    _add_speaker_names(turns, text, persona_names)
    return turns


def generate_continuation(
    personas: list[tuple[str, str]],
    prior_turns: list[dict],
    topic: str, scenario: str, style: str,
    backend,
    max_new_tokens: int = 2048,
    role_mapping: dict | None = None,
) -> list[dict] | None:
    """Generate a continuation of an existing conversation."""
    persona_names = [name for name, _ in personas]
    messages = build_continuation_messages(
        personas=personas, prior_turns=prior_turns,
        topic=topic, scenario=scenario, style=style,
    )
    text = backend.complete(messages, max_new_tokens=max_new_tokens)
    if not text:
        logger.warning("generate_continuation: backend returned no text.")
        return None
    turns, _ = parse_conversation_to_sharegpt(
        text, personas=persona_names, role_mapping=role_mapping
    )
    if turns is None:
        logger.warning("generate_continuation: failed to parse continuation output.")
        return None
    _add_speaker_names(turns, text, persona_names)
    return turns
```

- [ ] **Step 4: Update CLI call sites**

Edit `conversation_dataset_generator/cli.py`. We're temporarily breaking `main()` here — the next two tasks rewire it cleanly. For now, just rename the kwargs at the four `generate_*` call sites so the test suite passes.

In `main()`:
- Replace `generator_pipeline=text_generator, tokenizer=tokenizer` with `backend=backend` at the four call sites (continue, brief→args, variation, conversation, continuation).
- After the existing `text_generator, tokenizer = load_model_and_pipeline(...)` line, add:

```python
from conversation_dataset_generator.backend import HFBackend
backend = HFBackend(text_generator, tokenizer)
```

(This is a temporary bridge — Task 7 replaces it with the real backend factory.)

Find these specific call sites and update them:
- `generate_continuation(... generator_pipeline=text_generator, tokenizer=tokenizer ...)` → `... backend=backend ...`
- `generate_args_from_brief_safe(args.creative_brief, text_generator, tokenizer, ...)` → `generate_args_from_brief_safe(args.creative_brief, backend, ...)`
- `generate_topic_variation(... generator_pipeline=text_generator, tokenizer=tokenizer ...)` → `... backend=backend ...`
- `generate_conversation(... generator_pipeline=text_generator, tokenizer=tokenizer ...)` → `... backend=backend ...`

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS — all 146 tests (the test_generation tests now use `make_mock_backend`, the rest are unaffected).

- [ ] **Step 6: Commit**

```bash
git add conversation_dataset_generator/generation.py conversation_dataset_generator/cli.py tests/test_generation.py
git commit -m "thread ChatBackend through generation entry points"
```

---

## Task 6: CLI flags — `--backend`, `--api-base-url`, `--api-key`

**Files:**
- Modify: `conversation_dataset_generator/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
class TestBackendFlags:
    def test_backend_default_is_hf(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.backend == "hf"

    def test_backend_can_be_openai(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--backend", "openai"])
        assert args.backend == "openai"

    def test_backend_rejects_unknown_choice(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--creative-brief", "test", "--backend", "vllm"])

    def test_api_base_url_default(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.api_base_url == "http://localhost:1234/v1"

    def test_api_base_url_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test",
            "--api-base-url", "http://localhost:11434/v1",
        ])
        assert args.api_base_url == "http://localhost:11434/v1"

    def test_api_key_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.api_key is None

    def test_api_key_custom(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--api-key", "sk-xyz"])
        assert args.api_key == "sk-xyz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestBackendFlags -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'backend'`

- [ ] **Step 3: Add the flags**

In `conversation_dataset_generator/cli.py`, in `build_parser()`, inside the existing `general` argument group (after `--role-mapping`), add:

```python
    general.add_argument(
        "--backend", type=str, default="hf", choices=["hf", "openai"],
        help="Inference backend: 'hf' (local transformers, default) or "
             "'openai' (OpenAI-compatible HTTP server like LM Studio or Ollama).",
    )
    general.add_argument(
        "--api-base-url", type=str, default="http://localhost:1234/v1",
        help="Base URL for the OpenAI-compatible server. "
             "LM Studio: http://localhost:1234/v1 (default). "
             "Ollama: http://localhost:11434/v1.",
    )
    general.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the OpenAI-compatible server. "
             "If unset, falls back to env OPENAI_API_KEY then 'not-needed'.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS — all 26+ tests (existing 19 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add conversation_dataset_generator/cli.py tests/test_cli.py
git commit -m "add --backend, --api-base-url, --api-key CLI flags"
```

---

## Task 7: `build_backend_from_args` helper + wire it into `main()`

**Files:**
- Modify: `conversation_dataset_generator/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
from unittest.mock import MagicMock, patch


class TestBuildBackendFromArgs:
    def test_hf_backend(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend_from_args
        from conversation_dataset_generator.backend import HFBackend

        fake_pipe = MagicMock()
        fake_tok = MagicMock()
        monkeypatch.setattr(
            "conversation_dataset_generator.cli.load_model_and_pipeline",
            lambda **kw: (fake_pipe, fake_tok),
        )

        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        backend = build_backend_from_args(args)
        assert isinstance(backend, HFBackend)

    def test_hf_backend_passes_quantization_flag(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend_from_args

        captured = {}
        def fake_load(**kw):
            captured.update(kw)
            return MagicMock(), MagicMock()
        monkeypatch.setattr(
            "conversation_dataset_generator.cli.load_model_and_pipeline", fake_load,
        )

        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--load-in-4bit"])
        build_backend_from_args(args)
        assert captured["load_in_4bit"] is True

    def test_openai_backend(self):
        from conversation_dataset_generator.cli import build_backend_from_args
        from conversation_dataset_generator.backend import OpenAIBackend

        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test", "--backend", "openai",
            "--api-base-url", "http://localhost:11434/v1",
            "--api-key", "test-key",
            "--model-id", "llama3.1",
        ])
        backend = build_backend_from_args(args)
        assert isinstance(backend, OpenAIBackend)
        assert backend.model_id == "llama3.1"

    def test_openai_backend_falls_back_to_env_var(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend_from_args
        from conversation_dataset_generator.backend import OpenAIBackend

        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test", "--backend", "openai",
        ])
        backend = build_backend_from_args(args)
        assert isinstance(backend, OpenAIBackend)
        # The OpenAI client receives the env var value; we trust the SDK to
        # pick it up. Here we just verify the backend was constructed.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestBuildBackendFromArgs -v`
Expected: FAIL — `ImportError: cannot import name 'build_backend_from_args'`

- [ ] **Step 3: Implement the helper and rewire `main()`**

In `conversation_dataset_generator/cli.py`:

1. Move the `from conversation_dataset_generator.models import load_model_and_pipeline` import to the top of the file (so it can be monkey-patched in tests).

```python
# Near the top of cli.py, with the other imports:
from conversation_dataset_generator.models import DEFAULT_MODEL_ID, load_model_and_pipeline
```

2. Add the helper above `main()`:

```python
def build_backend_from_args(args):
    """Construct a ChatBackend from parsed CLI args."""
    import os
    from conversation_dataset_generator.backend import make_backend

    if args.backend == "hf":
        pipeline, tokenizer = load_model_and_pipeline(
            model_id=args.model_id, load_in_4bit=args.load_in_4bit,
        )
        return make_backend("hf", pipeline=pipeline, tokenizer=tokenizer)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    return make_backend(
        "openai",
        model_id=args.model_id,
        base_url=args.api_base_url,
        api_key=api_key,
    )
```

3. Replace the model-loading block in `main()`:

Find this block:
```python
    # --- Load model ---
    from conversation_dataset_generator.models import load_model_and_pipeline

    text_generator, tokenizer = load_model_and_pipeline(
        model_id=args.model_id, load_in_4bit=args.load_in_4bit,
    )
```

Plus the temporary bridge added in Task 5:
```python
    from conversation_dataset_generator.backend import HFBackend
    backend = HFBackend(text_generator, tokenizer)
```

Replace with:
```python
    # --- Build backend ---
    backend = build_backend_from_args(args)
```

4. Remove the now-unused `text_generator`/`tokenizer` references from the dataset-card block at the bottom of `main()`. (They aren't used in the card today, but if any reference lingers, drop it.)

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS — all tests including the new `TestBuildBackendFromArgs` class.

- [ ] **Step 5: Smoke test against a real OpenAI-compatible server (manual, optional but recommended)**

If the developer has LM Studio or Ollama running locally:

```bash
# Ollama: pull a small model first if needed
ollama pull llama3.2:1b

python generate.py \
  --backend openai \
  --api-base-url http://localhost:11434/v1 \
  --model-id llama3.2:1b \
  --persona1 "Alice" --persona1-desc "A friendly engineer" \
  --persona2 "Bob"   --persona2-desc "A curious student" \
  --topic "the weather" --scenario "a coffee shop" --style "Casual" \
  --num-examples 1 \
  --output-file /tmp/smoke.jsonl
```

Expected: a single conversation written to `/tmp/smoke.jsonl`. If this fails, do not proceed — the test suite passing isn't sufficient evidence that the OpenAI path actually round-trips through a real server.

- [ ] **Step 6: Commit**

```bash
git add conversation_dataset_generator/cli.py tests/test_cli.py
git commit -m "wire ChatBackend factory through CLI"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md architecture table**

In `CLAUDE.md`, in the architecture table (under `### Package: conversation_dataset_generator/`), add a row before `web_search.py`:

```
| `backend.py` | ChatBackend protocol + HFBackend (transformers) and OpenAIBackend (LM Studio / Ollama / OpenAI) implementations |
```

Also update the "Key Details" section: change "Default model: `Qwen/Qwen2.5-7B-Instruct`" by adding a follow-up line:

```
- Backends: `--backend hf` (default, local transformers) or `--backend openai` (OpenAI-compatible HTTP server). For openai, set `--api-base-url` (LM Studio default `http://localhost:1234/v1`, Ollama `http://localhost:11434/v1`) and optionally `--api-key`.
```

- [ ] **Step 2: Update README**

In `README.md`, add a new section after the existing "Running" / setup section:

````markdown
## Using a remote OpenAI-compatible server (no local GPU needed)

You can drive `generate.py` against any OpenAI-compatible inference server — LM Studio, Ollama, vLLM, TGI, or the real OpenAI API. This sidesteps local CUDA and lets you use models bigger than your VRAM.

### LM Studio

Start the server in LM Studio (Server tab, default port 1234), load a model, then:

```bash
python generate.py \
  --backend openai \
  --api-base-url http://localhost:1234/v1 \
  --model-id "lmstudio-community/Qwen2.5-7B-Instruct-GGUF" \
  --creative-brief "Sherlock and Watson debate AI" \
  --num-examples 5 \
  --output-file out.jsonl
```

### Ollama

```bash
ollama pull llama3.2:1b   # or any model you like
python generate.py \
  --backend openai \
  --api-base-url http://localhost:11434/v1 \
  --model-id llama3.2:1b \
  --creative-brief "Two chefs argue about umami" \
  --num-examples 5 \
  --output-file out.jsonl
```

### OpenAI (or OpenRouter, Together, etc.)

```bash
export OPENAI_API_KEY=sk-...
python generate.py \
  --backend openai \
  --api-base-url https://api.openai.com/v1 \
  --model-id gpt-4o-mini \
  --creative-brief "..." --num-examples 5 \
  --output-file out.jsonl
```

When `--backend openai` is set, `--load-in-4bit` is silently ignored (quantization happens server-side). The default `--backend hf` preserves the original local-transformers behavior.
````

- [ ] **Step 3: Run the full test suite one more time**

Run: `pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "document --backend openai usage with LM Studio / Ollama"
```

---

## Self-Review Checklist (already done; for posterity)

**Spec coverage:**
- "Integrate with the OpenAI API to connect with LM Studio or Ollama" — covered by Tasks 3, 6, 7 (OpenAIBackend + flags + factory wiring).
- "Avoid GPU optimization headaches" — directly addressed: the openai backend has zero torch/transformers/cuda dependencies in its hot path.
- TDD discipline — every task starts with a failing test, verifies failure, implements minimal code, verifies pass, commits.

**Placeholder scan:** No "TBD", "implement later", or "similar to Task N" — every step contains the actual code.

**Type consistency:**
- `ChatBackend.complete` signature is identical at every reference (Task 1 protocol, Task 2 HFBackend, Task 3 OpenAIBackend, Task 5 generation.py call sites).
- `make_backend` kwargs (`pipeline`, `tokenizer`, `model_id`, `base_url`, `api_key`, `client`) match between Task 4 implementation and Task 7 caller.
- CLI attribute names (`args.backend`, `args.api_base_url`, `args.api_key`) match between Task 6 (parser definition) and Task 7 (consumer).

**Smoke-test gap:** Task 7 includes an optional manual smoke test against a real server. The unit test suite mocks the openai client; only a live run proves the wire-format compatibility (e.g., that LM Studio actually returns `choices[0].message.content` and not some variant). Strongly recommend running it before merge.

**Web UI (Idea 2 from issue #4):** Out of scope for this plan — separate plan to follow once this lands. The backend abstraction here is a prerequisite, since a hostable web UI needs a hostable backend.
