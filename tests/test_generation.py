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
