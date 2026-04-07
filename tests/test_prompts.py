import pytest
from conversation_dataset_generator.prompts import (
    build_conversation_messages,
    build_arg_generation_messages,
    build_variation_messages,
)


class TestBuildConversationMessages:
    def test_returns_two_messages(self):
        msgs = build_conversation_messages(
            topic="Weather", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Bus stop", style="Casual",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_personas_in_system_message(self):
        msgs = build_conversation_messages(
            topic="Weather", persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Bus stop", style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]
        assert "Friendly" in msgs[0]["content"]
        assert "Grumpy" in msgs[0]["content"]

    def test_topic_and_scenario_in_system(self):
        msgs = build_conversation_messages(
            topic="Quantum physics", persona1="A", persona2="B",
            persona1_desc="d1", persona2_desc="d2",
            scenario="Coffee shop", style="Educational",
        )
        assert "Quantum physics" in msgs[0]["content"]
        assert "Coffee shop" in msgs[0]["content"]

    def test_include_points_in_user_message(self):
        msgs = build_conversation_messages(
            topic="T", persona1="A", persona2="B",
            persona1_desc="d1", persona2_desc="d2",
            scenario="S", style="St",
            include_points="rain, sun, wind",
        )
        assert "rain" in msgs[1]["content"]

    def test_no_include_points(self):
        msgs = build_conversation_messages(
            topic="T", persona1="A", persona2="B",
            persona1_desc="d1", persona2_desc="d2",
            scenario="S", style="St",
        )
        assert len(msgs) == 2


class TestBuildArgGenerationMessages:
    def test_returns_two_messages(self):
        msgs = build_arg_generation_messages(brief="Sherlock meets Watson")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_brief_in_user_message(self):
        msgs = build_arg_generation_messages(brief="Sherlock meets Watson")
        assert "Sherlock meets Watson" in msgs[1]["content"]

    def test_web_context_appended(self):
        msgs = build_arg_generation_messages(
            brief="Test brief",
            persona1_context="Context about persona 1",
            persona2_context="Context about persona 2",
            search_term1="Term1",
            search_term2="Term2",
        )
        assert "Context about persona 1" in msgs[0]["content"]
        assert "Context about persona 2" in msgs[0]["content"]


class TestBuildVariationMessages:
    def test_returns_two_messages(self):
        msgs = build_variation_messages(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_personas_in_user_context(self):
        msgs = build_variation_messages(
            persona1="Sherlock", persona1_desc="Detective",
            persona2="Watson", persona2_desc="Doctor",
            initial_topic="Crime", initial_scenario="Baker Street",
            initial_style="Dramatic",
        )
        assert "Sherlock" in msgs[1]["content"]
        assert "Watson" in msgs[1]["content"]

    def test_original_brief_included(self):
        msgs = build_variation_messages(
            persona1="A", persona1_desc="d1",
            persona2="B", persona2_desc="d2",
            initial_topic="T", initial_scenario="S", initial_style="St",
            original_brief="Original brief text here",
        )
        assert "Original brief text here" in msgs[1]["content"]


class TestBuildConversationMessagesMulti:
    def test_three_speakers_in_system(self):
        msgs = build_conversation_messages(
            topic="Party planning",
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy"), ("Charlie", "Quiet")],
            scenario="Office", style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]
        assert "Charlie" in msgs[0]["content"]

    def test_legacy_two_speaker_still_works(self):
        msgs = build_conversation_messages(
            topic="Weather",
            persona1="Alice", persona2="Bob",
            persona1_desc="Friendly", persona2_desc="Grumpy",
            scenario="Bus stop", style="Casual",
        )
        assert "Alice" in msgs[0]["content"]
        assert "Bob" in msgs[0]["content"]


class TestBuildContinuationMessages:
    def test_returns_two_messages(self):
        from conversation_dataset_generator.prompts import build_continuation_messages
        prior_turns = [
            {"from": "human", "value": "Hello", "speaker_name": "Alice"},
            {"from": "gpt", "value": "Hi there", "speaker_name": "Bob"},
        ]
        msgs = build_continuation_messages(
            personas=[("Alice", "Friendly"), ("Bob", "Grumpy")],
            prior_turns=prior_turns,
            topic="Greeting", scenario="Online", style="Casual",
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_prior_turns_in_prompt(self):
        from conversation_dataset_generator.prompts import build_continuation_messages
        prior_turns = [
            {"from": "human", "value": "What about quantum?", "speaker_name": "Alice"},
        ]
        msgs = build_continuation_messages(
            personas=[("Alice", "Scientist"), ("Bob", "Student")],
            prior_turns=prior_turns,
            topic="Physics", scenario="Lab", style="Educational",
        )
        assert "What about quantum?" in msgs[1]["content"]
