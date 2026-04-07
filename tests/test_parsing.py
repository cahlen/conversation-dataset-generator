import pytest
from conversation_dataset_generator.parsing import parse_conversation_to_sharegpt


class TestParseConversationToSharegpt:
    def test_basic_two_turn(self):
        text = "Alice: Hello there!\nBob: Hi Alice, how are you?"
        turns, names = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert turns == [
            {"from": "human", "value": "Hello there!"},
            {"from": "gpt", "value": "Hi Alice, how are you?"},
        ]
        assert names == ["Alice", "Bob"]

    def test_multi_turn(self):
        text = (
            "Alice: First line\n"
            "Bob: Second line\n"
            "Alice: Third line\n"
            "Bob: Fourth line"
        )
        turns, names = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 4
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "human"
        assert turns[3]["from"] == "gpt"

    def test_multiline_turn(self):
        text = (
            "Alice: This is line one\n"
            "and this continues the same turn\n"
            "Bob: Got it"
        )
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert "line one\nand this continues" in turns[0]["value"]

    def test_case_insensitive_matching(self):
        text = "alice: Hello\nBOB: Hi"
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2

    def test_empty_text_returns_none(self):
        turns, names = parse_conversation_to_sharegpt("", "Alice", "Bob")
        assert turns is None
        assert names is None

    def test_no_matching_speakers_returns_none(self):
        text = "Charlie: Hello\nDave: Hi"
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert turns is None

    def test_blank_lines_ignored(self):
        text = "Alice: Hello\n\n\nBob: Hi"
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2

    def test_trailing_whitespace_on_lines(self):
        text = "Alice: Hello   \nBob: Hi   "
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert turns[0]["value"] == "Hello"
        assert turns[1]["value"] == "Hi"

    def test_custom_role_mapping(self):
        text = "Alice: Hello\nBob: Hi"
        turns, _ = parse_conversation_to_sharegpt(
            text, "Alice", "Bob", role_mapping={"p1": "gpt", "p2": "human"}
        )
        assert turns[0]["from"] == "gpt"
        assert turns[1]["from"] == "human"

    def test_custom_role_mapping_by_name(self):
        text = "Alice: Hello\nBob: Hi"
        turns, _ = parse_conversation_to_sharegpt(
            text, "Alice", "Bob", role_mapping={"Alice": "gpt", "Bob": "human"}
        )
        assert turns[0]["from"] == "gpt"
        assert turns[1]["from"] == "human"

    def test_skips_empty_turns(self):
        text = "Alice: \nBob: Hi there"
        turns, _ = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 1
        assert turns[0]["from"] == "gpt"


class TestParseConversationMultiSpeaker:
    def test_three_speakers(self):
        text = "Alice: Hello\nBob: Hi\nCharlie: Hey everyone"
        turns, names = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
            role_mapping={"Alice": "human", "Bob": "gpt", "Charlie": "gpt"},
        )
        assert len(turns) == 3
        assert turns[0] == {"from": "human", "value": "Hello"}
        assert turns[1] == {"from": "gpt", "value": "Hi"}
        assert turns[2] == {"from": "gpt", "value": "Hey everyone"}
        assert names == ["Alice", "Bob", "Charlie"]

    def test_four_speakers(self):
        text = "A: one\nB: two\nC: three\nD: four"
        turns, names = parse_conversation_to_sharegpt(
            text, personas=["A", "B", "C", "D"],
            role_mapping={"A": "human", "B": "gpt", "C": "gpt", "D": "gpt"},
        )
        assert len(turns) == 4

    def test_default_role_mapping_multi(self):
        text = "Alice: Hello\nBob: Hi\nCharlie: Hey"
        turns, _ = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
        )
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "gpt"

    def test_legacy_two_args_still_work(self):
        text = "Alice: Hello\nBob: Hi"
        turns, names = parse_conversation_to_sharegpt(text, "Alice", "Bob")
        assert len(turns) == 2
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert names == ["Alice", "Bob"]

    def test_train_speaker_mapping(self):
        text = "Alice: Hi\nBob: Hello\nCharlie: Hey"
        mapping = {"Alice": "human", "Bob": "gpt", "Charlie": "human"}
        turns, _ = parse_conversation_to_sharegpt(
            text, personas=["Alice", "Bob", "Charlie"],
            role_mapping=mapping,
        )
        assert turns[0]["from"] == "human"
        assert turns[1]["from"] == "gpt"
        assert turns[2]["from"] == "human"


from conversation_dataset_generator.parsing import parse_variation_output


class TestParseVariationOutput:
    def test_basic_three_fields(self):
        text = (
            '--topic "New topic here"\n'
            '--scenario "New scenario here"\n'
            '--style "New style here"'
        )
        result = parse_variation_output(text)
        assert result == {
            "topic": "New topic here",
            "scenario": "New scenario here",
            "style": "New style here",
        }

    def test_trailing_whitespace_on_lines(self):
        text = (
            '--topic "Topic value"   \n'
            '--scenario "Scenario value"  \n'
            '--style "Style value"  '
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Topic value"
        assert result["scenario"] == "Scenario value"
        assert result["style"] == "Style value"

    def test_single_quotes(self):
        text = (
            "--topic 'Single quoted topic'\n"
            "--scenario 'Single quoted scenario'\n"
            "--style 'Single quoted style'"
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Single quoted topic"

    def test_mixed_quotes(self):
        text = (
            '--topic "Double quoted"\n'
            "--scenario 'Single quoted'\n"
            '--style "Another double"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert len(result) == 3

    def test_missing_topic_returns_none(self):
        text = (
            '--scenario "Only scenario"\n'
            '--style "Only style"'
        )
        result = parse_variation_output(text)
        assert result is None

    def test_missing_scenario_returns_none(self):
        text = (
            '--topic "Only topic"\n'
            '--style "Only style"'
        )
        result = parse_variation_output(text)
        assert result is None

    def test_style_optional(self):
        text = (
            '--topic "Just topic"\n'
            '--scenario "Just scenario"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert "style" not in result

    def test_preamble_text_ignored(self):
        text = (
            "Here are the new parameters:\n"
            '--topic "Actual topic"\n'
            '--scenario "Actual scenario"\n'
            '--style "Actual style"'
        )
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "Actual topic"

    def test_empty_string_returns_none(self):
        result = parse_variation_output("")
        assert result is None

    def test_garbage_input_returns_none(self):
        result = parse_variation_output("This is just random text with no args")
        assert result is None

    def test_all_args_on_one_line(self):
        text = '--topic "New topic" --scenario "New scenario" --style "New style"'
        result = parse_variation_output(text)
        assert result is not None
        assert result["topic"] == "New topic"
        assert result["scenario"] == "New scenario"
        assert result["style"] == "New style"


from conversation_dataset_generator.parsing import parse_arg_generation_output


class TestParseArgGenerationOutput:
    def test_all_required_fields(self):
        text = (
            '--persona1 "Alice"\n'
            '--persona1-desc "A friendly person"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "A grumpy person"\n'
            '--topic "Weather"\n'
            '--scenario "At a bus stop"\n'
            '--style "Casual chat"'
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["persona1"] == "Alice"
        assert result["persona1_desc"] == "A friendly person"
        assert result["persona2"] == "Bob"
        assert result["persona2_desc"] == "A grumpy person"
        assert result["topic"] == "Weather"
        assert result["scenario"] == "At a bus stop"
        assert result["style"] == "Casual chat"

    def test_with_include_points(self):
        text = (
            '--persona1 "Alice"\n'
            '--persona1-desc "Desc"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "Desc"\n'
            '--topic "Topic"\n'
            '--scenario "Scenario"\n'
            '--style "Style"\n'
            '--include-points "rain, sun, wind"'
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["include_points"] == "rain, sun, wind"

    def test_trailing_whitespace(self):
        text = (
            '--persona1 "Alice"   \n'
            '--persona1-desc "Desc"   \n'
            '--persona2 "Bob"   \n'
            '--persona2-desc "Desc"   \n'
            '--topic "Topic"   \n'
            '--scenario "Scenario"   \n'
            '--style "Style"   '
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["persona1"] == "Alice"

    def test_single_quotes(self):
        text = (
            "--persona1 'Alice'\n"
            "--persona1-desc 'Desc'\n"
            "--persona2 'Bob'\n"
            "--persona2-desc 'Desc'\n"
            "--topic 'Topic'\n"
            "--scenario 'Scenario'\n"
            "--style 'Style'"
        )
        result = parse_arg_generation_output(text)
        assert result is not None

    def test_missing_persona1_returns_none(self):
        text = (
            '--persona1-desc "Desc"\n'
            '--persona2 "Bob"\n'
            '--persona2-desc "Desc"\n'
            '--topic "Topic"\n'
            '--scenario "Scenario"\n'
            '--style "Style"'
        )
        result = parse_arg_generation_output(text)
        assert result is None

    def test_empty_returns_none(self):
        result = parse_arg_generation_output("")
        assert result is None

    def test_all_args_on_one_line(self):
        text = (
            '--persona1 "Sherlock Holmes" --persona1-desc "Brilliant detective" '
            '--persona2 "Dr Watson" --persona2-desc "Loyal doctor" '
            '--topic "AI replacing detectives" --scenario "221B Baker Street" '
            '--style "Formal debate"'
        )
        result = parse_arg_generation_output(text)
        assert result is not None
        assert result["persona1"] == "Sherlock Holmes"
        assert result["persona2_desc"] == "Loyal doctor"
        assert result["topic"] == "AI replacing detectives"
