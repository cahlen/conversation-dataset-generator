import json
import os
import pytest
from conversation_dataset_generator.output import write_jsonl, build_dataset_card


class TestWriteJsonl:
    def test_writes_correct_structure(self, tmp_path):
        conversations = [
            {
                "turns": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there"},
                ],
                "topic": "Greeting",
                "scenario": "Online chat",
                "style": "Casual",
                "include_points": "",
                "persona1_name": "Alice",
                "persona2_name": "Bob",
            }
        ]
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl(conversations, outfile)
        assert count == 2

        with open(outfile, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

        row0 = json.loads(lines[0])
        assert row0["conversation_id"] == 0
        assert row0["turn_number"] == 0
        assert row0["role"] == "human"
        assert row0["speaker_name"] == "Alice"
        assert row0["content"] == "Hello"

        row1 = json.loads(lines[1])
        assert row1["conversation_id"] == 0
        assert row1["turn_number"] == 1
        assert row1["role"] == "gpt"
        assert row1["speaker_name"] == "Bob"

    def test_multiple_conversations(self, tmp_path):
        conversations = [
            {
                "turns": [{"from": "human", "value": "Hi"}],
                "topic": "T1", "scenario": "S1", "style": "St1",
                "include_points": "", "persona1_name": "A", "persona2_name": "B",
            },
            {
                "turns": [{"from": "gpt", "value": "Hey"}],
                "topic": "T2", "scenario": "S2", "style": "St2",
                "include_points": "", "persona1_name": "A", "persona2_name": "B",
            },
        ]
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl(conversations, outfile)
        assert count == 2

        with open(outfile, "r") as f:
            lines = f.readlines()
        row0 = json.loads(lines[0])
        row1 = json.loads(lines[1])
        assert row0["conversation_id"] == 0
        assert row1["conversation_id"] == 1

    def test_empty_conversations(self, tmp_path):
        outfile = str(tmp_path / "out.jsonl")
        count = write_jsonl([], outfile)
        assert count == 0
        assert not os.path.exists(outfile)


class TestBuildDatasetCard:
    def test_manual_mode_card(self):
        card = build_dataset_card(
            mode="manual", num_requested=10, num_generated=8, total_turns=64,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            persona1="Alice", persona1_desc="Friendly",
            persona2="Bob", persona2_desc="Grumpy",
            topic="Weather", scenario="Bus stop", style="Casual",
        )
        assert "Alice" in card
        assert "Bob" in card
        assert "Manual" in card
        assert "Qwen/Qwen2.5-7B-Instruct" in card
        assert "---" in card

    def test_brief_mode_card(self):
        card = build_dataset_card(
            mode="brief", num_requested=10, num_generated=10, total_turns=100,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            persona1="Sherlock", persona1_desc="Detective",
            persona2="Watson", persona2_desc="Doctor",
            topic="Crime", scenario="Baker Street", style="Dramatic",
            creative_brief="Sherlock and Watson discuss a case",
        )
        assert "Creative Brief" in card
        assert "Sherlock and Watson discuss a case" in card

    def test_random_pairings_card(self):
        card = build_dataset_card(
            mode="random_pairings", num_requested=5, num_generated=5, total_turns=40,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            topic="Tech", scenario="Office", style="Professional",
            character_pool=["Alice", "Bob", "Charlie"],
            character_descriptions={"Alice": "Friendly", "Bob": "Grumpy", "Charlie": "Quiet"},
        )
        assert "Character Pool" in card
        assert "Alice" in card
        assert "Bob" in card
        assert "Charlie" in card

    def test_card_has_yaml_frontmatter(self):
        card = build_dataset_card(
            mode="manual", num_requested=1, num_generated=1, total_turns=2,
            model_id="test",
            persona1="A", persona1_desc="d", persona2="B", persona2_desc="d",
            topic="T", scenario="S", style="St",
        )
        assert card.startswith("---\n")
        assert "license:" in card
        assert "tags:" in card
