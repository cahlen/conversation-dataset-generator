import pytest
from conversation_dataset_generator.cli import build_parser, detect_mode


class TestBuildParser:
    def test_default_model(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"

    def test_default_max_tokens(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.max_new_tokens == 2048

    def test_default_output_file(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.output_file == "generated_data.jsonl"

    def test_default_num_examples(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.num_examples == 3

    def test_no_delete_repo_flag(self):
        parser = build_parser()
        assert not hasattr(parser.parse_args(["--creative-brief", "test"]), "delete_repo")

    def test_has_load_in_4bit(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--load-in-4bit"])
        assert args.load_in_4bit is True

    def test_has_upload_to_hub(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--upload-to-hub", "user/repo"])
        assert args.upload_to_hub == "user/repo"

    def test_has_include_points(self):
        parser = build_parser()
        args = parser.parse_args([
            "--topic", "T", "--persona1", "A", "--persona1-desc", "d",
            "--persona2", "B", "--persona2-desc", "d",
            "--scenario", "S", "--style", "St",
            "--include-points", "rain, sun",
        ])
        assert args.include_points == "rain, sun"


class TestDetectMode:
    def test_brief_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        mode = detect_mode(args, parser)
        assert mode == "brief"

    def test_manual_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--topic", "T", "--persona1", "A", "--persona1-desc", "d1",
            "--persona2", "B", "--persona2-desc", "d2",
            "--scenario", "S", "--style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "manual"

    def test_fixed_persona_variation_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--enable-variation",
            "--fixed-persona1", "A", "--fixed-persona1-desc", "d1",
            "--fixed-persona2", "B", "--fixed-persona2-desc", "d2",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "fixed_persona_variation"

    def test_random_pairings_mode(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings",
            "--character-pool", "chars.yaml",
            "--persona-desc-pool", "descs.yaml",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "random_pairings"

    def test_random_pairings_with_variation(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings", "--enable-variation",
            "--character-pool", "chars.yaml",
            "--persona-desc-pool", "descs.yaml",
            "--initial-topic", "T", "--initial-scenario", "S",
            "--initial-style", "St",
        ])
        mode = detect_mode(args, parser)
        assert mode == "random_pairings_variation"

    def test_role_mapping_default(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.role_mapping is None

    def test_role_mapping_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test",
            "--role-mapping", "p1=gpt,p2=human",
        ])
        assert args.role_mapping == "p1=gpt,p2=human"
