import pytest
from conversation_dataset_generator.cli import (
    build_parser,
    build_role_mapping,
    detect_mode,
    load_personas_from_yaml,
)


class TestBuildParser:
    def test_default_model(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.model_id == "Qwen/Qwen2.5-7B-Instruct"

    def test_default_max_tokens(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test brief"])
        assert args.max_new_tokens == 4096

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


class TestNewFlags:
    def test_persona_flag_repeatable(self):
        parser = build_parser()
        args = parser.parse_args([
            "--persona", "Alice", "Friendly",
            "--persona", "Bob", "Grumpy",
            "--topic", "T", "--scenario", "S", "--style", "St",
        ])
        assert args.persona == [["Alice", "Friendly"], ["Bob", "Grumpy"]]

    def test_personas_file_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--personas", "chars.yaml",
            "--topic", "T", "--scenario", "S", "--style", "St",
        ])
        assert args.personas == "chars.yaml"

    def test_train_speaker_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test", "--train-speaker", "Captain America"])
        assert args.train_speaker == "Captain America"

    def test_continue_from_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl"])
        assert args.continue_from == "data.jsonl"

    def test_conversation_id_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl", "--conversation-id", "5"])
        assert args.conversation_id == 5

    def test_group_size_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--random-pairings",
            "--character-pool", "c.yaml", "--persona-desc-pool", "d.yaml",
            "--initial-topic", "T", "--initial-scenario", "S", "--initial-style", "St",
            "--group-size", "3",
        ])
        assert args.group_size == 3

    def test_group_size_default_is_two(self):
        parser = build_parser()
        args = parser.parse_args(["--creative-brief", "test"])
        assert args.group_size == 2


class TestDetectModeContinue:
    def test_continue_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl"])
        mode = detect_mode(args, parser)
        assert mode == "continue"

    def test_continue_with_conversation_id(self):
        parser = build_parser()
        args = parser.parse_args(["--continue-from", "data.jsonl", "--conversation-id", "3"])
        mode = detect_mode(args, parser)
        assert mode == "continue"


class TestBuildRoleMapping:
    def test_default_first_human_rest_gpt(self):
        result = build_role_mapping(["Alice", "Bob", "Charlie"])
        assert result == {"Alice": "human", "Bob": "gpt", "Charlie": "gpt"}

    def test_train_speaker(self):
        result = build_role_mapping(["Alice", "Bob", "Charlie"], train_speaker="Bob")
        assert result == {"Alice": "human", "Bob": "gpt", "Charlie": "human"}

    def test_role_mapping_str(self):
        result = build_role_mapping(
            ["Alice", "Bob"], role_mapping_str="Alice=gpt,Bob=human"
        )
        assert result == {"Alice": "gpt", "Bob": "human"}

    def test_two_speakers_default(self):
        result = build_role_mapping(["Alice", "Bob"])
        assert result == {"Alice": "human", "Bob": "gpt"}


class TestLoadPersonasFromYaml:
    def test_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "personas.yaml"
        yaml_file.write_text(
            "personas:\n"
            "  - name: Alice\n"
            "    description: Friendly\n"
            "  - name: Bob\n"
            "    description: Grumpy\n"
        )
        result = load_personas_from_yaml(str(yaml_file))
        assert result == [("Alice", "Friendly"), ("Bob", "Grumpy")]

    def test_invalid_yaml_missing_personas_key(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("characters:\n  - name: X\n")
        with pytest.raises(ValueError):
            load_personas_from_yaml(str(yaml_file))

    def test_invalid_yaml_missing_fields(self, tmp_path):
        yaml_file = tmp_path / "bad2.yaml"
        yaml_file.write_text("personas:\n  - name: X\n")
        with pytest.raises(ValueError):
            load_personas_from_yaml(str(yaml_file))


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
        assert backend._client.api_key == "env-key"

    def test_openai_backend_explicit_key_overrides_env(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend_from_args
        from conversation_dataset_generator.backend import OpenAIBackend

        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test", "--backend", "openai",
            "--api-key", "explicit-key",
        ])
        backend = build_backend_from_args(args)
        assert isinstance(backend, OpenAIBackend)
        assert backend._client.api_key == "explicit-key"

    def test_openai_backend_warns_when_model_id_is_hf_default(self, caplog):
        from conversation_dataset_generator.cli import build_backend_from_args
        parser = build_parser()
        args = parser.parse_args([
            "--creative-brief", "test", "--backend", "openai",
        ])
        with caplog.at_level("WARNING"):
            build_backend_from_args(args)
        assert any("model-id" in r.message.lower() for r in caplog.records)


class TestBuildBackend:
    """Direct tests for build_backend(), the args-free helper.

    build_backend() exists so the web UI (and any future caller) can construct
    a backend from raw values without needing to fabricate an argparse Namespace.
    build_backend_from_args() is now a thin wrapper around it.
    """

    def test_hf_kind(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend
        from conversation_dataset_generator.backend import HFBackend

        fake_pipe = MagicMock()
        fake_tok = MagicMock()
        monkeypatch.setattr(
            "conversation_dataset_generator.cli.load_model_and_pipeline",
            lambda **kw: (fake_pipe, fake_tok),
        )
        backend = build_backend(kind="hf", model_id="qwen-7b")
        assert isinstance(backend, HFBackend)

    def test_hf_kind_passes_load_in_4bit(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend

        captured = {}
        def fake_load(**kw):
            captured.update(kw)
            return MagicMock(), MagicMock()
        monkeypatch.setattr(
            "conversation_dataset_generator.cli.load_model_and_pipeline", fake_load,
        )
        build_backend(kind="hf", model_id="m", load_in_4bit=True)
        assert captured["load_in_4bit"] is True

    def test_openai_kind(self):
        from conversation_dataset_generator.cli import build_backend
        from conversation_dataset_generator.backend import OpenAIBackend

        backend = build_backend(
            kind="openai",
            model_id="llama3.2:1b",
            api_base_url="http://localhost:11434/v1",
            api_key="test-key",
        )
        assert isinstance(backend, OpenAIBackend)
        assert backend.model_id == "llama3.2:1b"
        assert backend._client.api_key == "test-key"

    def test_openai_kind_uses_env_var_when_key_omitted(self, monkeypatch):
        from conversation_dataset_generator.cli import build_backend
        from conversation_dataset_generator.backend import OpenAIBackend

        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        backend = build_backend(
            kind="openai",
            model_id="m",
            api_base_url="http://x/v1",
        )
        assert isinstance(backend, OpenAIBackend)
        assert backend._client.api_key == "env-key"
