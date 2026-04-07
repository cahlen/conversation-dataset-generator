"""CLI module: argument parsing, mode detection, and orchestration."""

import argparse
import logging
import os

from conversation_dataset_generator.models import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


def parse_role_mapping(mapping_str):
    """Parse a role mapping string like 'p1=human,p2=gpt' into a dict.

    Returns:
        dict mapping persona keys to role names.

    Raises:
        ValueError: If the mapping string is invalid.
    """
    if not mapping_str:
        return {}
    result = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise ValueError(
                f"Invalid role mapping entry '{pair}': expected 'key=value' format"
            )
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid role mapping entry '{pair}': key and value must be non-empty"
            )
        result[key] = value
    return result


def build_parser():
    """Create the argparse parser with all argument groups."""
    parser = argparse.ArgumentParser(
        description="Generate conversation datasets for fine-tuning."
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--creative-brief",
        type=str,
        default=None,
        help="A creative brief to generate conversation parameters from.",
    )
    mode_group.add_argument(
        "--enable-variation",
        action="store_true",
        default=False,
        help="Enable topic/scenario/style variation between conversations.",
    )

    # Random pairings (standalone group)
    random_group = parser.add_argument_group("Random Pairings")
    random_group.add_argument(
        "--random-pairings",
        action="store_true",
        default=False,
        help="Use random character pairings from a pool.",
    )
    random_group.add_argument(
        "--character-pool",
        type=str,
        default=None,
        help="Path to character pool YAML file.",
    )
    random_group.add_argument(
        "--persona-desc-pool",
        type=str,
        default=None,
        help="Path to persona description pool YAML file.",
    )

    # Manual mode arguments
    manual_group = parser.add_argument_group("Manual Mode")
    manual_group.add_argument("--topic", type=str, default=None)
    manual_group.add_argument("--persona1", type=str, default=None)
    manual_group.add_argument("--persona1-desc", type=str, default=None)
    manual_group.add_argument("--persona2", type=str, default=None)
    manual_group.add_argument("--persona2-desc", type=str, default=None)
    manual_group.add_argument("--scenario", type=str, default=None)
    manual_group.add_argument("--style", type=str, default=None)

    # Fixed persona variation arguments
    fixed_group = parser.add_argument_group("Fixed Persona Variation")
    fixed_group.add_argument("--fixed-persona1", type=str, default=None)
    fixed_group.add_argument("--fixed-persona1-desc", type=str, default=None)
    fixed_group.add_argument("--fixed-persona2", type=str, default=None)
    fixed_group.add_argument("--fixed-persona2-desc", type=str, default=None)
    fixed_group.add_argument("--initial-topic", type=str, default=None)
    fixed_group.add_argument("--initial-scenario", type=str, default=None)
    fixed_group.add_argument("--initial-style", type=str, default=None)

    # Brief context
    brief_group = parser.add_argument_group("Brief Context")
    brief_group.add_argument("--brief-context", type=str, default=None)

    # General arguments
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to use (default: {DEFAULT_MODEL_ID}).",
    )
    general_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048).",
    )
    general_group.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of conversation examples to generate.",
    )
    general_group.add_argument(
        "--output",
        type=str,
        default="conversations.jsonl",
        help="Output JSONL file path.",
    )
    general_group.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo to upload to.",
    )
    general_group.add_argument(
        "--role-mapping",
        type=str,
        default=None,
        help="Role mapping string, e.g. 'p1=gpt,p2=human'.",
    )
    general_group.add_argument(
        "--num-turns",
        type=int,
        default=5,
        help="Number of conversation turns.",
    )

    return parser


def _require(args, keys, parser):
    """Check that required arg keys are not None; call parser.error() if missing."""
    missing = [k for k in keys if getattr(args, k.replace("-", "_"), None) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join('--' + k for k in missing)}")


def detect_mode(args, parser):
    """Determine the mode from parsed args.

    Priority: brief -> random_pairings+variation -> random_pairings ->
              fixed_persona_variation -> manual
    """
    if args.creative_brief:
        return "brief"

    if args.random_pairings and args.enable_variation:
        _require(args, [
            "character-pool", "persona-desc-pool",
            "initial-topic", "initial-scenario", "initial-style",
        ], parser)
        return "random_pairings_variation"

    if args.random_pairings:
        _require(args, [
            "character-pool", "persona-desc-pool",
            "initial-topic", "initial-scenario", "initial-style",
        ], parser)
        return "random_pairings"

    if args.enable_variation:
        _require(args, [
            "fixed-persona1", "fixed-persona1-desc",
            "fixed-persona2", "fixed-persona2-desc",
            "initial-topic", "initial-scenario", "initial-style",
        ], parser)
        return "fixed_persona_variation"

    # Manual mode
    _require(args, [
        "topic", "persona1", "persona1-desc",
        "persona2", "persona2-desc", "scenario", "style",
    ], parser)
    return "manual"


def main():
    """Main orchestration function."""
    logging.basicConfig(level=logging.INFO)

    parser = build_parser()
    args = parser.parse_args()
    mode = detect_mode(args, parser)

    # Parse role mapping
    role_mapping = None
    if args.role_mapping:
        role_mapping = parse_role_mapping(args.role_mapping)

    # Import heavy dependencies only when running main
    from conversation_dataset_generator.models import load_model_and_pipeline
    from conversation_dataset_generator.generation import (
        generate_args_from_brief_safe,
        generate_conversation,
        generate_topic_variation,
    )
    from conversation_dataset_generator.output import (
        write_jsonl,
        build_dataset_card,
        create_hf_dataset,
    )
    from conversation_dataset_generator.hub import upload_to_hub

    from tqdm import tqdm

    logger.info("Loading model and pipeline...")
    pipe = load_model_and_pipeline(model_id=args.model_id, max_new_tokens=args.max_new_tokens)

    # Determine base args per mode
    if mode == "brief":
        base_args = generate_args_from_brief_safe(pipe, args.creative_brief)
    elif mode == "manual":
        base_args = {
            "topic": args.topic,
            "persona1": args.persona1,
            "persona1_desc": args.persona1_desc,
            "persona2": args.persona2,
            "persona2_desc": args.persona2_desc,
            "scenario": args.scenario,
            "style": args.style,
        }
    elif mode in ("fixed_persona_variation", "random_pairings", "random_pairings_variation"):
        base_args = {
            "topic": args.initial_topic,
            "scenario": args.initial_scenario,
            "style": args.initial_style,
        }
        if mode == "fixed_persona_variation":
            base_args.update({
                "persona1": args.fixed_persona1,
                "persona1_desc": args.fixed_persona1_desc,
                "persona2": args.fixed_persona2,
                "persona2_desc": args.fixed_persona2_desc,
            })
        if mode in ("random_pairings", "random_pairings_variation"):
            from conversation_dataset_generator.character_pool import (
                load_character_pool,
                load_description_pool,
                validate_pools,
                select_random_pair,
            )
            # Prepend character-config/ if path is not absolute and doesn't start with it
            char_pool_path = args.character_pool
            desc_pool_path = args.persona_desc_pool
            if not os.path.isabs(char_pool_path) and not char_pool_path.startswith("character-config/"):
                char_pool_path = os.path.join("character-config", char_pool_path)
            if not os.path.isabs(desc_pool_path) and not desc_pool_path.startswith("character-config/"):
                desc_pool_path = os.path.join("character-config", desc_pool_path)

            character_pool = load_character_pool(char_pool_path)
            description_pool = load_description_pool(desc_pool_path)
            validate_pools(character_pool, description_pool)

    # Generation loop
    results = []
    for i in tqdm(range(args.num_examples), desc="Generating conversations"):
        conv_args = dict(base_args)

        # Random pairings: select a random pair
        if mode in ("random_pairings", "random_pairings_variation"):
            pair = select_random_pair(character_pool, description_pool)
            conv_args.update({
                "persona1": pair["persona1"],
                "persona1_desc": pair["persona1_desc"],
                "persona2": pair["persona2"],
                "persona2_desc": pair["persona2_desc"],
            })

        # Variation: generate new topic/scenario/style
        if mode in ("fixed_persona_variation", "random_pairings_variation"):
            variation = generate_topic_variation(
                pipe,
                conv_args.get("topic", ""),
                conv_args.get("scenario", ""),
                conv_args.get("style", ""),
            )
            conv_args.update(variation)

        # Generate conversation
        conversation = generate_conversation(
            pipe,
            topic=conv_args["topic"],
            persona1=conv_args["persona1"],
            persona1_desc=conv_args["persona1_desc"],
            persona2=conv_args["persona2"],
            persona2_desc=conv_args["persona2_desc"],
            scenario=conv_args["scenario"],
            style=conv_args["style"],
            num_turns=args.num_turns,
            role_mapping=role_mapping,
        )
        results.append(conversation)

    # Write output
    write_jsonl(results, args.output)
    logger.info("Wrote %d conversations to %s", len(results), args.output)

    # Optional HF upload
    if args.hf_repo:
        dataset_card = build_dataset_card(args.hf_repo, mode, len(results))
        hf_dataset = create_hf_dataset(results)
        upload_to_hub(hf_dataset, args.hf_repo, dataset_card)
        logger.info("Uploaded dataset to %s", args.hf_repo)

    logger.info("Summary: generated %d conversations in '%s' mode", len(results), mode)


if __name__ == "__main__":
    main()
