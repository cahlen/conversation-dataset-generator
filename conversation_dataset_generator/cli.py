"""Command-line interface and orchestration for conversation generation."""

import argparse
import logging
import os
import sys
import time

from tqdm import tqdm

from conversation_dataset_generator.models import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


def parse_role_mapping(mapping_str: str | None) -> dict:
    """Parse role mapping string like 'p1=human,p2=gpt' into a dict."""
    if mapping_str is None:
        return {"p1": "human", "p2": "gpt"}

    result = {}
    for part in mapping_str.split(","):
        key, _, value = part.strip().partition("=")
        if key in ("p1", "p2") and value in ("human", "gpt"):
            result[key] = value

    if "p1" not in result or "p2" not in result:
        raise ValueError(
            f"Invalid role mapping: '{mapping_str}'. "
            "Expected format: 'p1=human,p2=gpt' or 'p1=gpt,p2=human'"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all generation modes."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversational data for LLM fine-tuning.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--creative-brief", type=str,
        help="High-level brief for automatic argument generation + topic variation.",
    )
    mode_group.add_argument(
        "--enable-variation", action="store_true",
        help="Enable topic/scenario/style variation between conversations.",
    )
    mode_group.add_argument(
        "--random-pairings", action="store_true",
        help="Enable random pairing mode using character pools.",
    )

    # Manual mode args
    manual = parser.add_argument_group("Manual Mode")
    manual.add_argument("--topic", type=str)
    manual.add_argument("--persona1", type=str)
    manual.add_argument("--persona1-desc", type=str)
    manual.add_argument("--persona2", type=str)
    manual.add_argument("--persona2-desc", type=str)
    manual.add_argument("--scenario", type=str)
    manual.add_argument("--style", type=str)
    manual.add_argument("--include-points", type=str, default=None)

    # Fixed persona variation args
    fixed = parser.add_argument_group("Fixed Persona Variation Mode")
    fixed.add_argument("--fixed-persona1", type=str)
    fixed.add_argument("--fixed-persona1-desc", type=str)
    fixed.add_argument("--fixed-persona2", type=str)
    fixed.add_argument("--fixed-persona2-desc", type=str)
    fixed.add_argument("--initial-topic", type=str)
    fixed.add_argument("--initial-scenario", type=str)
    fixed.add_argument("--initial-style", type=str)

    # Random pairings args
    pool = parser.add_argument_group("Random Pairings Mode")
    pool.add_argument("--character-pool", type=str)
    pool.add_argument("--persona-desc-pool", type=str)

    # Brief context args
    brief_ctx = parser.add_argument_group("Creative Brief Web Context")
    brief_ctx.add_argument("--persona1-search-term", type=str, default=None)
    brief_ctx.add_argument("--persona2-search-term", type=str, default=None)

    # General args
    general = parser.add_argument_group("General")
    general.add_argument("--num-examples", type=int, default=3)
    general.add_argument("--output-file", type=str, default="generated_data.jsonl")
    general.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    general.add_argument("--max-new-tokens", type=int, default=2048)
    general.add_argument("--upload-to-hub", type=str, default=None, metavar="REPO_ID")
    general.add_argument("--load-in-4bit", action="store_true")
    general.add_argument("--force-upload", action="store_true")
    general.add_argument("--role-mapping", type=str, default=None,
                         help="Role mapping: 'p1=human,p2=gpt' or 'p1=gpt,p2=human'")

    return parser


def detect_mode(args, parser) -> str:
    """Determine the generation mode from parsed arguments."""
    if args.creative_brief:
        return "brief"

    if args.enable_variation and args.random_pairings:
        _require(args, ["character_pool", "persona_desc_pool",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "random_pairings_variation"

    if args.enable_variation:
        _require(args, ["fixed_persona1", "fixed_persona1_desc",
                        "fixed_persona2", "fixed_persona2_desc",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "fixed_persona_variation"

    if args.random_pairings:
        _require(args, ["character_pool", "persona_desc_pool",
                        "initial_topic", "initial_scenario", "initial_style"], parser)
        return "random_pairings"

    if args.persona1 and args.topic:
        _require(args, ["persona1", "persona1_desc", "persona2", "persona2_desc",
                        "topic", "scenario", "style"], parser)
        return "manual"

    parser.error(
        "Insufficient arguments. Provide --creative-brief, OR manual args, "
        "OR fixed persona args with --enable-variation, "
        "OR --random-pairings with pool files."
    )


def _require(args, keys: list[str], parser) -> None:
    """Check that all required keys are present in args."""
    missing = [f"--{k.replace('_', '-')}" for k in keys if getattr(args, k) is None]
    if missing:
        parser.error(f"Missing required arguments: {' '.join(missing)}")


def main():
    """Main entry point for the conversation dataset generator."""
    script_start = time.monotonic()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for lib in ("huggingface_hub", "datasets", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    parser = build_parser()
    args = parser.parse_args()

    mode = detect_mode(args, parser)
    logger.info("Mode: %s", mode)

    role_mapping = parse_role_mapping(args.role_mapping)

    # --- Load model ---
    from conversation_dataset_generator.models import load_model_and_pipeline

    text_generator, tokenizer = load_model_and_pipeline(
        model_id=args.model_id, load_in_4bit=args.load_in_4bit,
    )

    # --- Imports ---
    from conversation_dataset_generator.generation import (
        generate_args_from_brief_safe,
        generate_conversation,
        generate_topic_variation,
    )
    from conversation_dataset_generator.output import (
        build_dataset_card,
        create_hf_dataset,
        write_jsonl,
    )

    # --- Determine base arguments per mode ---
    persona1 = None
    persona1_desc = None
    persona2 = None
    persona2_desc = None
    initial_topic = None
    initial_scenario = None
    initial_style = None
    initial_include_points = None
    variation_enabled = False
    character_pool = None
    character_descriptions = None

    if mode == "brief":
        variation_enabled = True
        generated = generate_args_from_brief_safe(
            args.creative_brief, text_generator, tokenizer,
            args.persona1_search_term, args.persona2_search_term,
        )
        if generated is None:
            logger.error("Failed to generate args from brief. Exiting.")
            sys.exit(1)

        persona1 = generated["persona1"]
        persona1_desc = generated["persona1_desc"]
        persona2 = generated["persona2"]
        persona2_desc = generated["persona2_desc"]
        initial_topic = generated["topic"]
        initial_scenario = generated["scenario"]
        initial_style = generated["style"]
        initial_include_points = generated.get("include_points")

    elif mode == "fixed_persona_variation":
        variation_enabled = True
        persona1 = args.fixed_persona1
        persona1_desc = args.fixed_persona1_desc
        persona2 = args.fixed_persona2
        persona2_desc = args.fixed_persona2_desc
        initial_topic = args.initial_topic
        initial_scenario = args.initial_scenario
        initial_style = args.initial_style
        initial_include_points = args.include_points

    elif mode in ("random_pairings", "random_pairings_variation"):
        from conversation_dataset_generator.character_pool import (
            load_character_pool,
            load_description_pool,
            select_random_pair,
            validate_pools,
        )

        char_path = args.character_pool
        desc_path = args.persona_desc_pool

        if not os.path.isabs(char_path) and not char_path.startswith("character-config/"):
            char_path = os.path.join("character-config", char_path)
        if not os.path.isabs(desc_path) and not desc_path.startswith("character-config/"):
            desc_path = os.path.join("character-config", desc_path)

        character_pool = load_character_pool(char_path)
        character_descriptions = load_description_pool(desc_path)
        validate_pools(character_pool, character_descriptions)

        variation_enabled = mode == "random_pairings_variation"
        initial_topic = args.initial_topic
        initial_scenario = args.initial_scenario
        initial_style = args.initial_style
        initial_include_points = args.include_points

    elif mode == "manual":
        variation_enabled = False
        persona1 = args.persona1
        persona1_desc = args.persona1_desc
        persona2 = args.persona2
        persona2_desc = args.persona2_desc
        initial_topic = args.topic
        initial_scenario = args.scenario
        initial_style = args.style
        initial_include_points = args.include_points

    # --- Generation loop ---
    conversations = []
    total_llm_time = 0.0

    for i in tqdm(range(args.num_examples), desc="Generating", unit="example"):
        # Select random pair if needed
        if mode in ("random_pairings", "random_pairings_variation"):
            persona1, persona1_desc, persona2, persona2_desc = select_random_pair(
                character_pool, character_descriptions,
            )
            logger.info("Pair %d: %s & %s", i + 1, persona1, persona2)

        # Topic variation
        current_topic = initial_topic
        current_scenario = initial_scenario
        current_style = initial_style
        current_include_points = initial_include_points

        if variation_enabled:
            variation = generate_topic_variation(
                persona1=persona1, persona1_desc=persona1_desc,
                persona2=persona2, persona2_desc=persona2_desc,
                initial_topic=initial_topic, initial_scenario=initial_scenario,
                initial_style=initial_style,
                generator_pipeline=text_generator, tokenizer=tokenizer,
                original_brief=args.creative_brief if mode == "brief" else None,
            )
            if variation:
                current_topic = variation.get("topic", initial_topic)
                current_scenario = variation.get("scenario", initial_scenario)
                current_style = variation.get("style", initial_style)

        # Generate conversation
        start = time.monotonic()
        turns = generate_conversation(
            topic=current_topic, persona1=persona1, persona2=persona2,
            persona1_desc=persona1_desc, persona2_desc=persona2_desc,
            scenario=current_scenario, style=current_style,
            generator_pipeline=text_generator, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            include_points=current_include_points,
            role_mapping=role_mapping,
        )
        total_llm_time += time.monotonic() - start

        if turns:
            conversations.append({
                "turns": turns,
                "topic": current_topic,
                "scenario": current_scenario,
                "style": current_style,
                "include_points": current_include_points or "",
                "persona1_name": persona1,
                "persona2_name": persona2,
            })

    # --- Write output ---
    num_generated = len(conversations)
    logger.info("Generated %d/%d conversations", num_generated, args.num_examples)

    if not conversations:
        logger.error("No conversations generated. Exiting.")
        sys.exit(1)

    total_turns = write_jsonl(conversations, args.output_file)

    # --- Optional HF upload ---
    if args.upload_to_hub:
        ds = create_hf_dataset(args.output_file)
        if ds is not None:
            card = build_dataset_card(
                mode=mode,
                num_requested=args.num_examples,
                num_generated=num_generated,
                total_turns=total_turns,
                model_id=args.model_id,
                persona1=persona1, persona1_desc=persona1_desc,
                persona2=persona2, persona2_desc=persona2_desc,
                topic=current_topic if conversations else initial_topic,
                scenario=current_scenario if conversations else initial_scenario,
                style=current_style if conversations else initial_style,
                include_points=current_include_points,
                creative_brief=args.creative_brief if mode == "brief" else None,
                search_term1=args.persona1_search_term,
                search_term2=args.persona2_search_term,
                character_pool=character_pool,
                character_descriptions=character_descriptions,
                variation_enabled=variation_enabled,
                repo_id=args.upload_to_hub,
            )

            from conversation_dataset_generator.hub import upload_to_hub
            upload_to_hub(ds, args.upload_to_hub, card, force=args.force_upload)

    elapsed = time.monotonic() - script_start
    logger.info(
        "Done. %d conversations, %d turns, %.2fs total (%.2fs LLM time)",
        num_generated, total_turns, elapsed, total_llm_time,
    )
