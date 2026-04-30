"""Command-line interface and orchestration for conversation generation."""

import argparse
import logging
import os
import sys
import time

import yaml
from tqdm import tqdm

from conversation_dataset_generator.models import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)


def parse_role_mapping(mapping_str: str | None) -> dict:
    """Parse role mapping string like 'p1=human,p2=gpt' into a dict.

    Legacy function kept for backward compatibility.
    Prefer build_role_mapping() for N-speaker support.
    """
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


def build_role_mapping(
    persona_names: list[str],
    train_speaker: str | None = None,
    role_mapping_str: str | None = None,
) -> dict:
    """Build a {name: role} mapping for N speakers.

    - If train_speaker is set, that speaker gets 'gpt', rest get 'human'.
    - If role_mapping_str is set, parse 'Name1=human,Name2=gpt' format.
    - Default: first speaker = 'human', rest = 'gpt'.
    """
    if train_speaker:
        return {
            name: "gpt" if name == train_speaker else "human"
            for name in persona_names
        }

    if role_mapping_str:
        result = {}
        for part in role_mapping_str.split(","):
            key, _, value = part.strip().partition("=")
            if key and value in ("human", "gpt"):
                result[key] = value
        return result

    # Default: first = human, rest = gpt
    mapping = {}
    for i, name in enumerate(persona_names):
        mapping[name] = "human" if i == 0 else "gpt"
    return mapping


def load_personas_from_yaml(path: str) -> list[tuple[str, str]]:
    """Load personas from a YAML file.

    Expected format:
        personas:
          - name: "X"
            description: "Y"
          - ...

    Returns list of (name, description) tuples.
    Raises ValueError if the format is invalid.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "personas" not in data:
        raise ValueError(f"YAML file {path} must contain a 'personas' key.")

    personas = []
    for entry in data["personas"]:
        if not isinstance(entry, dict) or "name" not in entry or "description" not in entry:
            raise ValueError(
                f"Each persona must have 'name' and 'description' fields. Got: {entry}"
            )
        personas.append((entry["name"], entry["description"]))
    return personas


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
    mode_group.add_argument(
        "--continue-from", type=str,
        help="Continue from existing JSONL file.",
    )
    mode_group.add_argument(
        "--conversation-id", type=int, default=None,
        help="Conversation ID to continue (default: last).",
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

    # Multi-Speaker args
    multi = parser.add_argument_group("Multi-Speaker")
    multi.add_argument(
        "--persona", nargs=2, action="append", metavar=("NAME", "DESC"),
        help="Add a persona (repeatable).",
    )
    multi.add_argument("--personas", type=str, help="YAML file with personas list.")
    multi.add_argument(
        "--train-speaker", type=str, default=None,
        help="Speaker to assign 'gpt' role (rest become 'human').",
    )
    multi.add_argument(
        "--group-size", type=int, default=2,
        help="Characters per conversation in random pairings (default: 2).",
    )

    # General args
    general = parser.add_argument_group("General")
    general.add_argument("--num-examples", type=int, default=3)
    general.add_argument("--output-file", type=str, default="generated_data.jsonl")
    general.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    general.add_argument("--max-new-tokens", type=int, default=4096)
    general.add_argument("--upload-to-hub", type=str, default=None, metavar="REPO_ID")
    general.add_argument("--load-in-4bit", action="store_true")
    general.add_argument("--force-upload", action="store_true")
    general.add_argument("--role-mapping", type=str, default=None,
                         help="Role mapping: 'p1=human,p2=gpt' or 'p1=gpt,p2=human'")

    return parser


def detect_mode(args, parser) -> str:
    """Determine the generation mode from parsed arguments."""
    if args.continue_from:
        return "continue"

    if args.creative_brief:
        return "brief"

    if (args.persona or args.personas) and args.topic:
        return "manual"

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

    # --- Load model ---
    from conversation_dataset_generator.models import load_model_and_pipeline

    text_generator, tokenizer = load_model_and_pipeline(
        model_id=args.model_id, load_in_4bit=args.load_in_4bit,
    )
    from conversation_dataset_generator.backend import HFBackend
    backend = HFBackend(text_generator, tokenizer)

    # --- Imports ---
    from conversation_dataset_generator.generation import (
        generate_args_from_brief_safe,
        generate_continuation,
        generate_conversation,
        generate_topic_variation,
    )
    from conversation_dataset_generator.output import (
        build_dataset_card,
        create_hf_dataset,
        load_conversation_from_jsonl,
        write_jsonl,
    )

    # --- Determine base arguments per mode ---
    personas = None
    initial_topic = None
    initial_scenario = None
    initial_style = None
    initial_include_points = None
    variation_enabled = False
    character_pool = None
    character_descriptions = None

    # --- Normalize personas from any input method ---
    if args.persona:
        personas = [(name, desc) for name, desc in args.persona]
    elif hasattr(args, 'personas') and args.personas:
        personas = load_personas_from_yaml(args.personas)

    if mode == "continue":
        conv_data = load_conversation_from_jsonl(
            args.continue_from, conversation_id=args.conversation_id,
        )
        if conv_data is None:
            logger.error("Could not load conversation from %s", args.continue_from)
            sys.exit(1)

        if personas is None:
            personas = [(name, "") for name in conv_data["speaker_names"]]

        persona_names = [name for name, _ in personas]
        role_mapping = build_role_mapping(persona_names, args.train_speaker, args.role_mapping)

        conversations = []
        total_llm_time = 0.0

        for i in tqdm(range(args.num_examples), desc="Continuing", unit="example"):
            start = time.monotonic()
            turns = generate_continuation(
                personas=personas, prior_turns=conv_data["turns"],
                topic=conv_data["topic"], scenario=conv_data["scenario"],
                style=conv_data["style"],
                backend=backend,
                max_new_tokens=args.max_new_tokens, role_mapping=role_mapping,
            )
            total_llm_time += time.monotonic() - start

            if turns:
                conversations.append({
                    "turns": turns,
                    "topic": conv_data["topic"],
                    "scenario": conv_data["scenario"],
                    "style": conv_data["style"],
                    "include_points": conv_data.get("include_points", ""),
                })

    else:
        # Non-continue modes
        if mode == "brief":
            variation_enabled = True
            generated = generate_args_from_brief_safe(
                args.creative_brief, backend,
                args.persona1_search_term, args.persona2_search_term,
            )
            if generated is None:
                logger.error("Failed to generate args from brief. Exiting.")
                sys.exit(1)

            personas = [
                (generated["persona1"], generated["persona1_desc"]),
                (generated["persona2"], generated["persona2_desc"]),
            ]
            initial_topic = generated["topic"]
            initial_scenario = generated["scenario"]
            initial_style = generated["style"]
            initial_include_points = generated.get("include_points")

        elif mode == "fixed_persona_variation":
            variation_enabled = True
            personas = [
                (args.fixed_persona1, args.fixed_persona1_desc),
                (args.fixed_persona2, args.fixed_persona2_desc),
            ]
            initial_topic = args.initial_topic
            initial_scenario = args.initial_scenario
            initial_style = args.initial_style
            initial_include_points = args.include_points

        elif mode in ("random_pairings", "random_pairings_variation"):
            from conversation_dataset_generator.character_pool import (
                load_character_pool,
                load_description_pool,
                select_random_group,
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
            if personas is None and args.persona1 and args.persona2:
                personas = [
                    (args.persona1, args.persona1_desc or ""),
                    (args.persona2, args.persona2_desc or ""),
                ]
            initial_topic = args.topic
            initial_scenario = args.scenario
            initial_style = args.style
            initial_include_points = args.include_points

        # --- Generation loop ---
        conversations = []
        total_llm_time = 0.0

        for i in tqdm(range(args.num_examples), desc="Generating", unit="example"):
            personas_for_this_conv = personas

            # Select random group if needed
            if mode in ("random_pairings", "random_pairings_variation"):
                personas_for_this_conv = select_random_group(
                    character_pool, character_descriptions, count=args.group_size,
                )
                names = [n for n, _ in personas_for_this_conv]
                logger.info("Group %d: %s", i + 1, " & ".join(names))

            # Build role mapping for this conversation's personas
            if personas_for_this_conv:
                persona_names = [name for name, _ in personas_for_this_conv]
                role_mapping = build_role_mapping(
                    persona_names, args.train_speaker, args.role_mapping,
                )
            else:
                role_mapping = parse_role_mapping(args.role_mapping)

            # Topic variation
            current_topic = initial_topic
            current_scenario = initial_scenario
            current_style = initial_style
            current_include_points = initial_include_points

            if variation_enabled and personas_for_this_conv:
                # Use first two personas for variation prompt (backward compat)
                p1_name, p1_desc = personas_for_this_conv[0]
                p2_name, p2_desc = (personas_for_this_conv[1]
                                     if len(personas_for_this_conv) > 1
                                     else personas_for_this_conv[0])
                variation = generate_topic_variation(
                    persona1=p1_name, persona1_desc=p1_desc,
                    persona2=p2_name, persona2_desc=p2_desc,
                    initial_topic=initial_topic, initial_scenario=initial_scenario,
                    initial_style=initial_style,
                    backend=backend,
                    original_brief=args.creative_brief if mode == "brief" else None,
                )
                if variation:
                    current_topic = variation.get("topic", initial_topic)
                    current_scenario = variation.get("scenario", initial_scenario)
                    current_style = variation.get("style", initial_style)

            # Generate conversation
            start = time.monotonic()
            turns = generate_conversation(
                topic=current_topic, personas=personas_for_this_conv,
                scenario=current_scenario, style=current_style,
                backend=backend,
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
        # Extract persona1/persona2 from personas for backward-compatible card
        p1 = personas[0][0] if personas and len(personas) > 0 else None
        p1d = personas[0][1] if personas and len(personas) > 0 else None
        p2 = personas[1][0] if personas and len(personas) > 1 else None
        p2d = personas[1][1] if personas and len(personas) > 1 else None

        ds = create_hf_dataset(args.output_file)
        if ds is not None:
            card = build_dataset_card(
                mode=mode,
                num_requested=args.num_examples,
                num_generated=num_generated,
                total_turns=total_turns,
                model_id=args.model_id,
                persona1=p1, persona1_desc=p1d,
                persona2=p2, persona2_desc=p2d,
                topic=initial_topic,
                scenario=initial_scenario,
                style=initial_style,
                include_points=initial_include_points,
                creative_brief=args.creative_brief if mode == "brief" else None,
                search_term1=getattr(args, 'persona1_search_term', None),
                search_term2=getattr(args, 'persona2_search_term', None),
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
