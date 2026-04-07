"""Character pool loading, validation, and random pair selection."""

import logging
import random
import yaml

logger = logging.getLogger(__name__)


def load_character_pool(path: str) -> list[str]:
    """Load character names from a YAML file.
    Expects top-level 'characters' key with a list of at least 2 names."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "characters" not in data:
        raise ValueError(
            f"Character pool YAML must contain a 'characters' key. "
            f"Found keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    characters = data["characters"]
    if not isinstance(characters, list) or len(characters) < 2:
        raise ValueError(
            f"Character pool must be a list with at least 2 characters. "
            f"Found {len(characters) if isinstance(characters, list) else 0}."
        )

    logger.info("Loaded %d characters from %s", len(characters), path)
    return characters


def load_description_pool(path: str) -> dict[str, str]:
    """Load character descriptions from a YAML file.
    Expects top-level 'descriptions' key with a dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "descriptions" not in data:
        raise ValueError(
            f"Description pool YAML must contain a 'descriptions' key. "
            f"Found keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    descriptions = data["descriptions"]
    if not isinstance(descriptions, dict):
        raise ValueError(f"Descriptions must be a dictionary. Found: {type(descriptions)}")

    logger.info("Loaded %d descriptions from %s", len(descriptions), path)
    return descriptions


def validate_pools(characters: list[str], descriptions: dict[str, str]) -> None:
    """Validate that all characters have descriptions. Raises ValueError if not."""
    missing = [c for c in characters if c not in descriptions]
    if missing:
        raise ValueError(f"Characters missing descriptions: {missing}")


def select_random_pair(
    characters: list[str], descriptions: dict[str, str]
) -> tuple[str, str, str, str]:
    """Select two random characters and return (name1, desc1, name2, desc2)."""
    selected = random.sample(characters, 2)
    p1, p2 = selected
    return p1, descriptions[p1], p2, descriptions[p2]
