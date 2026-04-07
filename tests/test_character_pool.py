import pytest
import yaml
from conversation_dataset_generator.character_pool import (
    load_character_pool,
    load_description_pool,
    validate_pools,
    select_random_pair,
    select_random_group,
)


@pytest.fixture
def pool_dir(tmp_path):
    characters = {"characters": ["Alice", "Bob", "Charlie"]}
    descriptions = {
        "descriptions": {
            "Alice": "A friendly person",
            "Bob": "A grumpy person",
            "Charlie": "A quiet person",
        }
    }
    char_file = tmp_path / "characters.yaml"
    desc_file = tmp_path / "descriptions.yaml"
    char_file.write_text(yaml.dump(characters))
    desc_file.write_text(yaml.dump(descriptions))
    return str(char_file), str(desc_file)


class TestLoadCharacterPool:
    def test_loads_characters(self, pool_dir):
        char_file, _ = pool_dir
        pool = load_character_pool(char_file)
        assert pool == ["Alice", "Bob", "Charlie"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_character_pool("/nonexistent/file.yaml")

    def test_missing_characters_key_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(yaml.dump({"wrong_key": ["a", "b"]}))
        with pytest.raises(ValueError, match="characters"):
            load_character_pool(str(bad_file))

    def test_too_few_characters_raises(self, tmp_path):
        bad_file = tmp_path / "one.yaml"
        bad_file.write_text(yaml.dump({"characters": ["OnlyOne"]}))
        with pytest.raises(ValueError, match="at least 2"):
            load_character_pool(str(bad_file))


class TestLoadDescriptionPool:
    def test_loads_descriptions(self, pool_dir):
        _, desc_file = pool_dir
        pool = load_description_pool(desc_file)
        assert pool["Alice"] == "A friendly person"

    def test_missing_descriptions_key_raises(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(yaml.dump({"wrong_key": {}}))
        with pytest.raises(ValueError, match="descriptions"):
            load_description_pool(str(bad_file))


class TestValidatePools:
    def test_valid_pools_pass(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        validate_pools(characters, descriptions)

    def test_missing_description_raises(self):
        characters = ["Alice", "Bob", "Charlie"]
        descriptions = {"Alice": "Desc", "Bob": "Desc"}
        with pytest.raises(ValueError, match="Charlie"):
            validate_pools(characters, descriptions)


class TestSelectRandomPair:
    def test_returns_two_different_characters(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        p1_name, p1_desc, p2_name, p2_desc = select_random_pair(characters, descriptions)
        assert p1_name != p2_name
        assert p1_name in characters
        assert p2_name in characters
        assert p1_desc == descriptions[p1_name]
        assert p2_desc == descriptions[p2_name]

    def test_many_selections_never_duplicate(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        for _ in range(50):
            p1, _, p2, _ = select_random_pair(characters, descriptions)
            assert p1 != p2


class TestSelectRandomGroup:
    def test_default_count_is_two(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions)
        assert len(group) == 2
        names = [name for name, _ in group]
        assert len(set(names)) == 2

    def test_count_three(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions, count=3)
        assert len(group) == 3
        names = [name for name, _ in group]
        assert len(set(names)) == 3

    def test_returns_name_desc_tuples(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        group = select_random_group(characters, descriptions, count=2)
        for name, desc in group:
            assert name in characters
            assert desc == descriptions[name]

    def test_never_duplicates(self, pool_dir):
        char_file, desc_file = pool_dir
        characters = load_character_pool(char_file)
        descriptions = load_description_pool(desc_file)
        for _ in range(50):
            group = select_random_group(characters, descriptions, count=2)
            names = [n for n, _ in group]
            assert len(set(names)) == 2
