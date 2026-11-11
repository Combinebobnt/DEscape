"""GH #263: a data mod's civilizations.json and eras.json. Qt-free, on stand-in
files; the Players tab and the window's refreshes are in test_data_mod_viewer.py."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import (
    asset_source,
    civ_art,
    civ_availability,
    dat_derive,
    data_mod_side,
    data_mods,
    game_dataset,
    player_fields,
    train_locations,
)

CIVS = [
    {"data_name": "GAIA", "internal_name": "Gaia", "name_string_id": 1, "era": "base"},
    {"data_name": "CHINESE-CIV", "internal_name": "Zed", "name_string_id": 2, "era": "future"},
    {"data_name": "MOD-A-CIV", "internal_name": "Alpha", "name_string_id": 3, "era": "future"},
    {"data_name": "MOD-B-CIV", "internal_name": "Beta", "name_string_id": 999, "era": "old"},
    {"data_name": "MOD-C-CIV", "era": "unlisted"},
]
ERAS = [
    {"Name": "base", "Ages": [{"NameId": id_} for id_ in range(11, 16)]},
    {"Name": "future", "Ages": [{"NameId": id_} for id_ in range(21, 26)]},
    {"Name": "short", "Ages": [{"NameId": id_} for id_ in range(31, 35)]},
    {"Name": "old", "Ages": [{"NameId": id_} for id_ in (41, 42, 43, 44, 999)]},
]
STRINGS = {
    2: "Zed Empire", 3: "Alpha Union",
    **{id_: f"Base {id_}" for id_ in range(11, 16)},
    **{id_: f"Tech {id_}" for id_ in range(21, 26)},
    **{id_: f"Old {id_}" for id_ in range(41, 45)},
}


def _dat(root: Path) -> Path:
    dat = root / "resources" / "_common" / "dat"
    dat.mkdir(parents=True, exist_ok=True)
    return dat


def _write_strings(root: Path, lang: str, strings: dict[int, str]) -> None:
    folder = root / "resources" / lang / "strings" / "key-value"
    folder.mkdir(parents=True)
    text = "".join(f'{id_} "{value}"\n' for id_, value in strings.items())
    (folder / "key-value-modded-strings-utf8.txt").write_text(text, encoding="utf-8")


def _tables(root: Path, side_key: str = "") -> game_dataset.DataModTables:
    """Copies of the AoE2 tables: a switch's hooks read them, and the side files are what is under test."""
    folder = root / "tables"
    if not folder.is_dir():
        folder.mkdir(parents=True)
        for name in dat_derive.TABLE_NAMES:
            shutil.copy(Path(game_dataset.__file__).resolve().parent / name, folder / name)
    return game_dataset.DataModTables("key", folder, root, "Stand-in", side_key=side_key)


@pytest.fixture()
def mod(tmp_path, monkeypatch) -> game_dataset.DataModTables:
    """A mod with both files and its strings in effect, over an empty install."""
    root = tmp_path / "mod"
    dat = _dat(root)
    (dat / "civilizations.json").write_text(json.dumps({"civilization_list": CIVS}))
    (dat / "eras.json").write_text(json.dumps(ERAS))
    _write_strings(root, "en", STRINGS)
    install = tmp_path / "install"
    install.mkdir()
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(install))
    monkeypatch.setattr(asset_source, "get_language", lambda: "en")
    asset_source.clear_install_caches()
    tables = _tables(root)
    game_dataset.set_data_mod(tables)
    return tables


def _codec(monkeypatch, codec: str) -> SimpleNamespace:
    monkeypatch.setattr(player_fields, "_player_data_1_field_codec", lambda loaded, field: codec)
    return SimpleNamespace()


# --- files --------------------------------------------------------------


def test_side_files_match_case_insensitively_exact_name_first(tmp_path):
    (tmp_path / "CIVILIZATIONS.JSON").write_text("{}")
    # .casefold(): /mnt/c answers the exact name for it too.
    assert data_mod_side.side_file(tmp_path, "civilizations.json").name.casefold() == "civilizations.json"
    assert data_mod_side.side_file(tmp_path, "eras.json") is None
    assert data_mod_side.side_file(tmp_path / "gone", "eras.json") is None


def test_fingerprint_moves_on_any_byte_and_is_empty_with_no_files(tmp_path):
    assert data_mod_side.fingerprint(None) == ""
    assert data_mod_side.fingerprint(tmp_path) == ""
    (tmp_path / "unitlines.json").write_text("{}")
    assert data_mod_side.fingerprint(tmp_path) == "", "only the files DEscape reads"
    (tmp_path / "eras.json").write_text("[]")
    first = data_mod_side.fingerprint(tmp_path)
    assert first
    (tmp_path / "eras.json").write_text("[ ]")
    second = data_mod_side.fingerprint(tmp_path)
    assert second != first
    (tmp_path / "civilizations.json").write_text("{}")
    assert data_mod_side.fingerprint(tmp_path) != second


def test_fingerprint_hashes_the_portrait_tables_only(tmp_path):
    """GH #363: the mod's own widgetui/ icons.json, materials.json and morematerials.json join side_key."""
    ui = tmp_path / "widgetui"
    ui.mkdir()
    assert data_mod_side.fingerprint(None, ui) == ""
    (ui / "techtreepreviewpanel.json").write_text("{}")
    assert data_mod_side.fingerprint(None, ui) == "", "only the files DEscape reads"
    (ui / "icons.json").write_text("{}")
    first = data_mod_side.fingerprint(None, ui)
    assert first
    (ui / "icons.json").write_text("{ }")
    second = data_mod_side.fingerprint(None, ui)
    assert second != first
    (ui / "MoreMaterials.json").write_text("{}")
    assert data_mod_side.fingerprint(None, ui) != second
    (tmp_path / "eras.json").write_text("[]")
    (ui / "techtreepreviewpanel.json").write_text("[]")
    with_side = data_mod_side.fingerprint(tmp_path, ui)
    assert with_side != data_mod_side.fingerprint(tmp_path) and with_side != data_mod_side.fingerprint(None, ui)


@pytest.mark.parametrize("text", [
    "not json", "[]", "{}", '{"civilization_list": 3}', '{"civilization_list": [1, 2]}',
    '{"civilization_list": [{"data_name": 1}, {"data_name": "A"}]}',
    '{"civilization_list": [{"data_name": "GAIA"}]}',
])
def test_a_bad_civilizations_file_counts_as_absent(tmp_path, text):
    path = tmp_path / "civilizations.json"
    path.write_text(text)
    assert data_mod_side.parse_civilizations(path) is None


@pytest.mark.parametrize("text", ["not json", "{}", "[1]", '[{"Name": "a"}]', "[]"])
def test_a_bad_eras_file_counts_as_absent(tmp_path, text):
    path = tmp_path / "eras.json"
    path.write_text(text)
    assert data_mod_side.parse_eras(path) is None


def test_an_era_without_five_int_ids_is_left_out(tmp_path):
    path = tmp_path / "eras.json"
    path.write_text(json.dumps([*ERAS, {"Name": "text", "Ages": [{"NameId": "1"}] * 5}]))
    assert set(data_mod_side.parse_eras(path)) == {"base", "future", "old"}


def test_a_civs_index_is_its_list_position(tmp_path):
    path = tmp_path / "civilizations.json"
    path.write_text(json.dumps({"civilization_list": CIVS}))
    civs = data_mod_side.parse_civilizations(path)
    assert [civ.index for civ in civs] == list(range(len(CIVS)))
    assert (civs[1].token, civs[1].era) == ("CHINESE-CIV", "future")


# --- side_key -----------------------------------------------------------


def test_tables_for_fills_side_key_from_the_mods_files(tmp_path, monkeypatch):
    root = tmp_path / "mod"
    dat = _dat(root)
    (dat / "eras.json").write_text("[]")
    entry = SimpleNamespace(path=tmp_path / "tables", mod_only_ids=frozenset(), warnings=())
    monkeypatch.setattr(data_mods.dat_cache, "lookup", lambda key: entry)
    mod = data_mods.DataMod("s", "Stand-in", None, root, dat / "empires2_x2_p1.dat")
    first = data_mods.tables_for(mod, "key")
    assert first.side_key == data_mod_side.fingerprint(dat) != ""
    (dat / "eras.json").write_text("[ ]")
    second = data_mods.tables_for(mod, "key")
    assert second.key == first.key and second.side_key != first.side_key
    assert first != second, "an edit is a new overlay"


def test_tables_for_side_key_follows_the_mods_icon_table(tmp_path, monkeypatch):
    root = tmp_path / "mod"
    ui = root / "widgetui"
    ui.mkdir(parents=True)
    (ui / "icons.json").write_text("{}")
    entry = SimpleNamespace(path=tmp_path / "tables", mod_only_ids=frozenset(), warnings=())
    monkeypatch.setattr(data_mods.dat_cache, "lookup", lambda key: entry)
    mod = data_mods.DataMod("s", "Stand-in", None, root, _dat(root) / "empires2_x2_p1.dat")
    first = data_mods.tables_for(mod, "key")
    assert first.side_key == data_mod_side.fingerprint(None, ui) != ""
    (ui / "icons.json").write_text('{"Units": {}}')
    assert data_mods.tables_for(mod, "key").side_key != first.side_key


def test_only_side_key_changed_still_switches_and_rereads(mod):
    assert [civ.token for civ in data_mod_side.civilizations()][1] == "CHINESE-CIV"
    path = mod.root / "resources" / "_common" / "dat" / "civilizations.json"
    path.write_text(json.dumps({"civilization_list": CIVS[:2]}))
    assert len(data_mod_side.civilizations()) == len(CIVS), "cached while the overlay stays"
    assert game_dataset.set_data_mod(_tables(mod.root, side_key="edited")) is True
    assert len(data_mod_side.civilizations()) == 2


def test_no_mod_no_file_and_return_of_rome_read_nothing(mod, tmp_path):
    assert data_mod_side.civilizations() is not None
    game_dataset.set_active(game_dataset.ROR)
    assert data_mod_side.civilizations() is None
    game_dataset.set_active(game_dataset.AOE2)
    game_dataset.set_data_mod(None)
    assert data_mod_side.civilizations() is None and data_mod_side.age_labels() is None
    game_dataset.set_data_mod(_tables(tmp_path / "bare"))
    assert data_mod_side.civilizations() is None and data_mod_side.age_labels() is None


# --- names and lists ----------------------------------------------------


def test_civ_names_fall_back_to_internal_name_then_the_token(mod):
    resolve = player_fields.resolve_civilization_name
    assert resolve("CHINESE-CIV") == "Zed Empire", "the mod's civ, not AoE2's Chinese"
    assert resolve(2) == "Alpha Union"
    assert resolve("MOD-B-CIV") == "Beta"
    assert resolve("MOD-C-CIV") == "MOD-C-CIV"
    assert resolve("GAIA") == "Gaia"
    assert resolve("HUN-CIV") == "Huns", "a civ the mod lacks falls through"
    assert resolve(4444) == "4444"
    game_dataset.set_data_mod(None)
    assert resolve("CHINESE-CIV") == "Chinese"


@pytest.mark.parametrize(("codec", "alpha", "full", "random"), [
    ("str16", "MOD-A-CIV", "FULL-RANDOM-CIV", "RANDOM-CIV"),
    ("u32", 2, 65539, 65537),
])
def test_civ_choices_are_the_mods_by_name_then_full_random_and_random(mod, monkeypatch, codec, alpha, full, random):
    choices = player_fields.civilization_choices(_codec(monkeypatch, codec))
    assert [name for _value, name in choices] == [
        "Alpha Union", "Beta", "MOD-C-CIV", "Zed Empire", "Full Random", "Random",
    ]
    assert choices[0][0] == alpha and choices[-2][0] == full and choices[-1][0] == random


def test_civ_choices_are_todays_with_no_civs_file(tmp_path, monkeypatch):
    loaded = _codec(monkeypatch, "str16")
    today = player_fields.civilization_choices(loaded)
    game_dataset.set_data_mod(_tables(tmp_path / "bare"))
    assert player_fields.civilization_choices(loaded) == today


def test_a_language_change_reads_that_languages_names(mod, monkeypatch):
    _write_strings(mod.root, "de", {3: "Alpha Bund"})
    assert data_mod_side.civ_name(data_mod_side.find_civ(2), "de") == "Alpha Bund"
    asset_source.clear_install_caches()
    monkeypatch.setattr(asset_source, "get_language", lambda: "de")
    assert player_fields.resolve_civilization_name(2) == "Alpha Bund"
    assert data_mod_side.age_labels("MOD-A-CIV") is None, "no German age names, so AoE2's"


def test_a_stored_civ_the_mod_lacks_shows_the_first_civ_or_random(mod, monkeypatch):
    shown = player_fields.unlisted_civ_shown
    choices = player_fields.civilization_choices(_codec(monkeypatch, "str16"))
    assert shown("civilization", "HUN-CIV", choices) == "MOD-A-CIV", "Alpha Union, first by name"
    assert shown("architecture", "HUN-CIV", choices) == "RANDOM-CIV"
    assert shown("civilization", 4444, choices) == "MOD-A-CIV"
    assert "first civilization, Alpha Union" in player_fields.unlisted_civ_tooltip("civilization", "HUN-CIV", choices)
    assert "Random is shown" in player_fields.unlisted_civ_tooltip("architecture", "HUN-CIV", choices)
    assert shown("civilization", "MOD-A-CIV", choices) is None
    assert shown("civilization", "GAIA", choices) is None, "GAIA is the mod's index 0"
    assert shown("civilization", "RANDOM-CIV", choices) is None, "already listed"
    assert shown("architecture", "FULL-RANDOM-CIV", choices) is None
    old_choices = player_fields.civilization_choices(_codec(monkeypatch, "u32"))
    assert shown("civilization", 65537, old_choices) is None
    assert shown("civilization", 17, old_choices) == 2, "an int past the mod's list"
    game_dataset.set_data_mod(None)
    assert shown("civilization", "HUN-CIV", choices) is None
    assert player_fields.unlisted_civ_tooltip("civilization", "HUN-CIV", choices) == ""


# --- tokens -------------------------------------------------------------


def test_a_mod_only_token_encodes_with_the_mod_and_after_a_switch_to_none(mod):
    payload = player_fields._encode_str16("MOD-A-CIV")
    assert payload.endswith(b"MOD-A-CIV")
    game_dataset.set_data_mod(None)
    assert player_fields._encode_str16("MOD-A-CIV") == payload


def test_a_mod_put_in_effect_needs_no_encode_to_be_remembered(mod):
    game_dataset.set_data_mod(None)
    assert "MOD-B-CIV" in player_fields.encodable_civilizations()


def test_a_session_that_never_had_the_mod_refuses_its_token():
    with pytest.raises(ValueError):
        player_fields._encode_str16("MOD-A-CIV")
    assert player_fields._encode_str16("CHINESE-CIV").endswith(b"CHINESE-CIV")


# --- ages ---------------------------------------------------------------


def test_age_labels_follow_the_civs_era_with_values_2_to_6(mod):
    choices = player_fields.starting_age_choices("MOD-A-CIV")
    assert choices == ((2, "Tech 21"), (3, "Tech 22"), (4, "Tech 23"), (5, "Tech 24"), (6, "Tech 25"))
    assert player_fields.starting_age_choices(2) == choices, "an int names the civ too"
    assert player_fields.starting_age_choices(0)[0][1] == "Base 11", "index 0 is GAIA, with its own era"


@pytest.mark.parametrize("civ_value", [None, "RANDOM-CIV", "HUN-CIV", 65537])
def test_a_civ_naming_nothing_uses_the_era_most_civs_share(mod, civ_value):
    assert player_fields.starting_age_choices(civ_value)[0][1] == "Tech 21"


def test_with_no_usable_shared_era_a_civ_naming_nothing_gets_aoe2s_labels(mod):
    eras = [era for era in ERAS if era["Name"] != "future"]
    (mod.root / "resources" / "_common" / "dat" / "eras.json").write_text(json.dumps(eras))
    game_dataset.set_data_mod(_tables(mod.root, side_key="edited"))
    assert player_fields.starting_age_choices(None) == player_fields._STARTING_AGE_CHOICES
    assert player_fields.starting_age_choices("GAIA")[0][1] == "Base 11"


@pytest.mark.parametrize("civ_value", ["MOD-B-CIV", "MOD-C-CIV"])
def test_an_era_missing_or_with_a_missing_name_gives_aoe2s_labels(mod, civ_value):
    assert player_fields.starting_age_choices(civ_value) == player_fields._STARTING_AGE_CHOICES
    assert player_fields.starting_age_tooltip(civ_value) == ""


def test_no_eras_file_gives_aoe2s_labels(mod):
    (mod.root / "resources" / "_common" / "dat" / "eras.json").unlink()
    game_dataset.set_data_mod(_tables(mod.root, side_key="edited"))
    assert player_fields.starting_age_choices("MOD-A-CIV") == player_fields._STARTING_AGE_CHOICES


def test_the_tooltip_names_the_editors_ages_only_when_they_differ(mod):
    assert "Dark Age to Post Imperial Age" in player_fields.starting_age_tooltip("MOD-A-CIV")
    game_dataset.set_data_mod(None)
    assert player_fields.starting_age_tooltip("MOD-A-CIV") == ""
    assert player_fields.starting_age_choices("MOD-A-CIV") == player_fields._STARTING_AGE_CHOICES


def test_return_of_rome_ages_are_unchanged(mod):
    game_dataset.set_active(game_dataset.ROR)
    assert player_fields.starting_age_choices("MOD-A-CIV") == player_fields._ROR_STARTING_AGE_CHOICES
    assert player_fields.starting_age_tooltip("MOD-A-CIV") == ""


def test_start_age_clause_keeps_aoe2_wording_and_uses_the_mods_name(mod):
    clause = train_locations.missing_text
    reason = civ_availability.MISSING_START
    assert clause(reason, "tech", "Loom", "Zed", 1, 3, "MOD-A-CIV") == "player 1 starts in Tech 22, which already has Loom"
    assert train_locations.PlayerContext(player=1, start_age=3, civ_value="MOD-A-CIV").civ_value == "MOD-A-CIV"
    game_dataset.set_data_mod(None)
    assert clause(reason, "tech", "Loom", "Franks", 1, 3, "FRANKISH-CIV") == (
        "player 1 starts in the Feudal Age, which already has Loom"
    )
    assert clause(reason, "tech", "Loom", "Franks", 1, 3) == "player 1 starts in the Feudal Age, which already has Loom"


# --- building art and civ filtering -------------------------------------


@pytest.mark.parametrize(("value", "expected"), [
    ("CHINESE-CIV", 1), ("MOD-A-CIV", 2), (1, 1), (4, 4),
    ("GAIA", None), (0, None), (5, None), ("HUN-CIV", None), ("RANDOM-CIV", None), (65537, None), (True, None),
])
def test_civ_index_reads_the_mods_list(mod, value, expected):
    assert civ_art.civ_index(value) == expected


def test_civ_index_is_todays_with_no_file_and_for_a_return_of_rome_file(mod, tmp_path):
    assert civ_art.civ_index(1, ror=True) == 1
    assert civ_art.civ_index("MOD-A-CIV", ror=True) is None, "a RoR file's civs, whatever mod is set"
    game_dataset.set_data_mod(_tables(tmp_path / "bare"))
    assert civ_art.civ_index("CHINESE-CIV") == 6


def test_player_civ_maps_a_mod_token(mod, monkeypatch):
    monkeypatch.setattr(civ_availability, "player_civ_value", lambda loaded, player: "MOD-A-CIV")
    assert civ_availability.player_civ(SimpleNamespace(), 1) == 2
