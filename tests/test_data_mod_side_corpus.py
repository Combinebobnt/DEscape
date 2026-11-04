"""GH #263 on a real mod: pass --data-mod-dir the Galactic Battlegrounds Test -
Data folder. No derive and no install: the strings are the mod's alone. With
--scenario-dir holding gh263-gb-sample.aoe2scenario (the in-game Step 0 save),
its stored civs are checked too."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape import (
    asset_source,
    civ_availability,
    dat_derive,
    data_mod_side,
    game_dataset,
    player_fields,
    scenario_io,
)

pytestmark = pytest.mark.corpus

SAMPLE = "gh263-gb-sample.aoe2scenario"
TOKENS = (
    "GAIA", "BRITON-CIV", "FRANKISH-CIV", "GOTHIC-CIV", "TEUTONIC-CIV",
    "JAPANESE-CIV", "CHINESE-CIV", "BYZANTINE-CIV", "PERSIAN-CIV",
)
NAMES = {
    "Galactic Empire", "Gungans", "Rebel Alliance", "Royal Naboo",
    "Trade Federation", "Wookiees", "Republic", "Confederacy",
}


@pytest.fixture()
def gb(request, tmp_path, monkeypatch) -> game_dataset.DataModTables:
    folder = request.config.getoption("--data-mod-dir")
    if not folder:
        pytest.skip("no --data-mod-dir")
    monkeypatch.setenv("AOE2DE_INSTALL_PATH", str(tmp_path))
    monkeypatch.setattr(asset_source, "get_language", lambda: "en")
    asset_source.clear_install_caches()
    # Copies of the AoE2 tables: a switch's hooks read them, and the side files are what is under test.
    for name in dat_derive.TABLE_NAMES:
        shutil.copy(Path(game_dataset.__file__).resolve().parent / name, tmp_path / name)
    tables = game_dataset.DataModTables("gb", tmp_path, Path(folder), "GB")
    game_dataset.set_data_mod(tables)
    return tables


def test_civs_match_what_the_game_lists(gb):
    civs = data_mod_side.civilizations()
    assert tuple(civ.token for civ in civs) == TOKENS
    assert {name for _civ, name in data_mod_side.playable_civs()} == NAMES
    assert civs[0].era == "base" and {civ.era for civ in civs[1:]} == {"swgb"}


def test_age_names_are_the_tech_levels(gb):
    labels = data_mod_side.age_labels("BRITON-CIV")
    assert labels[0] == "Tech Level 1" and labels[-1] == "Post-Tech Level 4"
    assert data_mod_side.age_labels() == labels


def test_the_step_0_sample_stores_civs_the_mod_names(gb, request):
    folder = request.config.getoption("--scenario-dir")
    path = Path(folder) / SAMPLE if folder else None
    if path is None or not path.is_file():
        pytest.skip(f"no {SAMPLE} in --scenario-dir")
    loaded = scenario_io.load_map_and_units(path)
    value = civ_availability.player_civ_value(loaded, 1)
    assert player_fields.resolve_civilization_name(value) == "Gungans"
    assert civ_availability.player_civ(loaded, 1) == 2
    assert civ_availability.start_age_text(civ_availability.player_start_age(loaded, 1), value) == "Tech Level 2"
