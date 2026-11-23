"""TASK-429's independent oracle for the legacy structures: AoE2:DE's own
importer. Each pair is a pre-DE original and the same file opened and saved
by the DE editor, both in examples/ (gitignored, so this skips elsewhere).
The legacy read must agree with DE's read of the resave everywhere the
importer does not rewrite a value, and every rewrite it does make is listed
here as a measured remap, never waved through.

This is what the byte-level corpus checks (test_legacy_corpus.py) cannot
tell: which of two same-size fields a structure has. The panel readers'
values are compared, so a field read from the wrong bytes shows as a value
the importer would have no reason to change.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from descape import diplomacy_fields, messages_fields, option_fields, player_fields
from descape.scenario_io import load_map_and_units

pytestmark = pytest.mark.corpus

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# (original, DE resave, the Messages the resave replaced with DE campaign text).
PAIRS = {
    "atilla_1": ("atilla_1.scn", "atilla_1_scn_resaved.aoe2scenario", {"ascii_scouts"}),
    "ring75_v0": ("ring75_v0.scx", "ring75_v0_scx_resaved.aoe2scenario", set()),
    # Age of Kings (data 1.20): the resave is owed, saved by AoE2:DE's editor.
    "wallace_5": ("wallace_5.scn", "wallace_5_scn_resaved.aoe2scenario", set()),
}

# Measured importer remaps on the AoC pairs (2026-11-22).
TERRAIN_REMAP = {33: 6, 34: 0}
CIV_TOKENS = {
    1: "BRITON-CIV", 2: "FRANKISH-CIV", 3: "GOTHIC-CIV", 4: "TEUTONIC-CIV", 5: "JAPANESE-CIV",
    6: "CHINESE-CIV", 7: "BYZANTINE-CIV", 8: "PERSIAN-CIV", 9: "SARACEN-CIV", 12: "MONGOL-CIV", 17: "HUN-CIV",
}  # fmt: skip
_U32_MINUS_ONE = 0xFFFFFFFF
DE_POP_LIMIT = 200
# Attila's campaign text: DE renamed player 5's tribe.
TRIBE_RENAMES = {("atilla_1", 5): ("Western Roman Empire", "The Eastern Roman Empire")}


def _expected_resave(name: str, field_id: str, player_id: int, value):
    """The value DE's importer writes for a legacy field value."""
    if field_id == "civilization":
        return CIV_TOKENS.get(value, f"<civ {value}: remap not measured>")
    if field_id == "starting_age":
        return 0 if value == _U32_MINUS_ONE else value + 2
    if field_id == "pop_limit":
        return DE_POP_LIMIT
    if field_id == "player_name_string_id" and value == -1:
        return -2
    if field_id == "tribe_name" and (name, player_id) in TRIBE_RENAMES:
        old, new = TRIBE_RENAMES[(name, player_id)]
        return new if value == old else value
    return value


@pytest.fixture(params=sorted(PAIRS))
def pair(request):
    original, resave, replaced = PAIRS[request.param]
    if not (EXAMPLES / original).is_file():
        pytest.skip(f"no {original} in examples/")
    if not (EXAMPLES / resave).is_file():
        pytest.skip(f"no {resave} in examples/ (open {original} in AoE2:DE's editor and save it under that name)")
    return request.param, load_map_and_units(EXAMPLES / original), load_map_and_units(EXAMPLES / resave), replaced


def test_map_and_units_match_the_de_resave(pair) -> None:
    _name, legacy, de, _replaced = pair
    assert legacy.source_format.legacy and not de.source_format.legacy
    lm, dm = legacy.map_manager, de.map_manager
    assert (lm.map_width, lm.map_height) == (dm.map_width, dm.map_height)
    assert [t.elevation for t in lm.terrain] == [t.elevation for t in dm.terrain]
    moved = Counter((a.terrain_id, b.terrain_id) for a, b in zip(lm.terrain, dm.terrain, strict=True) if a.terrain_id != b.terrain_id)
    assert all(TERRAIN_REMAP.get(old) == new for old, new in moved), moved

    def placements(loaded):
        return [(pid, u.unit_const, u.x, u.y) for pid, units in enumerate(loaded.unit_manager.units) for u in units]

    def ref_ids(loaded):
        return [u.reference_id for units in loaded.unit_manager.units for u in units]

    assert Counter(placements(legacy)) == Counter(placements(de))
    assert ref_ids(legacy) == ref_ids(de)
    assert legacy.player_colors == de.player_colors
    assert legacy.team_indices == de.team_indices


def test_player_values_match_the_de_resave_after_measured_remaps(pair) -> None:
    name, legacy, de, _replaced = pair
    diffs = []
    for player_id in range(9):
        de_specs = {s.field_id: s for s in player_fields.specs_for_player(de, player_id)}
        for spec in player_fields.specs_for_player(legacy, player_id):
            assert spec.field_id in de_specs, f"p{player_id}.{spec.field_id} is legacy-only"
            value = player_fields.current_value(legacy, spec, player_id)
            if spec.field_id == "pop_limit" and not (25 <= value <= DE_POP_LIMIT and value % 25 == 0):
                diffs.append(f"p{player_id}.pop_limit: legacy {value!r} is no AoC pop limit (25..200, step 25)")
            want = _expected_resave(name, spec.field_id, player_id, value)
            got = player_fields.current_value(de, de_specs[spec.field_id], player_id)
            if want != got:
                diffs.append(f"p{player_id}.{spec.field_id}: legacy {value!r}, expected {want!r}, DE {got!r}")
    assert not diffs, diffs


def test_options_and_diplomacy_match_the_de_resave(pair) -> None:
    _name, legacy, de, _replaced = pair
    de_specs = {s.field_id: s for s in option_fields.specs_for(de)}
    diffs = []
    for spec in option_fields.specs_for(legacy):
        assert spec.field_id in de_specs, f"{spec.field_id} is legacy-only"
        a, b = option_fields.current_value(legacy, spec), option_fields.current_value(de, de_specs[spec.field_id])
        if a != b:
            diffs.append(f"{spec.field_id}: legacy {a!r}, DE {b!r}")
    for row in range(1, 9):
        for col in range(1, 9):
            a, b = diplomacy_fields.stance_value(legacy, row, col), diplomacy_fields.stance_value(de, row, col)
            if a != b:
                diffs.append(f"stance {row}->{col}: legacy {a}, DE {b}")
        a, b = diplomacy_fields.allied_victory_value(legacy, row), diplomacy_fields.allied_victory_value(de, row)
        if a != b:
            diffs.append(f"allied victory {row}: legacy {a}, DE {b}")
    assert not diffs, diffs


def test_messages_match_unless_the_resave_emptied_or_replaced_them(pair) -> None:
    _name, legacy, de, replaced = pair
    lm = legacy._scenario.sections["Messages"].retriever_map
    dm = de._scenario.sections["Messages"].retriever_map
    diffs = []
    for spec in messages_fields.present_fields(lm):
        a, b = lm[spec.retriever].data, dm[spec.retriever].data
        if b and a != b and spec.retriever not in replaced:
            diffs.append(f"{spec.retriever}: {a[:60]!r} -> {b[:60]!r}")
    assert not diffs, diffs
