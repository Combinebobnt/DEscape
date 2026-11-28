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

from descape import diplomacy_fields, legacy_convert, legacy_import, messages_fields, option_fields, player_fields
from descape.scenario_io import load_map_and_units

pytestmark = pytest.mark.corpus

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# (original, DE resave, the Messages the resave replaced with DE campaign text).
PAIRS = {
    "atilla_1": ("atilla_1.scn", "atilla_1_scn_resaved.aoe2scenario", {"ascii_scouts"}),
    "ring75_v0": ("ring75_v0.scx", "ring75_v0_scx_resaved.aoe2scenario", set()),
    # Age of Kings (data 1.20): the resave is owed, saved by AoE2:DE's editor.
    "wallace_5": ("wallace_5.scn", "wallace_5_scn_resaved.aoe2scenario", set()),
    # GH #316: short-count effects and conditions (Voobly AoC); the resave is owed.
    "dodgeball_tiw": ("dodgeball_tiw.scx", "dodgeball_tiw_scx_resaved.aoe2scenario", set()),
}

# Measured importer remaps on the AoC pairs (2026-11-22), shared with the converter (GH #315).
TERRAIN_REMAP = {old: new for old, (new, _layer) in legacy_convert.TERRAIN_REMAP.items()}
CIV_TOKENS = legacy_convert.CIV_TOKENS
_U32_MINUS_ONE = 0xFFFFFFFF
DE_POP_LIMIT = legacy_convert.DE_POP_LIMIT
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


def test_legacy_import_age_map_matches_the_de_resave(pair) -> None:
    """legacy_import's pure map on every pair, whatever its band (TASK-685). The AoC pairs store
    only legacy -1, 0 and 1: 2..4 (Castle to Post-Imperial) follow the offset, unmeasured."""
    _name, legacy, de, _replaced = pair
    diffs = []
    for player_id in range(1, 9):
        de_specs = {s.field_id: s for s in player_fields.specs_for_player(de, player_id)}
        spec = next((s for s in player_fields.specs_for_player(legacy, player_id) if s.field_id == "starting_age"), None)
        if spec is None:
            continue
        raw = player_fields.current_value(legacy, spec, player_id)
        got = player_fields.current_value(de, de_specs["starting_age"], player_id)
        if legacy_import.starting_age_to_de(raw) != got:
            diffs.append(f"p{player_id}: legacy {raw!r} -> {legacy_import.starting_age_to_de(raw)!r}, DE {got!r}")
    assert not diffs, diffs


def test_legacy_import_tree_map_matches_the_de_resave(pair) -> None:
    """Every legacy tree, unchanged v = 0 units included, lands in the importer's 3v..3v+2 run."""
    _name, legacy, de, _replaced = pair
    de_by_ref = {u.reference_id: u for units in de.unit_manager.units for u in units}
    diffs, checked = [], 0
    for units in legacy.unit_manager.units:
        for u in units:
            if u.unit_const not in legacy_import.TREE_CONSTS:
                continue
            drawn = legacy_import.tree_variant(u.rotation, u.reference_id)
            want = de_by_ref[u.reference_id].rotation
            checked += 1
            if drawn is None or int(drawn) // 3 != int(want) // 3:
                diffs.append((u.unit_const, u.rotation, drawn, want))
    assert checked
    assert not diffs, diffs[:20]


def test_legacy_import_applies_exactly_on_measured_bands(pair) -> None:
    _name, legacy, de, _replaced = pair
    assert legacy_import.applies(legacy) == (legacy.source_format.game in legacy_import.MEASURED_BANDS)
    assert not legacy_import.applies(de)


def test_legacy_import_tree_remap_keeps_the_resaves_variety(pair, monkeypatch) -> None:
    """Distinct drawn variants per tree const, remap on, against the resave; and the control:
    with the remap off, the legacy oak draws at most the 14 old values."""
    _name, legacy, de, _replaced = pair
    if not legacy_import.applies(legacy):
        pytest.skip(f"{legacy.source_format.game} is not a measured band")
    overrides = legacy_import.drawn_rotation_overrides(legacy)

    def distinct(loaded, drawn):
        found: dict[int, set] = {}
        for pid, units in enumerate(loaded.unit_manager.units):
            for i, u in enumerate(units):
                if u.unit_const in legacy_import.TREE_CONSTS:
                    found.setdefault(u.unit_const, set()).add(drawn(pid, i, u))
        return found

    remapped = distinct(legacy, lambda pid, i, u: overrides.get((pid, i), u.rotation))
    resave = distinct(de, lambda pid, i, u: u.rotation)
    raw = distinct(legacy, lambda pid, i, u: u.rotation)
    assert remapped.keys() == resave.keys()
    for const, values in resave.items():
        # The importer picks among three at random; ours is reference_id % 3, so small samples differ a little.
        assert len(remapped[const]) >= min(len(values), 3 * len(raw[const])) * 0.8, (const, len(remapped[const]), len(values))

    # The control: the band switched off draws the stored values, at most the 14 old ones.
    monkeypatch.setattr(legacy_import, "MEASURED_BANDS", frozenset())
    off = legacy_import.drawn_rotation_overrides(legacy)
    assert off == {}
    for values in distinct(legacy, lambda pid, i, u: off.get((pid, i), u.rotation)).values():
        assert len(values) <= legacy_import.LEGACY_TREE_VARIANTS


# GH #316: the 1.6 fields each legacy effect, condition and trigger stores, as the library names them.
_LEGACY_EFFECT_FIELDS = (
    "effect_type", "ai_script_goal", "quantity", "tribute_list", "diplomacy", "location_object_reference",
    "object_list_unit_id", "source_player", "target_player", "technology", "string_id",
    "display_time", "trigger_id", "location_x", "location_y", "area_x1", "area_y1", "area_x2", "area_y2",
    "object_group", "object_type", "instruction_panel_position", "message", "sound_name", "selected_object_ids",
)  # fmt: skip
_LEGACY_CONDITION_FIELDS = (
    "condition_type", "quantity", "attribute", "unit_object", "next_object", "object_list", "source_player",
    "technology", "timer", "area_x1", "area_y1", "area_x2", "area_y2", "object_group", "object_type", "ai_signal",
)  # fmt: skip
_LEGACY_TRIGGER_FIELDS = (
    "name", "description", "enabled", "looping", "description_stid", "display_as_objective", "description_order",
)  # fmt: skip
_UNLISTED = {"effect_type", "condition_type"}  # the type itself, never a listed attribute


def _value(obj, attribute):
    value = getattr(obj, attribute)
    return tuple(value) if isinstance(value, list) else value


def _trigger_rows(manager):
    return [
        (
            tuple(_value(t, f) for f in _LEGACY_TRIGGER_FIELDS),
            [tuple(_value(e, f) for f in _LEGACY_EFFECT_FIELDS) for e in t.effects],
            [tuple(_value(c, f) for f in _LEGACY_CONDITION_FIELDS) for c in t.conditions],
        )
        for t in manager.triggers
    ]


def _crosses(loaded, manager) -> Counter:
    from descape import library_compat, trigger_status
    from descape.unit_references import build_reference_index

    vocabulary = library_compat.load_vocabulary(loaded.vocabulary_version)
    index = build_reference_index(loaded)
    context = trigger_status.StatusContext(
        vocabulary=vocabulary,
        trigger_ids=frozenset(range(len(manager.triggers))),
        unit_exists=lambda ref_id, by_id=index.by_id: ref_id in by_id,
    )
    return Counter(
        status.status
        for trigger in manager.triggers
        for kind in ("conditions", "effects")
        for status in getattr(trigger_status.evaluate_trigger(trigger, context), kind)
    )


def test_triggers_match_the_de_resave(pair) -> None:
    from descape import library_compat, scenario_io, trigger_model

    _name, legacy, de, _replaced = pair
    keep = [legacy, de]  # noqa: F841 -- the library holds scenarios weakly; a dropped one breaks Effect.quantity
    if legacy.trigger_version != 1.6:
        pytest.skip(f"trigger format {legacy.trigger_version:g} isn't read")
    legacy_manager = scenario_io.parse_triggers(legacy)
    legacy_rows = _trigger_rows(legacy_manager)
    legacy_order = list(legacy_manager.trigger_display_order)
    legacy_crosses = _crosses(legacy, scenario_io.parse_triggers(legacy))
    vocabulary = library_compat.load_vocabulary(legacy.vocabulary_version)
    unlisted = Counter()
    for trigger in scenario_io.parse_triggers(legacy).triggers:
        for kind, entries, defined, fields in (
            ("effect", trigger.effects, vocabulary.effects, _LEGACY_EFFECT_FIELDS),
            ("condition", trigger.conditions, vocabulary.conditions, _LEGACY_CONDITION_FIELDS),
        ):
            for entry in entries:
                definition = defined.get(getattr(entry, f"{kind}_type"))
                for field in fields:
                    if definition is None or field in _UNLISTED or field in definition.attributes:
                        continue
                    if _value(entry, field) not in (-1, "", (), None, definition.default_attributes.get(field)):
                        unlisted[(kind, definition.name, field)] += 1
    assert trigger_model.exec_order_value(legacy) == 1

    de_manager = scenario_io.parse_triggers(de)
    assert _trigger_rows(de_manager) == legacy_rows
    assert list(de_manager.trigger_display_order) == legacy_order
    assert trigger_model.exec_order_value(de) == 1, "DE's importer runs a legacy file in ID order"
    assert not unlisted, f"set in the file but hidden by the legacy vocabulary: {dict(unlisted)}"
    assert _crosses(de, scenario_io.parse_triggers(de)) == legacy_crosses
