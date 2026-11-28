"""legacy_import: AoE2:DE's importer encodings for a pre-DE file shown with DE data (TASK-685)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import civ_art, civ_availability, legacy_import, player_fields, render_units, unit_sprites
from descape.legacy_import import U32_MINUS_ONE, starting_age_from_de, starting_age_to_de, tree_variant
from descape.scenario_format import AOC, AOK, DE_FORMAT, HD, ScenarioFormat
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest

UNITS = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def test_starting_age_maps_both_ways() -> None:
    assert [starting_age_to_de(v) for v in (-1, U32_MINUS_ONE, 0, 1, 2, 3, 4)] == [0, 0, 2, 3, 4, 5, 6]
    assert [starting_age_from_de(v) for v in (0, 2, 3, 4, 5, 6)] == [U32_MINUS_ONE, 0, 1, 2, 3, 4]
    for raw in (U32_MINUS_ONE, 0, 1, 2, 3, 4):
        assert starting_age_from_de(starting_age_to_de(raw)) == raw
    # Outside the legacy values a value passes through, so to_de is not injective there.
    assert starting_age_to_de(6) == starting_age_to_de(4) == 6
    assert starting_age_to_de("x") == "x" and starting_age_to_de(True) is True


def test_tree_variant_range_and_refusals() -> None:
    for v in range(legacy_import.LEGACY_TREE_VARIANTS):
        for ref in range(3):
            assert tree_variant(float(v), ref) == 3 * v + ref
    assert tree_variant(13, 5) == 41.0
    for stored in (14.0, -1.0, 2 * math.pi / 5, 0.5, None):
        assert tree_variant(stored, 0) is None, stored


@pytest.mark.parametrize(
    ("fmt", "applies"),
    [
        (ScenarioFormat("1.21", 2, "1.22", AOC, True), True),
        (ScenarioFormat("1.21", 2, "1.20", AOK, True), False),  # unmeasured band
        (ScenarioFormat("1.21", 2, "1.26", HD, True), False),
        (ScenarioFormat("1.21", 3, None, HD, False), False),  # header-3 HD: not legacy
        (DE_FORMAT, False),
    ],
)
def test_applies_only_on_a_measured_legacy_band(fmt, applies) -> None:
    assert legacy_import.applies(SimpleNamespace(source_format=fmt)) is applies


def test_applies_is_false_without_a_source_format() -> None:
    assert legacy_import.applies(SimpleNamespace()) is False


def test_tree_consts_never_collide_with_wall_override_keys() -> None:
    assert not [c for c in legacy_import.TREE_CONSTS if unit_sprites.rotation_variant_eligible(c)]


def _trees(loaded):
    return {(p, i): u for p, units in enumerate(loaded.unit_manager.units) for i, u in enumerate(units)
            if u.unit_const in legacy_import.TREE_CONSTS}  # fmt: skip


def test_legacy_trees_draw_through_the_rotation_overrides_and_stay_stored() -> None:
    loaded = conftest.legacy_loaded(UNITS)
    trees = _trees(loaded)
    assert trees, "the fixture carries trees"
    stored = {k: u.rotation for k, u in trees.items()}
    overrides = render_units.wall_variant_rotation_overrides(loaded)
    want = {k: tree_variant(u.rotation, u.reference_id) for k, u in trees.items()}
    want = {k: v for k, v in want.items() if v is not None}
    assert want, "the fixture has an in-range legacy variant"
    assert {k: overrides[k] for k in want} == want
    assert not [k for k, v in stored.items() if v not in range(14) and k in overrides]  # out-of-range keeps its value
    assert {k: u.rotation for k, u in trees.items()} == stored  # draw-only

    de = load_map_and_units(UNITS)
    assert not set(_trees(de)) & set(render_units.wall_variant_rotation_overrides(de))


def test_splice_overrides_cover_a_legacy_tree() -> None:
    from descape import render_cache_splice

    loaded = conftest.legacy_loaded(UNITS)
    key, unit = next((k, u) for k, u in _trees(loaded).items() if tree_variant(u.rotation, u.reference_id) is not None)
    splice = SimpleNamespace(unit=unit)
    assert key in render_cache_splice._splice_overrides(loaded, [splice])
    assert render_cache_splice._splice_overrides(load_map_and_units(UNITS), [SimpleNamespace(unit=unit)]) == {}


def _set_stored_age(loaded, player_id: int, value: int) -> None:
    spec = next(s for s in player_fields.specs_for(loaded) if s.field_id == "starting_age")
    data = loaded._scenario.sections[spec.section].retriever_map[spec.retriever].data
    data[spec.layout.index_for(player_id)] = value


def test_a_legacy_starting_age_reads_in_des_encoding_for_display_only() -> None:
    loaded = conftest.legacy_loaded()
    _set_stored_age(loaded, 1, 3)  # legacy Imperial
    spec = next(s for s in player_fields.specs_for(loaded) if s.field_id == "starting_age")
    assert player_fields.current_value(loaded, spec, 1) == 3
    assert player_fields.shown_value(loaded, spec, 1) == 5
    assert civ_art.player_art(loaded)[1][1] == civ_art.art_age(5)
    assert civ_availability.player_start_age(loaded, 1) == 5
    assert civ_availability.start_age_text(civ_availability.player_start_age(loaded, 1)) == "the Imperial Age"

    _set_stored_age(loaded, 2, U32_MINUS_ONE)
    assert player_fields.shown_value(loaded, spec, 2) == 0  # as a DE file's "none"

    de = load_map_and_units(BLANK_TEMPLATE_PATH)
    _set_stored_age(de, 1, 3)
    assert player_fields.shown_value(de, spec, 1) == 3


def test_the_players_panel_shows_a_legacy_age_decoded() -> None:
    from descape.players_panel import PlayersPanel

    conftest.ensure_qapp()
    loaded = conftest.legacy_loaded()
    _set_stored_age(loaded, 1, 3)
    de = load_map_and_units(BLANK_TEMPLATE_PATH)
    _set_stored_age(de, 1, 5)
    texts = []
    for doc in (loaded, de):
        panel = PlayersPanel()
        try:
            panel.show_scenario(doc)
            assert panel.current_values()["starting_age"] == 5
            texts.append(panel.widget_for("starting_age").currentText())
        finally:
            panel.deleteLater()
    assert texts[0] == texts[1] and "Imperial" in texts[0]
