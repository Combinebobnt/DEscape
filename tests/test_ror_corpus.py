"""GH #67 PR 2: the Return of Rome write-path rules, re-measured on real RoR files.

Only the corpus files whose header says Return of Rome (variant 2) are used,
with RoR game data active; elsewhere every test skips. Point it at RoR files:

    pytest -m corpus tests/test_ror_corpus.py --scenario-dir <RoR folder> --corpus-full

The other corpus tests assume AoE2 files and are not run on that folder.
Measured on 8 RoR scenarios (100,098 placements) on 2026-10-17; AGENTS.md
quotes these figures.
"""

from __future__ import annotations

import functools

import pytest

from descape import (
    game_dataset,
    gate_orientation,
    object_catalog,
    option_fields,
    render,
    scenario_io,
    terrain_palette,
    unit_rotation,
    unit_sprites,
    unit_variant,
)
from descape.messages_model import MessagesEditModel
from descape.options_model import diplomacy_write_supported, options_write_supported, players_write_supported
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel
from descape.unit_model import UnitEditModel, span_low_corner

pytestmark = pytest.mark.corpus

ZERO = 1e-6


@functools.cache
def _is_ror(path) -> bool:
    """One full load per file per session, not one per test: an AoE2 corpus run only skips here."""
    return scenario_io.load_map_and_units(path).game_variant == game_dataset.ROR_VARIANT


@pytest.fixture
def ror_files(corpus_files):
    files = [p for p in corpus_files if _is_ror(p)]
    if not files:
        pytest.skip("no Return of Rome files in the scenario dir")
    game_dataset.set_active(game_dataset.ROR)
    return files


def _placements(files):
    for path in files:
        loaded = scenario_io.load_map_and_units(path)
        for player_id, units in enumerate(loaded.unit_manager.units):
            for unit in units:
                yield path, loaded, player_id, unit


def _is_int(value: float) -> bool:
    return abs(value - round(value)) < ZERO


def test_every_placed_const_is_a_ror_object(ror_files):
    dat = object_catalog.dat_objects()
    unknown = {(p.name, u.unit_const) for p, _l, _pid, u in _placements(ror_files) if dat.get(u.unit_const) is None}
    assert not unknown, sorted(unknown)[:10]


def test_the_angle_whitelist_has_zero_counterexamples(ror_files):
    counterexamples, seen = [], 0
    for path, _loaded, _pid, unit in _placements(ror_files):
        if not unit_rotation.rotation_is_angle(unit.unit_const):
            continue
        seen += 1
        count = unit_rotation.angle_count_for(unit.unit_const)
        if _is_int(unit.rotation) and 0 <= round(unit.rotation) < count and abs(unit.rotation) > ZERO:
            counterexamples.append((path.name, unit.unit_const, unit.rotation))
    assert seen, "no ANGLE placement, so this proves nothing"
    assert not counterexamples, counterexamples[:10]


def test_walls_store_literal_indices_frame_0_and_agree_with_their_neighbours(ror_files):
    agree = total = walls = 0
    for path in ror_files:
        loaded = scenario_io.load_map_and_units(path)
        connectors = unit_sprites.wall_connector_consts()
        tiles = {
            (int(u.x), int(u.y))
            for units in loaded.unit_manager.units for u in units if u.unit_const in connectors
        }
        for units in loaded.unit_manager.units:
            for unit in units:
                if not unit_sprites.rotation_variant_eligible(unit.unit_const):
                    continue
                walls += 1
                assert _is_int(unit.rotation) and 0 <= round(unit.rotation) <= 4, (path.name, unit.rotation)
                assert unit.initial_animation_frame == 0, path.name
                derived = unit_sprites.wall_variant_from_neighbours(
                    unit_sprites.neighbour_mask(int(unit.x), int(unit.y), tiles)
                )
                if derived is not None:
                    total += 1
                    agree += derived == round(unit.rotation)
    assert walls, "no wall placement"
    assert total and agree / total >= 0.95, f"{agree}/{total}"


def test_cliffs_store_their_frame_as_an_integer_rotation(ror_files):
    from descape import cliff_catalog

    cliffs = 0
    for path, _loaded, _pid, unit in _placements(ror_files):
        if unit.unit_const not in unit_sprites.cliff_consts():
            continue
        cliffs += 1
        assert _is_int(unit.rotation) and round(unit.rotation) == unit.initial_animation_frame, path.name
        assert 0 <= unit.initial_animation_frame < cliff_catalog.frame_count(unit.unit_const), path.name
    assert cliffs, "no cliff placement"


def test_gates_carry_no_rotation_and_sit_on_their_span_anchor(ror_files):
    gates = {c for group in gate_orientation.groups().values() for c in group}
    seen = 0
    for path, _loaded, _pid, unit in _placements(ror_files):
        if unit.unit_const not in gates:
            continue
        seen += 1
        assert unit.rotation in (0.0, 7.0), (path.name, unit.rotation)
        span = terrain_palette.tile_span(unit.unit_const, render.NON_BUILDING_SPAN)
        assert (unit.x, unit.y) == render.span_anchor(*span_low_corner(unit), *span), path.name
    assert seen, "no gate placement"


def test_cyclable_placements_store_literal_indices(ror_files):
    seen = 0
    for path, _loaded, _pid, unit in _placements(ror_files):
        if unit_variant.is_cyclable(unit.unit_const):
            seen += 1
            assert _is_int(unit.rotation), (path.name, unit.unit_const, unit.rotation)
    assert seen, "no cyclable placement"


def test_every_write_gate_opens(ror_files):
    for path in ror_files:
        loaded = scenario_io.load_map_and_units(path)
        assert loaded.units_write_supported and loaded.terrain_write_supported, path.name
        assert players_write_supported(loaded), path.name
        assert diplomacy_write_supported(loaded), path.name
        MessagesEditModel(loaded)
        TriggerEditModel(loaded)
        specs = option_fields.specs_for(loaded)
        if any(not isinstance(_retriever(loaded, spec).data, int) for spec in specs):
            continue  # GH #240: v1.49 types secondary_game_modes as raw bytes
        assert options_write_supported(loaded, specs), path.name


def _header_fields(loaded) -> dict:
    """FileHeader's fields by name; the DLC list and variant must survive a save."""
    return {name: r.data for name, r in loaded._scenario.sections["FileHeader"].retriever_map.items()}


def _retriever(loaded, spec):
    return loaded._scenario.sections[spec.section].retriever_map[spec.retriever]


def test_an_edited_ror_file_round_trips(ror_files, tmp_path):
    for path in ror_files:
        loaded = scenario_io.load_map_and_units(path)
        old_header = _header_fields(loaded)
        model = UnitEditModel(loaded)
        before = len(loaded.unit_manager.get_all_units())
        walls = [model.add(1, 117, 2.5 + i, 2.5) for i in range(3)]
        for wall in walls:
            model.set_wall_variant(wall, 0)
        angled = next(
            (u for u in loaded.unit_manager.get_all_units() if unit_rotation.rotation_is_angle(u.unit_const)), None
        )
        if angled is not None:
            model.set_rotation(angled, 0.5)
        triggers = TriggerEditModel(loaded)
        trigger_count = triggers.trigger_count
        triggers.structural_edit(lambda manager: manager.add_trigger("RoR round trip"))
        out = tmp_path / path.name
        write_scenario(loaded, out, units=model, triggers=triggers, backup=False)
        reloaded = scenario_io.load_map_and_units(out)
        assert TriggerEditModel(reloaded).trigger_count == trigger_count + 1, path.name
        new_header = _header_fields(reloaded)
        assert new_header.pop("trigger_count") == old_header.pop("trigger_count") + 1, path.name
        assert new_header == old_header, path.name
        assert reloaded.game_variant == game_dataset.ROR_VARIANT
        assert len(reloaded.unit_manager.get_all_units()) == before + 3, path.name
