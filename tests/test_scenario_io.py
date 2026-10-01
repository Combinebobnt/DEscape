"""Migrated from tools/verify_roundtrip.py (Phase 3 of the pytest migration
plan -- Ordering step 5, first script in the migration order).

Verifies the scenario_io.py loader shim: every file loads, and the
captured trigger_tail is byte-complete (i.e. concatenating everything
parsed before it with the tail reconstructs the full decompressed body).
This is the v1 (read-only) safety net -- see tests/test_write_path.py
(once migrated) for v2's write path, kept separate since it exercises
different, newer code with a different failure mode from this one's pure-
parsing concern.

The terrain fast path's oracle and fallback tests live here too: the fast
load against the library walk it replaces, on every fixture, a varied-tile
file and (corpus tier) every examples/ file.

Unlike the original standalone script, each check here runs twice: once
against the shipped blank template descape/templates/blank_120x120.aoe2scenario
(default tier -- also File > New's source, see descape/viewer.py -- the
"single biggest coverage win for a fresh clone" the migration plan's new-test
#5 calls out, since scenario_io.py's core load guarantees had no default-tier
coverage at all before this), and once parametrized against the real
examples/ corpus (corpus tier, preserving the original script's actual
behavior against every real file).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
from AoE2ScenarioParser.scenarios.aoe2_scenario import (
    _decompress_bytes,
    _get_file_version,
    _get_scenario_variant,
    _initialise_version_dependencies,
)

from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units


def _check_tail_completeness(path: Path) -> tuple[bool, str]:
    """Re-derives the decompressed body independently and confirms the
    loader's trigger_tail, combined with everything parsed before it,
    reconstructs it exactly."""
    igen = IncrementalGenerator.from_file(str(path))
    scenario_version = _get_file_version(igen)
    scenario_variant = _get_scenario_variant(igen)
    scenario = AoE2DEScenario(
        "DE", scenario_version, source_location=str(path), name="", variant=scenario_variant
    )
    scenario._load_structure()
    _initialise_version_dependencies(scenario.game_version, scenario.scenario_version)
    scenario._load_header_section(igen)
    decompressed = _decompress_bytes(igen.get_remaining_bytes())

    data_igen = IncrementalGenerator(name="Scenario Data", file_content=decompressed)
    scenario._decompressed_file_data = decompressed
    for section_name in scenario.structure:
        if section_name == "FileHeader":
            continue
        scenario._create_and_load_section(section_name, data_igen)
        if section_name == "Units":
            break
    progress = data_igen.progress
    tail = data_igen.get_remaining_bytes()

    recombined = decompressed[:progress] + tail
    if recombined != decompressed:
        return False, f"MISMATCH (progress={progress}, tail={len(tail)}, total={len(decompressed)})"
    return True, f"OK (progress={progress}, tail={len(tail)} bytes)"


def test_tail_completeness() -> None:
    ok, detail = _check_tail_completeness(FIXTURE_PATH)
    assert ok, detail


@pytest.mark.corpus
def test_tail_completeness_corpus(scenario_path) -> None:
    ok, detail = _check_tail_completeness(scenario_path)
    assert ok, detail


def test_scenario_loads_without_raising() -> None:
    """verify_roundtrip.py's own main() wrapped load_map_and_units(path) in
    its own try/except, separate from check_tail_completeness's own
    concern -- "every file in the directory loads without raising" is a
    real, separate assertion (see migration_manifest.py's 42+1+1
    reconciliation), not folded into the tail-completeness check above."""
    load_map_and_units(FIXTURE_PATH)


@pytest.mark.corpus
def test_scenario_loads_without_raising_corpus(scenario_path) -> None:
    load_map_and_units(scenario_path)


# -- Terrain fast path: tiles built off the raw bytes must equal the library
# walk's (fast_terrain=False, the fallback) field for field, along with every
# layout/trust value derived from them.

_FIXTURES = sorted((Path(__file__).resolve().parent / "fixtures").glob("*.aoe2scenario"))

_ORACLE_ATTRS = (
    "terrain_block_offset",
    "terrain_struct_size",
    "terrain_has_layer",
    "terrain_write_supported",
    "units_block_offset",
    "units_section_end",
    "players_units_end",
    "units_write_supported",
    "map_is_square",
)


def _tile_dicts(loaded) -> list[dict]:
    """Every tile's __dict__ minus `_uuid`, which is checked against its own
    load's scenario instead: two loads are two scenarios."""
    uuid = loaded._scenario.uuid
    out = []
    for tile in loaded.map_manager.terrain:
        fields = dict(vars(tile))
        assert fields.pop("_uuid") == uuid
        out.append(fields)
    return out


def _took_fast_path(loaded) -> bool:
    # The fast walk leaves terrain_data empty; the library one fills it per tile.
    return loaded._scenario.sections["Map"].retriever_map["terrain_data"].data == []


def _assert_paths_agree(library, fast) -> None:
    assert _took_fast_path(fast) and not _took_fast_path(library)
    mm_l, mm_f = library.map_manager, fast.map_manager
    assert (mm_f.map_width, mm_f.map_height) == (mm_l.map_width, mm_l.map_height)
    assert mm_f.map_color_mood == mm_l.map_color_mood
    for attr in _ORACLE_ATTRS:
        assert getattr(fast, attr) == getattr(library, attr), attr
    assert fast._scenario.sections["Map"].byte_length == library._scenario.sections["Map"].byte_length
    fast_tiles, library_tiles = _tile_dicts(fast), _tile_dicts(library)
    assert len(fast_tiles) == mm_l.map_width * mm_l.map_height
    mismatched = [i for i, (a, b) in enumerate(zip(fast_tiles, library_tiles, strict=True)) if a != b]
    if mismatched:
        first = mismatched[0]
        pytest.fail(f"{len(mismatched)} tiles differ, first {first}: {fast_tiles[first]} != {library_tiles[first]}")


def test_fast_terrain_oracle_covers_the_named_fixtures() -> None:
    """An empty glob would parametrize the oracle below to nothing."""
    names = {path.name for path in _FIXTURES}
    assert {"real_blank_240x240.aoe2scenario", "units_120x120.aoe2scenario", "v159_units_triggers.aoe2scenario"} <= names


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.name)
def test_fast_terrain_matches_the_library_walk(path: Path) -> None:
    _assert_paths_agree(load_map_and_units(path, fast_terrain=False), load_map_and_units(path))


def test_fast_terrain_matches_the_library_walk_on_varied_tiles(tmp_path: Path) -> None:
    """Every fixture's tiles are uniform (layer -1 over "00 ff ff" padding
    reads as -1 through a misplaced s32 too), so this writes varied
    terrain/elevation/layer, negative layers included, through the real
    write path first."""
    from descape.scenario_write import write_scenario

    source = load_map_and_units(FIXTURE_PATH, fast_terrain=False)
    for i, tile in enumerate(source.map_manager.terrain):
        tile.terrain_id = i % 53
        tile.elevation = (i // 7) % 8
        tile.layer = (i * 37) % 301 - 150
    out = tmp_path / "varied.aoe2scenario"
    write_scenario(source, out, backup=False)
    library = load_map_and_units(out, fast_terrain=False)
    assert len({tile.layer for tile in library.map_manager.terrain}) > 100
    _assert_paths_agree(library, load_map_and_units(out))


@pytest.mark.corpus
def test_fast_terrain_matches_the_library_walk_corpus(scenario_path) -> None:
    _assert_paths_agree(load_map_and_units(scenario_path, fast_terrain=False), load_map_and_units(scenario_path))


def test_fast_terrain_new_map_matches_the_library_walk() -> None:
    """File > New Map's bytes-only load takes the same path."""
    from descape.scenario_io import load_map_and_units_from_bytes
    from descape.scenario_new import blank_scenario_bytes

    data = blank_scenario_bytes(144)
    _assert_paths_agree(
        load_map_and_units_from_bytes(data, "blank.aoe2scenario", fast_terrain=False),
        load_map_and_units_from_bytes(data, "blank.aoe2scenario"),
    )


def test_fast_load_leaves_map_manager_links_pristine() -> None:
    """The prebuilt-tiles relink is undone right after MapManager.construct(),
    so the class never keeps a closed document's tiles alive."""
    from AoE2ScenarioParser.objects.managers.map_manager import MapManager

    from descape import library_compat

    loaded = load_map_and_units(FIXTURE_PATH)
    assert _took_fast_path(loaded)
    assert library_compat.class_state_delta(MapManager) == ([], [])


_DE_TERRAIN_STRUCT = {
    "terrain_id": {"type": "u8"},
    "elevation": {"type": "u8"},
    "unused": {"type": "3"},
    "layer": {"type": "s16"},
}


def test_terrain_struct_format_is_derived_from_the_model() -> None:
    from descape import scenario_io

    de = scenario_io._terrain_struct_format(_DE_TERRAIN_STRUCT)
    assert (de.unpack.format, de.fields) == ("<BB3xh", ("terrain_id", "elevation", "layer"))
    assert dict(de.layout) == {"terrain_id": (0, "u8"), "elevation": (1, "u8"), "layer": (5, "s16")}
    v121 = scenario_io._terrain_struct_format(
        {"terrain_id": {"type": "u8"}, "elevation": {"type": "u8"}, "unused": {"type": "1"}}
    )
    assert (v121.unpack.format, v121.fields) == ("<BB1x", ("terrain_id", "elevation"))


def test_fast_terrain_tiles_without_layer_keep_the_default() -> None:
    from descape import scenario_io

    fmt = scenario_io._terrain_struct_format(
        {"elevation": {"type": "u8"}, "unused": {"type": "1"}, "terrain_id": {"type": "u8"}}
    )
    tiles = scenario_io._fast_terrain_tiles(fmt, bytes([3, 0x5C, 9, 4, 0x5C, 10]), uuid="u")
    assert [(t.terrain_id, t.elevation, t.layer, t._index) for t in tiles] == [(9, 3, -1, 0), (10, 4, -1, 1)]


@pytest.mark.parametrize(
    "change",
    [
        {"unused": {"type": "u8", "repeat": 3}},
        {"unused": {"type": "c3"}},
        {"layer": {"type": "2"}},
        {"layer": {"type": "s16", "dependencies": {"on_construct": {"action": "REFRESH_SELF"}}}},
        {"terrain_id": {"type": "str16"}},
    ],
    ids=["repeat", "char-type", "linked-raw-bytes", "dependency", "variable-width"],
)
def test_terrain_struct_format_refuses_what_is_not_fixed_width(change: dict) -> None:
    from descape import scenario_io

    assert scenario_io._terrain_struct_format({**_DE_TERRAIN_STRUCT, **change}) is None


def _fast_layout(model=None, *, repeat=6, end=100 + 42, w=3, h=2, next_section="Units", units=True):
    from descape import scenario_io

    fmt = scenario_io._terrain_struct_format(model or _DE_TERRAIN_STRUCT)
    fast = scenario_io._FastTerrain(tiles=[], block_offset=100, repeat=repeat, terrain_format=fmt)
    return scenario_io._fast_terrain_layout(fast, end, w, h, next_section, units)


def test_fast_terrain_layout_trusts_an_exact_block() -> None:
    assert _fast_layout() == (100, 7, True, True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end": 100 + 43},
        {"repeat": 5},
        {"w": 0, "h": 0, "repeat": 0, "end": 100},
        {"next_section": "Triggers"},
        {"units": False},
        {"model": {**_DE_TERRAIN_STRUCT, "unused": {"type": "1"}, "layer": {"type": "s16"}, "pad": {"type": "2"}}},
        {"model": {"elevation": {"type": "u8"}, "terrain_id": {"type": "u8"}, "unused": {"type": "3"}, "layer": {"type": "s16"}}},
    ],
    ids=["not-map-end", "repeat", "empty-map", "units-not-next", "units-unverified", "layer-moved", "fields-swapped"],
)
def test_fast_terrain_layout_distrusts_a_wrong_skip(kwargs: dict) -> None:
    offset, _stride, _layer, trusted = _fast_layout(**kwargs)
    assert (offset, trusted) == (-1, False)


@pytest.mark.parametrize("enabled", [True, False], ids=["gc-on", "gc-off"])
def test_load_pauses_gc_and_restores_the_callers_state(enabled: bool, monkeypatch) -> None:
    """GC is off for the section walk, and the caller finds it exactly as it
    left it: on stays on, off stays off."""
    import gc

    from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

    during: list[bool] = []
    original = AoE2Scenario._create_and_load_section

    def spy(self, name, igen):
        during.append(gc.isenabled())
        return original(self, name, igen)

    monkeypatch.setattr(AoE2Scenario, "_create_and_load_section", spy)
    was_enabled = gc.isenabled()
    try:
        if enabled:
            gc.enable()
        else:
            gc.disable()
        load_map_and_units(FIXTURE_PATH)
        after = gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
    assert during and not any(during)
    assert after is enabled


def _spy_sections(monkeypatch) -> list[str]:
    from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

    seen: list[str] = []
    original = AoE2Scenario._create_and_load_section

    def spy(self, name, igen):
        seen.append(name)
        return original(self, name, igen)

    monkeypatch.setattr(AoE2Scenario, "_create_and_load_section", spy)
    return seen


def test_non_fixed_terrain_struct_falls_back_to_the_library_walk(monkeypatch) -> None:
    """A structure copy whose TerrainStruct pads with a repeated u8 (same
    bytes, not a fixed single field) takes the library walk and still loads,
    equal to a forced library load of the same structure."""
    original = AoE2DEScenario._load_structure

    def load_structure(self):
        original(self)
        self.structure["Map"]["structs"]["TerrainStruct"]["retrievers"]["unused"] = {
            "type": "u8",
            "repeat": 3,
            "default": [0, 255, 255],
        }

    monkeypatch.setattr(AoE2DEScenario, "_load_structure", load_structure)
    seen = _spy_sections(monkeypatch)
    loaded = load_map_and_units(FIXTURE_PATH)
    assert "Map" in seen, "the Map section never went through the library walk"
    assert not _took_fast_path(loaded)
    library = load_map_and_units(FIXTURE_PATH, fast_terrain=False)
    assert _tile_dicts(loaded) == _tile_dicts(library)
    assert loaded.terrain_write_supported and loaded.terrain_block_offset == library.terrain_block_offset


class _MapHandoff(Exception):
    def __init__(self, progress: int) -> None:
        self.progress = progress


def test_terrain_block_past_the_body_falls_back_to_the_library_walk(monkeypatch) -> None:
    """map_height inflated so terrain_data would run past the body: the fast
    walk rewinds to Map's start and hands Map to the library walk. Stopped
    at the handoff, since the library would only fail its own way after
    printing every struct it read."""
    from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

    from descape.scenario_io import load_map_and_units_from_bytes
    from descape.scenario_new import _SIZE_STRUCT
    from descape.scenario_write import _compress_bytes

    donor = load_map_and_units(FIXTURE_PATH)
    body = bytearray(donor.decompressed_body)
    _SIZE_STRUCT.pack_into(body, donor.terrain_block_offset - _SIZE_STRUCT.size, 120, 100_000)
    data = donor.header_bytes + _compress_bytes(bytes(body))

    ends: dict[str, int] = {}
    original = AoE2Scenario._create_and_load_section

    def spy(self, name, igen):
        if name == "Map":
            raise _MapHandoff(igen.progress)
        original(self, name, igen)
        ends[name] = igen.progress

    monkeypatch.setattr(AoE2Scenario, "_create_and_load_section", spy)
    with pytest.raises(_MapHandoff) as handoff:
        load_map_and_units_from_bytes(data, "inflated.aoe2scenario")
    assert handoff.value.progress == ends["Options"]
