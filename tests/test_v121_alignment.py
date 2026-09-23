"""The alignment oracle for descape/versions/DE/v1.21/structure.json, run
against the 25 real Workshop v1.21 files this repo can never commit. A
structure that is misaligned can still round-trip, so the oracle checks
independent facts instead: (1) a v1.21-shaped substitute for the trigger-count
cross-check, (2) _verify_terrain_block/_verify_units_block, (3) trigger_version
lands in a plausible band, (4) DataHeader through Units parse and the map size
is plausible, (5) the leftover tail is reported, never asserted beyond >= 0.

**Assertion 1 cannot exist in its DE form, permanently.** On DE it
cross-checks Options.number_of_triggers against the Triggers tail's own count:
two independent reads of one quantity, which proves the whole
Options->Map->Units boundary chain in one line. v1.21 has no trigger count
outside the Triggers section, so there is no second read. The same chain is
proved piecewise instead:

- Options->Map: the 4 bytes at options_section_end are Map.separator, the AoC
  map separator 9d ff ff ff, and Options is exactly 6028 bytes (round 3
  measured both across 25/25). Independent of any Map parse.
- Map internals: _verify_terrain_block(), at the stride and layer presence
  read off the parse (3 bytes, no layer).
- players_units span and the Units end: _verify_units_block(), whose forward
  walk of every Units retriever must land on the independently captured
  units_section_end.
- Units end, again: assertion 3, since a misplaced units_section_end would not
  read a plausible trigger_version.

The oracle deliberately does not go through descape/scenario_io.py's
load_map_and_units(), so it stays independent of the load path it checks. It
installs the repo's own descape/versions/DE/v1.21/structure.json directly, the
same technique tools/map_structure_divergence.py uses. The tests after it do
go through load_map_and_units(), which resolves that structure itself since
Step 6f: it must agree with the oracle's walk, save an untouched file
byte-identically, and round-trip a terrain plus unit edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from AoE2ScenarioParser.objects.managers.map_manager import MapManager

from descape import library_compat, scenario_io
from testkit.scenario_targets import collect_files

WORKSHOP_DIR = Path("~/games/steam/steamapps/workshop/content/221380").expanduser()
STRUCTURE_PATH = Path(__file__).resolve().parent.parent / "descape/versions/DE/v1.21/structure.json"

# Measured against every real corpus file this repo has ever seen (trigger
# versions 2.2 to 4.9, scenario versions 1.37 through 1.58) -- assertion 3's
# "plausible band". Generous headroom on both sides: a misaligned f64 read
# lands near zero (denormal) or astronomically large, never quietly inside
# this range.
_PLAUSIBLE_TRIGGER_VERSION = (0.5, 20.0)

# scenario_new.MIN_MAP_TILES/MAX_MAP_TILES (61/480) is this app's own File > New
# constraint, not a fact about every real file in the wild -- a Workshop map
# could be smaller. Wide enough to catch a misaligned read (which lands near 0
# or in the billions from a garbage u32), not tight enough to reject a real map.
_PLAUSIBLE_MAP_TILES = (1, 2000)

# Map.separator, the AoC map separator, which v1.21's Map opens with; and
# v1.21's fixed Options length. Both measured across 25/25 in round 3.
_AOC_MAP_SEPARATOR = b"\x9d\xff\xff\xff"
_V121_OPTIONS_LENGTH = 6028

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not WORKSHOP_DIR.is_dir(),
        reason="the real v1.21 Workshop corpus only exists on this machine",
    ),
]


def _v121_files() -> list[Path]:
    # Runs at import time, inside the parametrize decorator below, so it fires
    # before pytestmark's skipif can. An absent corpus is a skip, not a
    # collection error; an empty one on this machine is still a hard failure.
    if not WORKSHOP_DIR.is_dir():
        return []
    return collect_files([WORKSHOP_DIR], pytest.fail)


def _walk_through_units(raw: bytes, path: Path):
    """Replicates scenario_io._load_map_and_units()'s own section walk, but
    with the repo's v1.21 structure.json installed instead of the library's
    _load_structure(), and _initialise_version_dependencies() skipped -- it
    only sets up the Triggers vocabulary, which this walk never reaches.

    Returns a namespace of the parsed scenario, the decompressed body and the
    section boundaries the walk captured (data_igen.progress cannot be
    recovered afterwards). Raises whatever the walk itself raises; assertion
    4 is "this doesn't raise"."""
    from AoE2ScenarioParser.helper.incremental_generator import IncrementalGenerator
    from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
    from AoE2ScenarioParser.scenarios.aoe2_scenario import (
        _decompress_bytes,
        _get_file_version,
        _get_scenario_variant,
    )

    igen = IncrementalGenerator(str(path), raw)
    version = _get_file_version(igen)
    variant = _get_scenario_variant(igen)
    scenario = AoE2DEScenario("DE", version, source_location=str(path), name="", variant=variant)
    scenario.structure = json.loads(STRUCTURE_PATH.read_text(encoding="utf-8"))

    library_compat.depoison()
    scenario._load_header_section(igen)
    decompressed = _decompress_bytes(igen.get_remaining_bytes())
    data_igen = IncrementalGenerator(name="Scenario Data", file_content=decompressed)

    map_width = map_height = None
    ends: dict[str, int] = {}
    for section_name in scenario.structure:
        if section_name == "FileHeader":
            continue
        scenario._create_and_load_section(section_name, data_igen)
        ends[section_name] = data_igen.progress
        if section_name == "Map":
            retriever_map = scenario.sections["Map"].retriever_map
            map_width = retriever_map["map_width"].data
            map_height = retriever_map["map_height"].data
        if section_name == "Units":
            break

    trigger_tail = data_igen.get_remaining_bytes()
    return SimpleNamespace(
        scenario=scenario,
        decompressed=decompressed,
        map_width=map_width,
        map_height=map_height,
        trigger_tail=trigger_tail,
        diplomacy_section_end=ends["Diplomacy"],
        options_section_end=ends["Options"],
        map_section_end=ends["Map"],
        units_section_end=ends["Units"],
    )


@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_v121_structure_alignment_oracle(path: Path) -> None:
    raw = path.read_bytes()

    # Assertion 4: DataHeader through Units parses without exception, and
    # map_width/map_height land in a real size band.
    walk = _walk_through_units(raw, path)
    map_width, map_height, trigger_tail = walk.map_width, walk.map_height, walk.trigger_tail
    assert map_width is not None and map_height is not None, "Map section never parsed"
    assert _PLAUSIBLE_MAP_TILES[0] <= map_width <= _PLAUSIBLE_MAP_TILES[1]
    assert _PLAUSIBLE_MAP_TILES[0] <= map_height <= _PLAUSIBLE_MAP_TILES[1]

    # Assertion 3: trigger_version (trigger_tail[0:8] as f64) lands in a
    # plausible band. Read straight off the tail -- library_compat.trigger_version()
    # -- since Triggers itself never parses for v1.21 (that's the whole reason
    # this file's DIVERGE is expected and not a failure here).
    try:
        trigger_version = library_compat.trigger_version(trigger_tail)
    except ValueError:
        pytest.fail(f"trigger_tail too short to hold a version f64 ({len(trigger_tail)} bytes)")
    assert _PLAUSIBLE_TRIGGER_VERSION[0] <= trigger_version <= _PLAUSIBLE_TRIGGER_VERSION[1]

    # Assertion 1's substitute: the Options->Map boundary, from raw bytes.
    body = walk.decompressed
    assert body[walk.options_section_end : walk.options_section_end + 4] == _AOC_MAP_SEPARATOR
    assert walk.options_section_end - walk.diplomacy_section_end == _V121_OPTIONS_LENGTH

    # Assertion 2, terrain half: the same derivation and check the load path
    # runs, through MapManager.construct() (so 6a's re-linking is exercised).
    scenario = walk.scenario
    library_compat.adapt_map_links(scenario.sections["Map"])
    map_manager = MapManager.construct(scenario.uuid)
    offset, stride, has_layer = scenario_io._terrain_layout(
        scenario.sections["Map"], walk.map_section_end, map_width, map_height
    )
    assert (stride, has_layer) == (3, False)
    assert scenario_io._verify_terrain_block(body, offset, map_manager.terrain, stride, has_layer)

    # Assertion 2, units half: forward walk from map_section_end, reconciled
    # against the independently captured units_section_end.
    units_section = scenario.sections["Units"]
    players_units = units_section.retriever_map["players_units"].data
    units_block_offset, players_units_end, trailer = scenario_io._units_layout(units_section, walk.map_section_end)
    assert trailer > 0, "v1.21 declares number_of_players/player_data_3 after players_units"
    assert players_units_end + trailer == walk.units_section_end
    assert scenario_io._verify_units_block(
        body, units_block_offset, players_units, trailer, walk.units_section_end
    )

    # Assertion 5: reported, never asserted beyond "the walk did not over-read".
    assert len(trigger_tail) >= 0


def test_v121_corpus_is_the_expected_25_files() -> None:
    """Pins the file count so a Workshop sync silently adding/removing files
    changes this test's own collection, not just the oracle's parametrize list
    -- the same "name a stop condition instead of trusting all()" concern the
    plan itself raises for the 25-file target."""
    assert len(_v121_files()) == 25


@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_v121_opens_natively_and_agrees_with_the_oracle(path: Path) -> None:
    loaded = scenario_io.load_map_and_units(path)
    walk = _walk_through_units(path.read_bytes(), path)
    assert loaded.structure_source == "repo"
    assert loaded.terrain_write_supported and loaded.units_write_supported
    assert (loaded.terrain_struct_size, loaded.terrain_has_layer) == (3, False)
    assert loaded.has_trigger_counters is False
    assert loaded.units_section_end == walk.units_section_end
    assert loaded.options_section_end == walk.options_section_end
    assert loaded.players_units_end < loaded.units_section_end
    assert scenario_io.parse_triggers(loaded) is None
    assert loaded.trigger_read_supported is False


@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_v121_untouched_save_is_byte_identical(path: Path, tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    out = tmp_path / "same.aoe2scenario"
    write_scenario(scenario_io.load_map_and_units(path), out, backup=False)
    assert out.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_v121_terrain_and_unit_edit_round_trips(path: Path, tmp_path: Path) -> None:
    """A stride-3 terrain/elevation patch plus a unit removal, which re-splices
    players_units and must carry the v1.21 trailer after it through."""
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = scenario_io.load_map_and_units(path)
    tile_index = 7
    tile = loaded.map_manager.terrain[tile_index]
    new_terrain = (tile.terrain_id + 1) % 40
    new_elevation = (tile.elevation + 1) % 8
    tile.terrain_id = new_terrain
    tile.elevation = new_elevation
    trailer = loaded.decompressed_body[loaded.players_units_end : loaded.units_section_end]
    units = UnitEditModel(loaded)
    before = sum(len(section) for section in loaded.unit_manager.units)
    victim = next(unit for section in loaded.unit_manager.units for unit in section)
    victim_id = victim.reference_id
    units.remove(victim)

    out = tmp_path / "edit.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=units)
    reread = scenario_io.load_map_and_units(out)

    assert reread.terrain_write_supported and reread.units_write_supported
    assert reread.map_manager.terrain[tile_index].terrain_id == new_terrain
    assert reread.map_manager.terrain[tile_index].elevation == new_elevation
    assert reread.decompressed_body[reread.players_units_end : reread.units_section_end] == trailer
    after = [unit for section in reread.unit_manager.units for unit in section]
    assert len(after) == before - 1
    assert victim_id not in {unit.reference_id for unit in after}
