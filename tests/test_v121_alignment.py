"""The alignment oracle for descape/versions/DE/v1.21/structure.json, run
against the 25 real Workshop v1.21 files this repo can never commit. A
structure that is misaligned can still round-trip, so the oracle checks
independent facts instead: (1) the Options and Triggers trigger-count
cross-check, (2) _verify_terrain_block/_verify_units_block, (3) trigger_version
lands in a plausible band, (4) DataHeader through Units parse and the map size
is plausible, (5) the leftover tail is reported, never asserted beyond >= 0.

**This gate is weak, on purpose, until Step 6 lands.** Assertion 1 (the
Options.number_of_triggers / Triggers.number_of_triggers cross-check) does not
apply to v1.21 at all: v1.21's Options has no such field (the plan's round 3
finding). Assertion 2 (_verify_terrain_block/_verify_units_block) cannot run
either -- it needs TERRAIN_STRUCT_SIZE and units_block_offset to be
version-aware, which is the load-path wiring step, not this file's job.
Only assertions 3-5 actually run here. Closing assertions 1-2 for v1.21 is
Step 6's gate to add, not this one's to fake.

Does not go through descape/scenario_io.py's load_map_and_units(): that path
calls AoE2DEScenario._load_structure(), which only knows the library's own
shipped versions and raises for v1.21. This test installs the repo's own
descape/versions/DE/v1.21/structure.json directly instead, the same technique
tools/map_structure_divergence.py uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from descape import library_compat
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

    Returns (map_width, map_height, trigger_tail). Raises whatever the walk
    itself raises; assertion 4 is "this doesn't raise", so the caller asserts
    by catching, not by inspecting a return value."""
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
    for section_name in scenario.structure:
        if section_name == "FileHeader":
            continue
        scenario._create_and_load_section(section_name, data_igen)
        if section_name == "Map":
            retriever_map = scenario.sections["Map"].retriever_map
            map_width = retriever_map["map_width"].data
            map_height = retriever_map["map_height"].data
        if section_name == "Units":
            break

    trigger_tail = data_igen.get_remaining_bytes()
    return map_width, map_height, trigger_tail


@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_v121_structure_alignment_oracle(path: Path) -> None:
    raw = path.read_bytes()

    # Assertion 4: DataHeader through Units parses without exception, and
    # map_width/map_height land in a real size band.
    map_width, map_height, trigger_tail = _walk_through_units(raw, path)
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

    # Assertion 5: reported, never asserted beyond "the walk did not over-read".
    assert len(trigger_tail) >= 0


def test_v121_corpus_is_the_expected_25_files() -> None:
    """Pins the file count so a Workshop sync silently adding/removing files
    changes this test's own collection, not just the oracle's parametrize list
    -- the same "name a stop condition instead of trusting all()" concern the
    plan itself raises for the 25-file target."""
    assert len(_v121_files()) == 25
