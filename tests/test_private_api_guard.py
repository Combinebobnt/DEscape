"""Guards the AoE2ScenarioParser==0.8.3 pin (see requirements.txt's own
comment on why it's pinned, not just listed): descape/scenario_io.py and
descape/elevation_tools.py each reach into several of that library's
*private* names. Nothing else asserts these still exist -- an unpinned or
bumped install could silently drop or rename one of them, and the first
sign would be a runtime AttributeError/ImportError deep inside a real
scenario load, not a clear failure here.

A few of the names below (sections, retriever_map, byte_length, Retriever.data)
aren't underscore-prefixed, but are undocumented parse-time internals in the
same sense -- not exposed through MapManager/UnitManager's own public
accessors, and nothing in the library's docs commits to them. Pinned here for
the same reason as the underscore-prefixed ones: scenario_io.py's units-strip
support (units_block_offset etc., used by tools/strip_units.py) reaches into
them directly to derive byte offsets that AoE2FileSection.byte_length tracks
correctly but AoE2ScenarioParser's own get_data_as_bytes() does not (see
scenario_io.py's load_map_and_units() and tools/strip_units.py's docstrings).
"""

from __future__ import annotations

from pathlib import Path

from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
from AoE2ScenarioParser.scenarios.aoe2_scenario import (
    AoE2Scenario,
    _decompress_bytes,
    _get_file_version,
    _get_scenario_variant,
    _initialise_version_dependencies,
)
from AoE2ScenarioParser.sections.aoe2_file_section import AoE2FileSection
from AoE2ScenarioParser.sections.retrievers.retriever import Retriever

from descape.scenario_io import load_map_and_units


def test_scenario_io_private_functions_importable() -> None:
    """descape/scenario_io.py imports these four module-level private
    functions directly (see its own import block)."""
    for fn in (_decompress_bytes, _get_file_version, _get_scenario_variant, _initialise_version_dependencies):
        assert callable(fn)


def test_aoe2de_scenario_has_load_structure_and_load_header_section() -> None:
    """descape/scenario_io.py calls scenario._load_structure() and
    scenario._load_header_section(igen) directly on an AoE2DEScenario
    instance."""
    assert callable(getattr(AoE2DEScenario, "_load_structure", None))
    assert callable(getattr(AoE2DEScenario, "_load_header_section", None))


def test_aoe2scenario_base_has_create_and_load_section() -> None:
    """descape/scenario_io.py calls scenario._create_and_load_section(name,
    igen) -- defined on the base AoE2Scenario class the DE subclass
    inherits from, not overridden on AoE2DEScenario itself."""
    assert callable(getattr(AoE2Scenario, "_create_and_load_section", None))


def test_map_manager_has_elevation_tile_recursion() -> None:
    """descape/elevation_tools.py calls MapManager._elevation_tile_recursion()
    directly to reproduce the library's own neighbor-propagation rule."""
    assert callable(getattr(MapManager, "_elevation_tile_recursion", None))


def test_scenario_units_section_exposes_sections_retriever_map_and_byte_length() -> None:
    """descape/scenario_io.py reads scenario.sections["Units"].retriever_map
    ["players_units"].data, a list of AoE2FileSection whose .byte_length and
    nested retriever_map["unit_count"].data drive units_block_offset -- exact
    parse-time accounting, unlike get_data_as_bytes()'s re-serialization (see
    tools/strip_units.py's docstring). Exercised against a real load rather
    than just checked for existence on the class, since these are plain
    instance attributes set in __init__, not class-level members."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
    s = load_map_and_units(fixture)
    assert isinstance(s._scenario, AoE2Scenario)

    units_section = s._scenario.sections["Units"]
    assert isinstance(units_section, AoE2FileSection)
    assert isinstance(units_section.byte_length, int)

    players_units = units_section.retriever_map["players_units"].data
    assert isinstance(players_units, list) and players_units
    for section in players_units:
        assert isinstance(section, AoE2FileSection)
        assert isinstance(section.byte_length, int)
        unit_count_retriever = section.retriever_map["unit_count"]
        assert isinstance(unit_count_retriever, Retriever)
        assert isinstance(unit_count_retriever.data, int)
