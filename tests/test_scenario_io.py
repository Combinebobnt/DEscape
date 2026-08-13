"""Migrated from tools/verify_roundtrip.py (Phase 3 of the pytest migration
plan -- Ordering step 5, first script in the migration order).

Verifies the scenario_io.py loader shim: every file loads, and the
captured trigger_tail is byte-complete (i.e. concatenating everything
parsed before it with the tail reconstructs the full decompressed body).
This is the v1 (read-only) safety net -- see tests/test_write_path.py
(once migrated) for v2's write path, kept separate since it exercises
different, newer code with a different failure mode from this one's pure-
parsing concern.

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
    for section_name in scenario.structure.keys():
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
