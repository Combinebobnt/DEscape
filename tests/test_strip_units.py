"""Verifies tools/strip_units.py: the byte-patch splice that removes every
unit from an .aoe2scenario file's Units section, originally written to
prepare descape/templates/blank_120x120.aoe2scenario and the real 240x240/
480x480 exports that have since moved to tests/fixtures/real_blank_*.
aoe2scenario as byte oracles for descape/scenario_new.py (see that module's
docstring) -- all three carried GRASS_GREEN ground-scatter doodads before
stripping, not a genuinely blank map (see the tool's own module docstring).

Two tiers, same split as tests/test_write_path.py:

- Default tier: the tool's own no-op self-test against the golden unit-free
  fixture (tests/fixtures/golden_blank_120x120.aoe2scenario), plus a check
  that every shipped template is now actually unit-free. Both fast and
  self-contained.

- Corpus tier (`@pytest.mark.corpus`, examples/ corpus, real units): the
  no-op self-test is necessary but not sufficient -- the golden fixture's
  players_units block is already 36 bytes of zeros, so replacing it with 36
  zeros is identity by construction regardless of whether the offset was
  computed correctly. It never exercises actual record removal. The corpus
  test strips a real file that has real units (into tmp_path -- the source
  corpus file is never touched) and checks the properties that actually
  discriminate a correct offset from a wrong one: the post-strip Units
  section size matches what the file's own (real, possibly non-golden-shaped)
  player_data_3 prefix implies, every count reads zero, the trigger tail is
  byte-identical, and terrain is unchanged tile-for-tile.
"""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

import pytest

import conftest
from descape.scenario_io import BLANK_TEMPLATE_SIZES, TEMPLATE_DIR, blank_template_path, load_map_and_units
from descape.scenario_write import _compress_bytes

strip_units_module = conftest.load_verify_module("strip_units")
strip_units = strip_units_module.strip_units
StripVerificationError = strip_units_module.StripVerificationError

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def test_golden_fixture_is_itself_unit_free() -> None:
    s = load_map_and_units(GOLDEN_PATH)
    assert sum(len(units) for units in s.unit_manager.units) == 0
    assert all(t.terrain_id == 0 and t.elevation == 0 for t in s.map_manager.terrain)


def test_strip_is_noop_on_already_unit_free_golden() -> None:
    s = load_map_and_units(GOLDEN_PATH)
    assert strip_units(s) == s.decompressed_body


def test_strip_refuses_when_units_block_verification_failed() -> None:
    s = load_map_and_units(GOLDEN_PATH)
    unsupported = dataclasses.replace(s, units_write_supported=False, units_block_offset=-1)
    with pytest.raises(StripVerificationError):
        strip_units(unsupported)


@pytest.mark.parametrize("tiles", BLANK_TEMPLATE_SIZES)
def test_shipped_template_is_unit_free(tiles: int) -> None:
    path = blank_template_path(tiles)
    assert path.parent == TEMPLATE_DIR
    s = load_map_and_units(path)

    assert s.map_manager.map_width == s.map_manager.map_height == tiles
    assert s.scenario_version == "1.58"
    assert s.terrain_write_supported
    assert all(t.terrain_id == 0 and t.elevation == 0 for t in s.map_manager.terrain)
    assert sum(len(units) for units in s.unit_manager.units) == 0
    (next_unit_id,) = struct.unpack_from("<I", s.decompressed_body, 0)
    assert next_unit_id == 0


REAL_ORACLE_PATHS = {
    240: Path(__file__).resolve().parent / "fixtures" / "real_blank_240x240.aoe2scenario",
    480: Path(__file__).resolve().parent / "fixtures" / "real_blank_480x480.aoe2scenario",
}


@pytest.mark.parametrize("tiles", sorted(REAL_ORACLE_PATHS))
def test_relocated_real_oracle_is_unit_free(tiles: int) -> None:
    """The same invariants test_shipped_template_is_unit_free() checks, run
    against the 240x240/480x480 real game exports now that BLANK_TEMPLATE_SIZES
    is just (120,) and no longer parametrizes over them -- otherwise this
    coverage would silently evaporate on exactly the two files being
    repurposed as descape/scenario_new.py's byte oracle (see
    tests/test_scenario_new.py)."""
    s = load_map_and_units(REAL_ORACLE_PATHS[tiles])

    assert s.map_manager.map_width == s.map_manager.map_height == tiles
    assert s.scenario_version == "1.58"
    assert s.terrain_write_supported
    assert all(t.terrain_id == 0 and t.elevation == 0 for t in s.map_manager.terrain)
    assert sum(len(units) for units in s.unit_manager.units) == 0
    (next_unit_id,) = struct.unpack_from("<I", s.decompressed_body, 0)
    assert next_unit_id == 0


@pytest.mark.corpus
def test_strip_removes_real_units_corpus(scenario_path, tmp_path: Path) -> None:
    original = load_map_and_units(scenario_path)
    if not original.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units block failed load-time verification")
    total_units = sum(len(units) for units in original.unit_manager.units)
    if total_units == 0:
        pytest.skip(f"{scenario_path.name}: already unit-free, doesn't exercise real record removal")

    players_units_bytes = original.units_section_end - original.units_block_offset
    units_section_byte_length = original._scenario.sections["Units"].byte_length
    prefix_bytes = units_section_byte_length - players_units_bytes
    expected_stripped_units_byte_length = prefix_bytes + 4 * original.number_of_unit_sections

    new_body = strip_units(original)
    out = tmp_path / scenario_path.name
    out.write_bytes(original.header_bytes + _compress_bytes(new_body))

    stripped = load_map_and_units(out)

    assert sum(len(units) for units in stripped.unit_manager.units) == 0
    assert stripped._scenario.sections["Units"].byte_length == expected_stripped_units_byte_length
    assert stripped.trigger_tail == original.trigger_tail
    if original.terrain_write_supported:
        orig_terrain = [(t.terrain_id, t.elevation, t.layer) for t in original.map_manager.terrain]
        new_terrain = [(t.terrain_id, t.elevation, t.layer) for t in stripped.map_manager.terrain]
        assert new_terrain == orig_terrain
    (next_unit_id,) = struct.unpack_from("<I", stripped.decompressed_body, 0)
    assert next_unit_id == 0
