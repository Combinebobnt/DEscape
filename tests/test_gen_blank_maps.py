"""tools/gen_blank_maps.py's non-square branch (TASK-032 Phase 0): the
asymmetric markers that make a W x H in-game gate able to tell row-major by
width from column-major by height, and the raw-bytes verifier that stands in
for the real loader until non-square maps load.
"""

from __future__ import annotations

import argparse
import sys

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH, TERRAIN_STRUCT_SIZE, load_map_and_units
from descape.scenario_new import blank_body
from descape.scenario_write import _compress_bytes

import conftest


@pytest.fixture(scope="module")
def tool():
    return conftest.load_verify_module("gen_blank_maps")


@pytest.fixture(scope="module")
def donor():
    return load_map_and_units(BLANK_TEMPLATE_PATH)


def _xy(i: int, w: int) -> tuple[int, int]:
    return i % w, i // w


def test_tall_markers_are_the_plans_stripe_and_corner_block(tool) -> None:
    markers = tool.marker_tiles(120, 168)
    stripe = {_xy(i, 120) for i, t in markers.items() if t == tool.STRIPE_TERRAIN}
    block = {_xy(i, 120) for i, t in markers.items() if t == tool.BLOCK_TERRAIN}
    assert stripe == {(x, 10) for x in range(120)}
    assert block == {(x, y) for x in range(5, 16) for y in range(150, 161)}


def test_wide_markers_are_the_tall_ones_transposed(tool) -> None:
    tall = {_xy(i, 120): t for i, t in tool.marker_tiles(120, 168).items()}
    wide = {_xy(i, 168): t for i, t in tool.marker_tiles(168, 120).items()}
    assert wide == {(y, x): t for (x, y), t in tall.items()}


def test_markers_refuse_a_square_or_a_non_discriminating_rectangle(tool) -> None:
    with pytest.raises(ValueError):
        tool.marker_tiles(120, 120)
    with pytest.raises(ValueError):
        tool.marker_tiles(120, 130)  # block would start at y=112, inside x's range


def test_parse_size(tool) -> None:
    assert tool.parse_size("168") == (168, 168)
    assert tool.parse_size("120x168") == (120, 168)
    for bad in ("120x", "60x120", "a", "1x2x3"):
        with pytest.raises(argparse.ArgumentTypeError):
            tool.parse_size(bad)


def test_main_writes_both_orientations_and_they_verify(tool, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["gen_blank_maps", "--out", str(tmp_path), "--sizes", "120x168", "168x120"])
    tool.main()
    out = capsys.readouterr().out
    assert "verified 120x168 on raw bytes, 241 marker tiles" in out
    assert "verified 168x120 on raw bytes, 241 marker tiles" in out
    assert "Near the SOUTH tip" in out and "Near the NORTH tip" in out
    assert {p.name for p in tmp_path.iterdir()} == {"blank_120x168.aoe2scenario", "blank_168x120.aoe2scenario"}


def _write(tmp_path, donor, body: bytes):
    path = tmp_path / "m.aoe2scenario"
    path.write_bytes(donor.header_bytes + _compress_bytes(body))
    return path


def test_raw_verifier_rejects_a_stray_or_a_missing_marker(tool, donor, tmp_path) -> None:
    """Negative controls for _verify_raw(): it is the only check those files
    get, so it must fail on a tile off the marker set and on a marker the
    body lacks."""
    markers = tool.marker_tiles(120, 168)
    off = donor.terrain_block_offset
    good = tool.apply_markers(blank_body(donor, 120, 168), off, markers)
    tool._verify_raw(_write(tmp_path, donor, good), donor, 120, 168, markers)

    stray = bytearray(good)
    stray[off + TERRAIN_STRUCT_SIZE * 0] = 1
    with pytest.raises(tool.GenerationVerificationError):
        tool._verify_raw(_write(tmp_path, donor, bytes(stray)), donor, 120, 168, markers)

    byte1 = bytearray(good)
    first = min(markers)
    byte1[off + TERRAIN_STRUCT_SIZE * first + 1] = 1  # elevation on a marker tile
    with pytest.raises(tool.GenerationVerificationError):
        tool._verify_raw(_write(tmp_path, donor, bytes(byte1)), donor, 120, 168, markers)

    missing = tool.apply_markers(blank_body(donor, 120, 168), off, dict(list(markers.items())[1:]))
    with pytest.raises(tool.GenerationVerificationError):
        tool._verify_raw(_write(tmp_path, donor, missing), donor, 120, 168, markers)

    transposed = tool.apply_markers(blank_body(donor, 168, 120), off, markers)
    with pytest.raises(tool.GenerationVerificationError):
        tool._verify_raw(_write(tmp_path, donor, transposed), donor, 120, 168, markers)


# --- New Map's fill and planting (the in-game pairs for trees on a new map) ---


@pytest.mark.parametrize(
    ("flags", "expect"),
    [
        (["--terrain", "10", "--trees"], "verified 120x120 aoe2 terrain 10, 14400 GAIA units planted, seed 3"),
        (["--eye-candy"], "verified 120x120 aoe2 terrain 0, "),  # off-centre doodads: f32 positions
        (["--terrain", "89", "--trees", "--game-data", "ror"], "verified 120x120 ror terrain 89, "),
    ],
)
def test_main_fills_and_plants_a_square_map_and_verifies_it(tool, tmp_path, monkeypatch, capsys, flags, expect) -> None:
    argv = ["gen_blank_maps", "--out", str(tmp_path), "--sizes", "120", "--seed", "3", "--name", "m.aoe2scenario"]
    monkeypatch.setattr(sys, "argv", argv + flags)
    tool.main()
    out = capsys.readouterr().out
    assert expect in out
    assert " 0 GAIA units" not in out
    assert [p.name for p in tmp_path.iterdir()] == ["m.aoe2scenario"]


def test_the_planted_verifier_rejects_units_other_than_the_planned_ones(tool, tmp_path) -> None:
    import random

    from descape import terrain_units
    from descape.scenario_new import blank_scenario_bytes

    path = tmp_path / "f.aoe2scenario"
    path.write_bytes(blank_scenario_bytes(120, terrain_id=10, trees=True, rng=random.Random(1)))
    planned = terrain_units.plan_fill(120, 120, 10, trees=True, doodads=False, rng=random.Random(1))
    tool._verify(path, 120, terrain_id=10, planted=planned)
    for wrong in (planned[1:], terrain_units.plan_fill(120, 120, 10, trees=True, doodads=False, rng=random.Random(2))):
        with pytest.raises(tool.GenerationVerificationError):
            tool._verify(path, 120, terrain_id=10, planted=wrong)
    with pytest.raises(tool.GenerationVerificationError):
        tool._verify(path, 120, terrain_id=10, game_data="ror", planted=planned)


@pytest.mark.parametrize(
    "argv",
    [["--sizes", "120x168", "--trees"], ["--sizes", "120", "144", "--name", "x.aoe2scenario"]],
)
def test_main_refuses_planting_a_non_square_size_or_one_name_for_two_sizes(tool, tmp_path, monkeypatch, argv) -> None:
    monkeypatch.setattr(sys, "argv", ["gen_blank_maps", "--out", str(tmp_path), *argv])
    with pytest.raises(SystemExit):
        tool.main()
    assert list(tmp_path.iterdir()) == []
