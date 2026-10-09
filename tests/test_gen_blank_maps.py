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
