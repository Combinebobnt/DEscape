"""descape/format_scan.py: the body, terrain-grid and trigger-version
landmarks, on committed fixtures plus negative controls."""

from __future__ import annotations

import json
import math
import os
import struct
import zlib

import pytest

from descape import format_scan as fs
from descape import iso_geometry, library_compat, scenario_new
from descape import structure_walk as sw

import conftest

FIXTURES = conftest.ROOT / "tests" / "fixtures"
TERRAIN_IDS = {int(k) for k in json.loads((conftest.ROOT / "descape" / "terrain_classes.json").read_text(encoding="utf-8"))["classes"]}
SAMPLES = ["v159_units_triggers.aoe2scenario", "real_blank_240x240.aoe2scenario", "units_120x120.aoe2scenario"]


def _deflate(data: bytes) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return packer.compress(data) + packer.flush()


def _structure(version: str) -> dict:
    return json.loads((library_compat.VERSIONS_DIR / f"v{version}" / "structure.json").read_text(encoding="utf-8"))


def test_constants_mirror_their_sources() -> None:
    assert fs.MAP_TILES_MAX == scenario_new.MAX_MAP_TILES
    assert fs.ELEVATION_MAX == iso_geometry.MAX_ELEVATION


@pytest.mark.parametrize("name", SAMPLES)
def test_landmarks_agree_with_the_walker(name: str) -> None:
    raw = (FIXTURES / name).read_bytes()
    walked = sw.walk_file(_structure(raw[:4].decode()), raw)
    body = fs.find_body(raw)
    assert isinstance(body, fs.Body)
    assert (body.start, body.trailing, body.data) == (walked.header.end, 0, walked.body_bytes)
    grid = fs.scan_terrain_grid(body.data, 7, TERRAIN_IDS)
    assert isinstance(grid, fs.GridMatch)
    assert grid.offset == walked.body.column("Map", "map_width").offsets[0]
    assert grid.end == walked.body.sections["Units"][0]


def _header() -> bytes:
    return b"1.59" + bytes(64)


def test_find_body_allows_trailing_bytes_after_the_stream() -> None:
    payload = os.urandom(1 << 17)
    body = fs.find_body(_header() + _deflate(payload) + b"junk" * 10)
    assert isinstance(body, fs.Body)
    assert (body.data, body.trailing) == (payload, 40)


def test_find_body_refuses_two_qualifying_starts() -> None:
    first = _deflate(os.urandom(1 << 17))
    raw = _header() + first + _deflate(os.urandom(1 << 17)) + b"junk"
    assert isinstance(fs.find_body(raw, limit=len(_header()) + len(first) + 1), str)


def test_find_body_prefers_the_stream_that_runs_to_the_end() -> None:
    short = _deflate(os.urandom(1 << 17))
    payload = os.urandom(1 << 17)
    raw = _header() + short + _deflate(payload)
    body = fs.find_body(raw, limit=len(_header()) + len(short) + 1)
    assert isinstance(body, fs.Body)
    assert (body.start, body.data, body.trailing) == (len(_header()) + len(short), payload, 0)


def test_find_body_refuses_a_stream_too_small_to_be_a_body() -> None:
    assert "no raw-deflate stream" in fs.find_body(_header() + _deflate(os.urandom(1000)))


def _grid_body(width: int, tiles: bytes, anchor: int = fs.UNITS_FIRST_VALUE) -> bytes:
    return b"\xff" * 13 + struct.pack("<II", width, width) + tiles + struct.pack("<I", anchor) + bytes(40)


def _tiles(width: int, terrain: int = 0, elevation: int = 1) -> bytearray:
    return bytearray(bytes([terrain, elevation, 0, 0, 0, 255, 255]) * (width * width))


def test_a_clean_synthetic_grid_matches() -> None:
    grid = fs.scan_terrain_grid(_grid_body(120, bytes(_tiles(120))), 7, TERRAIN_IDS)
    assert grid == fs.GridMatch(13, 120, 120, 7)


def test_a_corrupted_anchor_is_no_match() -> None:
    assert isinstance(fs.scan_terrain_grid(_grid_body(120, bytes(_tiles(120)), anchor=8), 7, TERRAIN_IDS), fs.GridMiss)


def test_two_grid_shaped_matches_are_refused() -> None:
    one = _grid_body(120, bytes(_tiles(120)))
    miss = fs.scan_terrain_grid(one + one, 7, TERRAIN_IDS)
    assert isinstance(miss, fs.GridMiss)
    assert "2 grid-shaped matches" in miss.reason


def test_an_unknown_terrain_id_fails_and_is_named() -> None:
    unknown = max(TERRAIN_IDS) + 1
    tiles = _tiles(120)
    tiles[7 * 500] = unknown
    miss = fs.scan_terrain_grid(_grid_body(120, bytes(tiles)), 7, TERRAIN_IDS)
    assert isinstance(miss, fs.GridMiss)
    assert miss.unknown_ids == (unknown,)


def test_an_elevation_over_the_bound_is_no_match() -> None:
    tiles = _tiles(120)
    tiles[7 * 9 + 1] = fs.ELEVATION_MAX + 1
    assert isinstance(fs.scan_terrain_grid(_grid_body(120, bytes(tiles)), 7, TERRAIN_IDS), fs.GridMiss)


@pytest.mark.parametrize(
    ("value", "expected"), [(5.0, True), (5.1, True), (7.0, True), (7.5, False), (5.05, False), (math.nan, False), (4.8, False)]
)
def test_plausible_trigger_version(value: float, expected: bool) -> None:
    assert fs.plausible_trigger_version(value, (4.9, 5.0)) is expected
