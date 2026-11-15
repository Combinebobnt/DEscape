"""descape/format_scan.py: the body, terrain-grid and trigger-version
landmarks, on committed fixtures plus negative controls."""

from __future__ import annotations

import json
import math
import random
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


def _noise(size: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(size)


def _two_streams(seed: int) -> tuple[bytes, bytes, int]:
    """Header, a short stream, then the body stream: (raw, body payload, body start)."""
    short = _deflate(_noise(1 << 17, seed))
    payload = _noise(1 << 17, seed + 1)
    return _header() + short + _deflate(payload), payload, len(_header()) + len(short)


def test_find_body_allows_trailing_bytes_after_the_stream() -> None:
    payload = _noise(1 << 17, 1)
    body = fs.find_body(_header() + _deflate(payload) + b"junk" * 10)
    assert isinstance(body, fs.Body)
    assert (body.data, body.trailing) == (payload, 40)


def test_find_body_refuses_two_qualifying_starts() -> None:
    first = _deflate(_noise(1 << 17, 2))
    raw = _header() + first + _deflate(_noise(1 << 17, 3)) + b"junk"
    assert isinstance(fs.find_body(raw, limit=len(_header()) + len(first) + 1), str)


def test_find_body_prefers_the_stream_that_runs_to_the_end() -> None:
    raw, payload, start = _two_streams(10)
    body = fs.find_body(raw, limit=start + 1)
    assert isinstance(body, fs.Body)
    assert (body.start, body.data, body.trailing) == (start, payload, 0)


def _rejoining_streams() -> tuple[bytes, bytes, int]:
    """The first seed where a start a few bytes before the body decodes junk,
    then rejoins the body stream and ends with it (about 3% of seeds). Searched
    rather than pinned, since it hangs on the zlib build's exact output."""
    for seed in range(400):
        raw, payload, start = _two_streams(seed)
        for early in range(start - 8, start):
            got = fs._inflate(raw, early, fs.BODY_CAP)
            if got is not None and got[1] == 0 and len(got[0]) > len(payload) and got[0].endswith(payload):
                return raw, payload, start
    pytest.fail("no seed in 0..399 gives a start that rejoins the body stream")


def test_find_body_ignores_a_start_that_falls_into_step_with_the_body() -> None:
    raw, payload, start = _rejoining_streams()
    body = fs.find_body(raw, limit=start + 1)
    assert isinstance(body, fs.Body)
    assert (body.start, body.data, body.trailing) == (start, payload, 0)


def test_find_body_refuses_a_stream_too_small_to_be_a_body() -> None:
    assert "no raw-deflate stream" in fs.find_body(_header() + _deflate(_noise(1000, 4)))


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
