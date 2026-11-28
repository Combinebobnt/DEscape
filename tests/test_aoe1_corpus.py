"""Every Age of Empires (1997) / Rise of Rome scenario in an install (TASK-515):
the loose scenario/*.scx files and every scenario embedded in campaign/*.cpn
and *.cpx load, and their map and units equal an independent hand-walk of
the 1.15 layout written out below, not read from the structure. Skips
without --aoe1-dir."""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from pathlib import Path

import pytest

from descape import scenario_format, scenario_io

pytestmark = pytest.mark.corpus

_SEPARATOR = struct.pack("<I", 0xFFFFFF9D)
_CAMPAIGN_ENTRY = struct.Struct("<ii255s257s")  # size, offset, name, file name
_UNIT = struct.Struct("<fffiHBf")  # x, y, z, reference_id, const, status, rotation: 23 bytes


def campaign_scenarios(raw: bytes) -> list[tuple[str, bytes]]:
    """(file name, bytes) of each scenario a .cpn/.cpx holds: char[4] version,
    char[256] name, i32 count, then one 520-byte entry per scenario."""
    (count,) = struct.unpack_from("<i", raw, 260)
    out = []
    for i in range(count):
        size, offset, _name, file_name = _CAMPAIGN_ENTRY.unpack_from(raw, 264 + i * _CAMPAIGN_ENTRY.size)
        out.append((file_name.split(b"\x00")[0].decode("latin-1"), raw[offset : offset + size]))
    return out


def _scenarios(root: Path) -> list[tuple[str, bytes]]:
    found = [(p.name, p.read_bytes()) for p in sorted((root / "scenario").glob("*.scx"))]
    for campaign in sorted(p for p in (root / "campaign").glob("*") if p.suffix.lower() in (".cpn", ".cpx")):
        found += [(f"{campaign.name}/{name}", data) for name, data in campaign_scenarios(campaign.read_bytes())]
    return found


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "aoe1_file" not in metafunc.fixturenames:
        return
    option = metafunc.config.getoption("--aoe1-dir")
    files = _scenarios(Path(option)) if option else []
    if not files:
        metafunc.parametrize("aoe1_file", [pytest.param(None, marks=pytest.mark.skip(reason="no --aoe1-dir"))])
        return
    metafunc.parametrize("aoe1_file", files, ids=[name for name, _data in files])


def _hand_walk(raw: bytes) -> tuple[int, int, Counter, list[list[tuple[int, float, float]]]]:
    """(width, height, terrain multiset, per-section (const, x, y)) off the
    inflated body. Map starts past the fourth 0xFFFFFF9D separator after the
    AI files (PlayerDataTwo, GlobalVictory, Diplomacy, then Options' fixed
    339 u32s), as measured on the corpus 2026-11-23."""
    body = zlib.decompress(raw[8 + struct.unpack_from("<I", raw, 4)[0] :], -zlib.MAX_WBITS)
    resources = body.index(_SEPARATOR) + 4
    victory = resources + 16 * 16
    assert body[victory : victory + 4] == _SEPARATOR
    diplomacy_end = victory + 44 + 16 * 16 * 4 + 11520
    assert body[diplomacy_end : diplomacy_end + 4] == _SEPARATOR
    map_start = diplomacy_end + 4 + 64 + 339 * 4
    assert body[map_start : map_start + 4] == _SEPARATOR
    width, height = struct.unpack_from("<ii", body, map_start + 4)
    o = map_start + 12
    terrain = Counter(body[o : o + width * height * 3 : 3])
    o += width * height * 3
    (sections,) = struct.unpack_from("<I", body, o)
    o += 4 + 8 * 16
    units = []
    for _ in range(sections):
        (count,) = struct.unpack_from("<I", body, o)
        o += 4
        rows = []
        for _ in range(count):
            x, y, _z, _ref, const, _status, _rotation = _UNIT.unpack_from(body, o)
            rows.append((const, x, y))
            o += _UNIT.size
        units.append(rows)
    # PlayerDataThree follows (u32 9, then 8 records whose tail varies by victory version);
    # the test's trigger_tail == b"" check is what pins that the body ends there.
    assert body[o : o + 4] == struct.pack("<I", 9)
    return width, height, terrain, units


def test_an_aoe1_scenario_loads_and_matches_the_hand_walk(aoe1_file) -> None:
    name, raw = aoe1_file
    fmt = scenario_format.detect_format(raw)
    assert (fmt.data_version, fmt.game) == ("1.15", scenario_format.AOE1)
    loaded = scenario_io.load_map_and_units_from_bytes(raw, name)
    assert loaded.terrain_block_offset >= 0 and loaded.units_block_offset >= 0, "both byte walks verified"
    assert loaded.trigger_tail == b""
    width, height, terrain, units = _hand_walk(raw)
    mm = loaded.map_manager
    assert (mm.map_width, mm.map_height) == (width, height)
    assert Counter(t.terrain_id for t in mm.terrain) == terrain
    got = [[(u.unit_const, u.x, u.y) for u in player] for player in loaded.unit_manager.units]
    assert got == units
    assert all(u.initial_animation_frame == 0 and u.garrisoned_in_id == -1 for p in loaded.unit_manager.units for u in p)


def test_an_aoe1_scenario_shows_its_five_messages(aoe1_file) -> None:
    """No string-table ids in 1.15: the texts walk from the section start (Somin1 stores all five)."""
    from descape.messages_fields import present_fields, stored_text
    from descape.messages_model import MessagesEditModel

    name, raw = aoe1_file
    loaded = scenario_io.load_map_and_units_from_bytes(raw, name)
    assert loaded.messages_write_supported
    fields = present_fields(loaded._scenario.sections["Messages"].retriever_map)
    assert [s.field_id for s in fields] == ["instructions", "hints", "victory", "loss", "history"]
    model = MessagesEditModel(loaded)
    start, end = model.section_span()
    assert model.serialize() == loaded.decompressed_body[start:end]
    if name == "Somin1.scx":
        assert all(stored_text(loaded, s) for s in fields), "all five texts stored"


def test_an_aoe1_scenario_refuses_the_library_walk_by_name(aoe1_file) -> None:
    name, raw = aoe1_file
    with pytest.raises(scenario_io.UnsupportedStructureVersion, match="fast unit reader"):
        scenario_io.load_map_and_units_from_bytes(raw, name, fast_units=False)
