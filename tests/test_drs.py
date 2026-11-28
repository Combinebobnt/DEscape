"""tools/_drs.py: DRS archives, layered lookup, SLP headers and JASC palettes.

The default tier packs small archives with literal `struct.pack` calls; the
corpus tier reads The Conquerors install's own archives.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The Conquerors install's Data folder for the corpus tier; DESCAPE_AOC_DATA overrides it.
AOC_DATA = Path(os.environ.get("DESCAPE_AOC_DATA") or Path.home() / "games" / "Age of Empires 2" / "Data")


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import _drs
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return _drs


drs = _load()


def pack_drs(path: Path, tables: dict[bytes, dict[int, bytes]], *, version: bytes = b"1.00") -> Path:
    """A DRS file: header, table entries, row tables, then the member bytes in order."""
    rows_at = 64 + 12 * len(tables)
    data_at = rows_at + sum(12 * len(m) for m in tables.values())
    head = struct.pack("<40s4s12sii", b"Copyright (c) 1997 Ensemble Studios.\x1a", version, b"tribe", len(tables), data_at)
    entries, rows, blobs = b"", b"", b""
    for tag, members in tables.items():
        entries += struct.pack("<4sii", tag, rows_at + len(rows), len(members))
        for member_id, blob in members.items():
            rows += struct.pack("<iii", member_id, data_at + len(blobs), len(blob))
            blobs += blob
    path.write_bytes(head + entries + rows + blobs)
    return path


def pack_slp(frames: list[tuple[int, int, int, int]], comment: bytes = b"ArtDesk 1.00 SLP Writer") -> bytes:
    out = struct.pack("<4si24s", b"2.0N", len(frames), comment)
    for k, (width, height, hx, hy) in enumerate(frames):
        out += struct.pack("<IIIIiiii", 100 + k, 50 + k, 0, 24, width, height, hx, hy)
    return out


def test_an_archive_lists_and_reads_members_by_type(tmp_path):
    slp = pack_slp([(47, 52, 29, 37)])
    path = pack_drs(tmp_path / "graphics.drs", {b" pls": {2: slp, 5: b"five"}, b"anib": {50500: b"pal"}})
    archive = drs.DrsArchive(path)
    assert archive.version == "1.00" and archive.file_type == "tribe"
    assert archive.copyright == "Copyright (c) 1997 Ensemble Studios."
    assert archive.ids("slp") == [2, 5] and archive.ids("bina") == [50500] and archive.ids("wav") == []
    assert archive.read("slp", 2) == slp and archive.read("bina", 50500) == b"pal"
    assert archive.members["slp"][5][1] == 4
    with pytest.raises(KeyError, match="slp 3"):
        archive.read("slp", 3)


def test_the_first_layer_holding_an_id_wins(tmp_path):
    override = drs.DrsArchive(pack_drs(tmp_path / "patch.drs", {b" pls": {5: b"new5"}}))
    base = drs.DrsArchive(pack_drs(tmp_path / "base.drs", {b" pls": {2: b"old2", 5: b"old5"}, b" vaw": {9: b"w"}}))
    layered = drs.LayeredDrs(override, base)
    assert layered.ids("slp") == [2, 5] and layered.ids("wav") == [9]
    assert layered.read("slp", 5) == b"new5" and layered.read("slp", 2) == b"old2"
    assert layered.source("slp", 5) is override and layered.source("slp", 7) is None
    assert drs.LayeredDrs(base, override).read("slp", 5) == b"old5"
    with pytest.raises(KeyError, match="slp 7"):
        layered.read("slp", 7)


def test_the_aoc_graphics_stack_the_engine_order_and_skip_absent_archives(tmp_path):
    from tools import _game_root

    data = tmp_path / "Data"
    data.mkdir()
    pack_drs(data / "gamedata_x1.drs", {b"anib": {54000: b"x1"}})
    with pytest.raises(FileNotFoundError, match=r"no graphics\.drs"):
        _game_root.classic_graphics(tmp_path, _game_root.AOC)
    (data / "gamedata_x1.drs").unlink()
    pack_drs(data / "graphics.drs", {b" pls": {5168: b"stock", 2: b"two"}})
    pack_drs(data / "gamedata_x1_p1.drs", {b" pls": {5168: b"patched"}})
    pack_drs(data / "terrain.drs", {b" pls": {15000: b"grass"}})
    layered = _game_root.classic_graphics(tmp_path, _game_root.AOC)
    assert [a.path.name for a in layered.archives] == ["gamedata_x1_p1.drs", "graphics.drs"]
    assert layered.read("slp", 5168) == b"patched" and not layered.has("slp", 15000)
    pack_drs(data / "gamedata.drs", {b"anib": {54000: b"stock"}})
    pack_drs(data / "gamedata_x1.drs", {b"anib": {54000: b"x1"}})
    layered = _game_root.classic_graphics(tmp_path, _game_root.AOC)
    names = ["gamedata_x1.drs", "gamedata_x1_p1.drs", "graphics.drs", "gamedata.drs"]
    assert [a.path.name for a in layered.archives] == names, "age2_x1.exe's measured order"
    assert layered.read("bina", 54000) == b"x1"
    with pytest.raises(FileNotFoundError):
        _game_root.classic_graphics(tmp_path / "missing", _game_root.AOC)


def test_a_bad_archive_is_refused(tmp_path):
    with pytest.raises(ValueError, match="version"):
        drs.DrsArchive(pack_drs(tmp_path / "v.drs", {b" pls": {}}, version=b"2.00"))
    path = pack_drs(tmp_path / "cut.drs", {b" pls": {1: b"abcdef"}})
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="outside the file"):
        drs.DrsArchive(path)
    (tmp_path / "short.drs").write_bytes(b"Copyright")
    with pytest.raises(ValueError, match="too short"):
        drs.DrsArchive(tmp_path / "short.drs")


def test_slp_header_reads_each_frame_row():
    header = drs.slp_header(pack_slp([(47, 52, 29, 37), (40, 54, -3, 37)]) + b"\0" * 8)
    assert header.version == "2.0N" and header.comment == "ArtDesk 1.00 SLP Writer"
    assert len(header.frames) == 2
    second = header.frames[1]
    assert (second.width, second.height, second.hotspot_x, second.hotspot_y) == (40, 54, -3, 37)
    assert (second.cmd_table_offset, second.outline_table_offset, second.palette_offset, second.properties) == (101, 51, 0, 24)
    with pytest.raises(ValueError, match="frame count"):
        drs.slp_header(struct.pack("<4si24s", b"2.0N", 3, b""))


def test_read_jasc_palette():
    text = b"JASC-PAL\r\n0100\r\n3\r\n0 0 0\r\n0 74 161\r\n255 255 255\r\n"
    assert drs.read_jasc_palette(text) == [(0, 0, 0), (0, 74, 161), (255, 255, 255)]
    with pytest.raises(ValueError, match="JASC-PAL"):
        drs.read_jasc_palette(b"RIFF....")
    with pytest.raises(ValueError, match="holds 1"):
        drs.read_jasc_palette(b"JASC-PAL\n0100\n2\n1 2 3\n")


@pytest.mark.corpus
def test_the_installed_conquerors_archives():
    if not (AOC_DATA / "graphics.drs").is_file():
        pytest.skip(f"no Conquerors install at {AOC_DATA}: set DESCAPE_AOC_DATA to its Data folder")
    graphics = drs.DrsArchive(AOC_DATA / "graphics.drs")
    ids = graphics.ids("slp")
    assert (len(ids), ids[0], ids[-1]) == (1768, 2, 5312)
    assert {drs.slp_header(graphics.read("slp", i)).version for i in ids[::50]} == {"2.0N"}
    terrain = drs.DrsArchive(AOC_DATA / "terrain.drs").ids("slp")
    assert (len(terrain), terrain[0], terrain[-1]) == (27, 15000, 15031)
    interfac = drs.DrsArchive(AOC_DATA / "interfac.drs")
    assert [len(interfac.ids(k)) for k in ("slp", "bina", "wav")] == [221, 126, 50]
    assert len(drs.read_jasc_palette(interfac.read("bina", 50500))) == 256
    from tools import _game_root

    layered = _game_root.classic_graphics(AOC_DATA.parent, _game_root.AOC)
    patched = [5168, 5176, 5182, 5218, 5219, 5220]
    assert drs.DrsArchive(AOC_DATA / "gamedata_x1_p1.drs").ids("slp") == patched
    for slp_id in patched:
        assert layered.source("slp", slp_id).path.name == "gamedata_x1_p1.drs"
        assert layered.read("slp", slp_id) != graphics.read("slp", slp_id)
        assert drs.slp_header(layered.read("slp", slp_id)).version == "2.0N"


def test_a_loose_slp_stack_serves_the_first_folder_holding_an_id(tmp_path):
    """HD Edition's loose <id>.slp folders (TASK-431): first folder wins, LayeredDrs's interface."""
    x2, x1, base = (tmp_path / name for name in ("gamedata_x2", "gamedata_x1", "graphics"))
    for folder, files in ((x2, {"8.slp": b"x2-8", "6000.slp": b"x2-6000"}), (x1, {"8.SLP": b"x1-8"}),
                          (base, {"8.slp": b"base-8", "305.slp": b"base-305", "notes.txt": b"", "a.slp": b""})):
        folder.mkdir()
        for name, data in files.items():
            (folder / name).write_bytes(data)
    stack = drs.LooseSlpStack(x2, x1, base, tmp_path / "absent")
    assert stack.ids("slp") == [8, 305, 6000] and stack.ids("bina") == []
    assert stack.read("slp", 8) == b"x2-8" and stack.read("slp", 305) == b"base-305"
    assert stack.has("slp", 6000) and not stack.has("slp", 7) and not stack.has("bina", 8)
    assert drs.LooseSlpStack(x1, base).read("slp", 8) == b"x1-8"
    with pytest.raises(KeyError, match="slp 7"):
        stack.read("slp", 7)
