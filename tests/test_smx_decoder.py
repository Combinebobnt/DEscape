"""descape/smx_decoder.py against SMX files written by a small encoder here.
No mod art is committed (it is third-party); the real-asset oracle is the
corpus test at the bottom, which needs the developer's own enabled mods."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from descape import smx_decoder
from descape.sld_decoder import LayerKind, SLDError
from descape.smx_decoder import SMXError, SMXFile, load_smx

SKIP, DRAW, PLAYER = "skip", "draw", "player"


def _pixels(stream: list[tuple[int, int]], eight_to_five: bool) -> bytes:
    out = bytearray()
    per = 2 if eight_to_five else 4
    for start in range(0, len(stream), per):
        chunk = stream[start : start + per] + [(0, 0)] * (per - len(stream[start : start + per]))
        if eight_to_five:
            (i0, s0), (i1, s1) = chunk
            bits = i0 | (s0 << 8) | (i1 << 10) | (s1 << 18)
            out += bits.to_bytes(5, "little")
        else:
            out += bytes(i for i, _s in chunk)
            out.append(sum(s << (2 * n) for n, (_i, s) in enumerate(chunk)))
    return bytes(out)


def _frame(rows, width: int, palette: int = 28, eight_to_five: bool = False, hotspot=(3, 4)) -> bytes:
    """rows: None for an empty row, else (left, [(kind, count_or_pixels)...]).
    A draw/player run is a list of (index, section)."""
    edges = bytearray()
    commands = bytearray()
    stream: list[tuple[int, int]] = []
    for row in rows:
        if row is None:
            edges += struct.pack("<HH", 0xFFFF, 0xFFFF)
            continue
        left, runs = row
        used = left
        for kind, value in runs:
            count = value if kind == SKIP else len(value)
            code = {SKIP: 0, DRAW: 1, PLAYER: 2}[kind]
            commands.append(((count - 1) << 2) | code)
            if kind != SKIP:
                stream += value
            used += count
        commands.append(3)
        edges += struct.pack("<HH", left, width - used)
    pixel_bytes = _pixels(stream, eight_to_five)
    layer = bytes(edges) + struct.pack("<II", len(commands), len(pixel_bytes)) + bytes(commands) + pixel_bytes
    frame_type = 0x1 | (0x8 if eight_to_five else 0)
    head = struct.pack("<BBI", frame_type, palette, 0)
    layer_head = struct.pack("<HHhhII", width, len(rows), hotspot[0], hotspot[1], len(layer), 0)
    return head + layer_head + layer


def _file(*frames: bytes) -> bytes:
    body = b"".join(frames)
    return struct.pack("<4sHHII", b"SMPX", 2, len(frames), len(body), len(body)) + bytes(16) + body


def _palettes() -> dict[int, np.ndarray]:
    big = np.zeros((1024, 4), dtype=np.uint8)
    big[:, 0] = np.arange(1024) % 256
    big[:, 1] = np.arange(1024) // 256
    big[:, 3] = 255
    small = big[:256].copy()
    return {28: big, 0: small}


def _rgb(frame, y, x):
    return tuple(int(v) for v in frame.main[y, x])


@pytest.mark.parametrize("eight_to_five", [False, True])
def test_every_command_and_both_compressions(eight_to_five):
    rows = [
        (1, [(DRAW, [(10, 0), (11, 1), (12, 2)]), (SKIP, 1), (DRAW, [(13, 3)])]),
        None,
        (0, [(PLAYER, [(5, 0), (6, 0)]), (DRAW, [(14, 1)])]),
    ]
    smx = SMXFile(_file(_frame(rows, width=8, eight_to_five=eight_to_five)))
    assert smx.frame_count == 1
    frame = smx.decode_frame(0, None, _palettes(), shades=np.arange(256, dtype=np.uint8))
    assert (frame.width, frame.height, frame.hotspot_x, frame.hotspot_y) == (8, 3, 3, 4)
    assert _rgb(frame, 0, 1) == (10, 0, 0, 255)
    assert _rgb(frame, 0, 2) == (11, 1, 0, 255)
    assert _rgb(frame, 0, 3) == (12, 2, 0, 255)
    assert _rgb(frame, 0, 4)[3] == 0  # skip
    assert _rgb(frame, 0, 5) == (13, 3, 0, 255)
    assert not frame.main[1].any()  # empty row
    assert _rgb(frame, 2, 0) == (5, 5, 5, 255) and _rgb(frame, 2, 1) == (6, 6, 6, 255)
    assert frame.playercolor[2, 0].tolist() == [255, 0, 0, 255]
    assert frame.playercolor[2, 2].tolist() == [0, 0, 0, 0]
    assert _rgb(frame, 2, 2) == (14, 1, 0, 255)


def test_multi_frame_walk_and_kinds():
    rows = [(0, [(DRAW, [(1, 0)])])]
    data = _file(_frame(rows, 2), _frame(rows, 3, hotspot=(-1, 0)))
    smx = SMXFile(data, keep_data=False)
    assert smx.frame_count == 2 and smx.frames[1].hotspot_x == -1
    with pytest.raises(SMXError):
        smx.decode_frame(0, None, _palettes())
    frame = smx.decode_frame(1, data, _palettes(), kinds=(LayerKind.PLAYERCOLOR,))
    assert frame.main is None and frame.width == 3


def test_truncated_file_is_refused(tmp_path):
    data = _file(_frame([(0, [(DRAW, [(1, 0)])])], 2))
    with pytest.raises(SMXError):
        SMXFile(data[:-3])
    path = tmp_path / "x_x1.smx"
    path.write_bytes(data[:40])
    assert load_smx(path) is None
    assert issubclass(SMXError, SLDError)


def test_section_past_a_256_entry_palette_is_refused():
    smx = SMXFile(_file(_frame([(0, [(DRAW, [(1, 1)])])], 2, palette=0)))
    with pytest.raises(SMXError, match="256-entry"):
        smx.decode_frame(0, None, _palettes())


def _patched(data: bytes, offset: int, value: bytes) -> bytes:
    return data[:offset] + value + data[offset + len(value):]


def test_malformed_streams_are_refused():
    rows = [(0, [(DRAW, [(1, 0), (2, 0)])])]
    data = _file(_frame(rows, 2))
    commands_at = 32 + 6 + 16 + 4 + 8  # file, frame and layer headers, one row edge, two lengths
    palettes = _palettes()
    no_end = SMXFile(_patched(data, commands_at + 1, bytes([0])))  # the row end becomes a skip
    with pytest.raises(SMXError, match="row ends"):
        no_end.decode_frame(0, None, palettes)
    overrun = SMXFile(_patched(data, commands_at, bytes([(2 << 2) | 1])))  # draw 3 in a 2-wide row
    with pytest.raises(SMXError, match="overrun"):
        overrun.decode_frame(0, None, palettes)
    short = SMXFile(_patched(data, commands_at - 4, struct.pack("<I", 0)))  # pixel array length 0
    with pytest.raises(SMXError, match="shorter"):
        short.decode_frame(0, None, palettes)


def test_player_shades_scale_the_white_palette():
    shades = smx_decoder.player_shades([(200, 200, 200), (10, 20, 30)])
    assert shades[0] == 100 and shades[1] == 10 and shades[2] == smx_decoder.DEFAULT_PLAYER_SHADES[2]


# -- real-asset oracle -------------------------------------------------------

_INSTALL = os.environ.get("AOE2DE_INSTALL_PATH")
_REQUIRE = os.environ.get("DESCAPE_REQUIRE_MODS") == "1"
# Read at import: conftest's autouse fixture drops AOE2DE_PROFILE_PATH before each test.
_PROFILE = os.environ.get("AOE2DE_PROFILE_PATH")
# Mods whose art is the install's original plus a range ring, so pixels opaque
# in both should match the install .sld up to its BC1 compression.
_ORACLE_MODS = ("Open TC with Range", "Tower range mod", "Castle Foundation Range Mod")
# Measured 2026-10-15 (median of per-frame median channel error over pixels
# opaque in both, frames sampled 6 per file): Open TC 7.0-19.3, towers (8to5)
# 9.3-17.0, castle foundation 6.7. A wrong section order or 8to5 layout
# measured 33-46; thresholds sit between with margin.
_MAX_FILE_ERROR = 25.0
_MAX_MEDIAN_ERROR = 15.0
# Player pixels (smx_decoder.player_shades()), per frame median error against
# the install MAIN under them: measured median 14.0 (a flat grey 48.0, doubled
# shades 38.0). Overlap is the share of them the install's PLAYERCOLOR also
# covers past 128: measured median 0.81.
_MAX_PLAYER_ERROR = 25.0
_MIN_PLAYER_OVERLAP = 0.6


def _oracle_pairs():
    from descape import content_roots

    if not _INSTALL:
        return []
    profile = content_roots.profile_path(Path(_INSTALL))
    if profile is None:
        return []
    graphics = Path(_INSTALL) / "resources" / "_common" / "drs" / "graphics"
    pairs = []
    for root in content_roots.content_roots(install=None, profile=profile):
        if not root.mod_id or not any(name in root.mod_id for name in _ORACLE_MODS):
            continue
        folder = content_roots.child_dir(root.common, "drs", "graphics")
        found = sorted(folder.glob("*_x1.smx")) if folder else []
        pairs.extend((path, graphics / f"{path.stem}.sld") for path in found if (graphics / f"{path.stem}.sld").is_file())
    return pairs


@pytest.mark.corpus
def test_mod_smx_matches_the_install_sld_it_was_drawn_from(monkeypatch):
    from descape import mod_graphics
    from descape.sld_decoder import load_sld

    if _PROFILE:
        monkeypatch.setenv("AOE2DE_PROFILE_PATH", _PROFILE)
    pairs = _oracle_pairs()
    if not pairs:
        if _REQUIRE:
            pytest.fail("DESCAPE_REQUIRE_MODS=1 but no range mod with install .sld twins was found")
        pytest.skip("needs AOE2DE_INSTALL_PATH and an enabled range mod")
    palettes = mod_graphics.palettes()
    shades = mod_graphics.player_shades()
    file_errors, player_errors, overlaps = [], [], []
    for mod_path, install_path in pairs:
        smx, sld = load_smx(mod_path), load_sld(install_path)
        if smx is None or sld is None or smx.frame_count != sld.frame_count:
            continue
        errors = []
        for i in range(0, smx.frame_count, max(1, smx.frame_count // 6)):
            a = smx.decode_frame(i, None, palettes, shades=shades)
            b = sld.decode_frame(i, kinds=(LayerKind.MAIN, LayerKind.PLAYERCOLOR), window_to_main=True)
            if a.main is None or b.main is None:
                continue
            ys, xs = np.nonzero(a.main[..., 3] > 0)
            ty, tx = ys + b.hotspot_y - a.hotspot_y, xs + b.hotspot_x - a.hotspot_x
            inside = (ty >= 0) & (ty < b.height) & (tx >= 0) & (tx < b.width)
            ys, xs, ty, tx = ys[inside], xs[inside], ty[inside], tx[inside]
            both = b.main[ty, tx, 3] > 0
            player = np.zeros_like(both) if a.playercolor is None else a.playercolor[ys, xs, 3] > 0
            diff = np.abs(a.main[ys, xs, :3].astype(int) - b.main[ty, tx, :3].astype(int)).mean(axis=1)
            if (both & ~player).sum() >= 50:
                errors.append(float(np.median(diff[both & ~player])))
            if (both & player).sum() >= 50 and b.playercolor is not None:
                player_errors.append(float(np.median(diff[both & player])))
                overlaps.append(float((b.playercolor[ty, tx, 0][both & player] > 128).mean()))
        if errors:
            file_errors.append((float(np.median(errors)), mod_path.name))
    assert file_errors, "no comparable frames"
    worst = max(file_errors)
    assert worst[0] <= _MAX_FILE_ERROR, worst
    assert float(np.median([e for e, _n in file_errors])) <= _MAX_MEDIAN_ERROR
    assert player_errors, "no comparable player-colour frames"
    assert float(np.median(player_errors)) <= _MAX_PLAYER_ERROR
    assert float(np.median(overlaps)) >= _MIN_PLAYER_OVERLAP


# Mean horizontal neighbour difference over 4plus1 frames on 1024-entry
# palettes, measured 2026-10-15 over the enabled mods' 660 such frames: 22.5
# with pixel 0's section in the low bits, 34.2 reversed (wrong-section stripes).
_MAX_NEIGHBOUR_DIFF = 28.0


@pytest.mark.corpus
def test_4plus1_sections_read_smooth(monkeypatch):
    from descape import content_roots, mod_graphics

    if _PROFILE:
        monkeypatch.setenv("AOE2DE_PROFILE_PATH", _PROFILE)
    profile = content_roots.profile_path(Path(_INSTALL)) if _INSTALL else None
    roots = content_roots.content_roots(install=None, profile=profile) if profile else []
    palettes = mod_graphics.palettes()
    diffs = []
    for root in roots:
        folder = content_roots.child_dir(root.common, "drs", "graphics") if root.mod_id else None
        for path in sorted(folder.glob("*_x1.smx")) if folder else []:
            smx = load_smx(path)
            for i in range(0, smx.frame_count if smx else 0, max(1, (smx.frame_count if smx else 1) // 4)):
                info = smx.frames[i]
                pal = palettes.get(info.palette)
                if info.compressed_8to5 or info.main_offset is None or pal is None or len(pal) != 1024:
                    continue
                main = smx.decode_frame(i, None, palettes).main
                both = (main[:, :-1, 3] > 0) & (main[:, 1:, 3] > 0)
                if both.sum() >= 50:
                    left, right = main[:, :-1][both, :3].astype(int), main[:, 1:][both, :3].astype(int)
                    diffs.append(float(np.abs(left - right).mean()))
    if not diffs:
        if _REQUIRE:
            pytest.fail("DESCAPE_REQUIRE_MODS=1 but no 4plus1 mod frame on a 1024-entry palette was found")
        pytest.skip("needs AOE2DE_INSTALL_PATH and enabled mods shipping 4plus1 .smx")
    assert float(np.mean(diffs)) <= _MAX_NEIGHBOUR_DIFF
