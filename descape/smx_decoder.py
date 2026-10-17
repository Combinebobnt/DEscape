"""Decodes .smx files (the palette-indexed sprite format AoE2:DE shipped before
.sld, which cosmetic mods still ship in mods/*/resources/_common/drs/graphics)
to the same DecodedFrame sld_decoder produces, so unit_sprites can draw either.

numpy and stdlib only; no Qt, no install-path logic: the caller passes the
palettes (mod_graphics.palettes()). Like sld_decoder.py, no game or mod assets
are shipped; every byte comes from the user's own files at runtime.

The format understanding comes from SFTtech's openage (GPL-3.0+): its doc
media/smx-files.md and its reference decoder smx.pyx, reimplemented here in
numpy. Measured over the 432 .smx files (56,338 frames) the enabled mods on
the development machine ship, 2026-10-15:

- **frame_type bits are LSB-first.** The doc numbers them from the top. The
  real values are 0x1, 0x2, 0x3, 0x7 and 0xb: bit 0 MAIN, bit 1 SHADOW,
  bit 2 OUTLINE, bit 3 the MAIN layer's compression (0 4plus1, 1 8to5).
- **The pixel stream is one run of chunks per layer.** Every 8to5 frame's
  pixel array is exactly ceil(drawn / 2) chunks, so pairs straddle rows and
  commands; a 4plus1 array is ceil(drawn / 4) chunks or one more.
- Every non-empty row's left edge, command advances and right edge sum to
  the layer width, and an empty row (0xFFFF) consumes no end-of-row command.

Layout: a 32-byte file header (signature, version, frame count, two sizes, a
16-byte comment), then the frames back to back with no offset table, so a
reader has to walk. A frame is a 6-byte header (frame_type, palette number,
uncompressed size) and its present layers in bit order. A layer is a 16-byte
header (width, height, hotspot x/y, the byte length of the data that follows,
one unread u32) and that data. MAIN's data is a per-row edge table (u16 left,
u16 right, 0xFFFF for an empty row), the command and pixel array lengths, then
the two arrays. Commands are one byte: `cmd & 3` is skip, draw, playercolor
draw or end of row, and `(cmd >> 2) + 1` the pixel count. Pixels are packed in
5-byte chunks whatever row or command they belong to: 4plus1 holds four
palette indices and a byte of 2-bit palette sections, 8to5 two 20-bit pixels
(an 8-bit index, a 2-bit section, then the second pixel's, then damage bits
nothing here reads). A pixel's colour is `palette[section * 256 + index]`.
"""

from __future__ import annotations

import struct
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from descape import debug_log
from descape.sld_decoder import DecodedFrame, LayerKind, SLDError

MAGIC = b"SMPX"
HEADER_SIZE = 32
_HEADER = struct.Struct("<4sHH")
_FRAME_HEADER = struct.Struct("<BBI")
_LAYER_HEADER = struct.Struct("<HHhhII")
_ARRAY_LENGTHS = struct.Struct("<II")

FRAME_MAIN = 0x1
FRAME_SHADOW = 0x2
FRAME_OUTLINE = 0x4
FRAME_8TO5 = 0x8
_LAYER_BITS = (FRAME_MAIN, FRAME_SHADOW, FRAME_OUTLINE)

CMD_SKIP, CMD_DRAW, CMD_PLAYER, CMD_END = 0, 1, 2, 3
EMPTY_ROW = 0xFFFF
CHUNK_BYTES = 5
SECTION_SIZE = 256

# The shift of each 4plus1 pixel's 2-bit palette section inside its chunk's
# fifth byte, pixel 0 first: pixel 0 in the LOW bits. Measured over the
# mods' 660 4plus1 frames on 1024-entry palettes: mean horizontal neighbour
# difference 22.5 this way, 34.2 reversed (wrong-section stripes).
_SECTION_SHIFTS_4PLUS1 = np.array([0, 2, 4, 6], dtype=np.uint8)

# A playercolor-draw pixel's index is a shade in the game's playercolor_*.pal
# ramps. MAIN gets the near-grey the .sld MAIN layer carries under its
# PLAYERCOLOR mask (unit_sprites' docstring, "How player colour is applied"):
# PLAYER_SHADE_SCALE times playercolor_white.pal's mean channel at that index.
# Fitted on 2026-10-15 against the install .sld of the same graphic over the
# four range mods' 54k player pixels: k = 0.508, RMSE 21.3 against 21.4 for a
# per-index median (the noise floor) and 35.3 for one constant grey.
PLAYER_SHADE_SCALE = 0.5
# Used when the caller has no playercolor_white.pal: one mid grey.
DEFAULT_PLAYER_SHADES = np.full(256, 96, dtype=np.uint8)


def player_shades(white_palette: Sequence[Sequence[int]]) -> np.ndarray:
    """The 256 MAIN greys for playercolor-draw pixels, from playercolor_white.pal's rows."""
    shades = DEFAULT_PLAYER_SHADES.copy()
    for i, colour in enumerate(white_palette[:256]):
        shades[i] = round(PLAYER_SHADE_SCALE * sum(colour[:3]) / 3)
    return shades


class SMXError(SLDError):
    """A .smx file this decoder will not read. An SLDError, so a caller's
    existing `except SLDError` covers both formats."""


@dataclass(frozen=True, slots=True)
class SMXFrame:
    """One walked frame: its type, palette number, and MAIN's geometry and
    data offset (main_offset is None when the frame has no MAIN layer)."""

    frame_type: int
    palette: int
    width: int
    height: int
    hotspot_x: int
    hotspot_y: int
    main_offset: int | None
    main_length: int

    @property
    def compressed_8to5(self) -> bool:
        return bool(self.frame_type & FRAME_8TO5)


class SMXFile:
    """A walked .smx file: per-frame MAIN geometry, pixels decoded only by
    decode_frame(). keep_data and release_data() behave as SLDFile's do."""

    def __init__(self, data: bytes, *, keep_data: bool = True):
        self.data: bytes | None = data
        self.byte_length = len(data)
        self._frames: list[SMXFrame] = []
        self._walk(data)
        if not keep_data:
            self.data = None

    @classmethod
    def from_file(cls, path: str | Path, *, keep_data: bool = True) -> SMXFile:
        return cls(Path(path).read_bytes(), keep_data=keep_data)

    def release_data(self) -> bytes | None:
        data, self.data = self.data, None
        return data

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> Sequence[SMXFrame]:
        return tuple(self._frames)

    def _walk(self, data: bytes) -> None:
        if len(data) < HEADER_SIZE:
            raise SMXError(f"too short to hold an SMX header ({len(data)} bytes)")
        magic, _version, frame_count = _HEADER.unpack_from(data, 0)
        if magic != MAGIC:
            raise SMXError(f"not an SMX file: magic {magic!r}")
        size = len(data)
        offset = HEADER_SIZE
        for position in range(frame_count):
            if offset + _FRAME_HEADER.size > size:
                raise SMXError(f"frame {position} header runs past the end of the file")
            frame_type, palette, _uncompressed = _FRAME_HEADER.unpack_from(data, offset)
            offset += _FRAME_HEADER.size
            main = None
            for bit in _LAYER_BITS:
                if not frame_type & bit:
                    continue
                if offset + _LAYER_HEADER.size > size:
                    raise SMXError(f"frame {position} layer {bit:#x} header runs past the end of the file")
                width, height, hotspot_x, hotspot_y, length, _unknown = _LAYER_HEADER.unpack_from(data, offset)
                start = offset + _LAYER_HEADER.size
                if start + length > size:
                    raise SMXError(f"frame {position} layer {bit:#x} length {length} runs past the end of the file")
                if bit == FRAME_MAIN:
                    main = (width, height, hotspot_x, hotspot_y, start, length)
                offset = start + length
            if main is None:
                self._frames.append(SMXFrame(frame_type, palette, 0, 0, 0, 0, None, 0))
            else:
                width, height, hotspot_x, hotspot_y, start, length = main
                self._frames.append(SMXFrame(frame_type, palette, width, height, hotspot_x, hotspot_y, start, length))

    def decode_frame(
        self,
        index: int,
        data: bytes | None = None,
        palettes: dict[int, np.ndarray] | None = None,
        *,
        kinds: Collection[LayerKind] | None = None,
        window_to_main: bool = True,
        shades: np.ndarray | None = None,
    ) -> DecodedFrame:
        """MAIN and PLAYERCOLOR of one frame as canvas-space RGBA, the canvas
        being MAIN's own box (window_to_main is accepted for SLDFile's
        signature; an SMX layer has no larger canvas). `palettes` maps a
        palette number to an (n, 4) uint8 RGBA array, `shades` is
        player_shades()' table. PLAYERCOLOR follows the .sld model: R is
        coverage (255), alpha 255 where it applies.

        Raises SMXError on a frame whose palette is missing, whose commands
        overrun their row or arrays, or whose section exceeds its palette."""
        del window_to_main
        if data is None:
            data = self.data
            if data is None:
                raise SMXError("walked with keep_data=False and decoded without passing the file's bytes")
        elif len(data) != self.byte_length:
            raise SMXError(f"file is {len(data)} bytes but was walked at {self.byte_length}; it changed on disk")
        frame = self._frames[index]
        if frame.main_offset is None or frame.width == 0 or frame.height == 0:
            return DecodedFrame(frame.width, frame.height, frame.hotspot_x, frame.hotspot_y)
        want_main = kinds is None or LayerKind.MAIN in kinds
        want_pc = kinds is None or LayerKind.PLAYERCOLOR in kinds
        palette = (palettes or {}).get(frame.palette)
        if palette is None:
            raise SMXError(f"frame {index} uses palette {frame.palette}, which is not loaded")
        main, player = _decode_main(data, frame, palette, DEFAULT_PLAYER_SHADES if shades is None else shades)
        return DecodedFrame(
            width=frame.width,
            height=frame.height,
            hotspot_x=frame.hotspot_x,
            hotspot_y=frame.hotspot_y,
            main=main if want_main else None,
            playercolor=player if want_pc and player is not None else None,
        )


def _decode_main(
    data: bytes, frame: SMXFrame, palette: np.ndarray, shades: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    width, height = frame.width, frame.height
    start, length = frame.main_offset, frame.main_length
    edges_end = start + 4 * height
    if edges_end + _ARRAY_LENGTHS.size > start + length:
        raise SMXError("MAIN row edges run past the layer")
    edges = np.frombuffer(data, dtype="<u2", count=2 * height, offset=start).reshape(height, 2)
    command_length, pixel_length = _ARRAY_LENGTHS.unpack_from(data, edges_end)
    commands_at = edges_end + _ARRAY_LENGTHS.size
    pixels_at = commands_at + command_length
    if pixels_at + pixel_length > start + length:
        raise SMXError("MAIN command or pixel array runs past the layer")

    main = np.zeros((height, width, 4), dtype=np.uint8)
    rows = np.flatnonzero(edges[:, 0] != EMPTY_ROW)
    commands = np.frombuffer(data, dtype=np.uint8, count=command_length, offset=commands_at)
    kind = commands & 3
    count = (commands >> 2).astype(np.int64) + 1
    is_end = kind == CMD_END
    ends = np.flatnonzero(is_end)
    if ends.size != rows.size:
        raise SMXError(f"MAIN has {ends.size} row ends for {rows.size} non-empty rows")
    if rows.size == 0:
        return main, None
    if ends[-1] != command_length - 1:
        raise SMXError("MAIN commands run on past the last row end")

    # Each command's row and its x inside the row, without a per-command loop.
    advance = np.where(is_end, 0, count)
    before = np.cumsum(advance) - advance
    row_of = np.searchsorted(ends, np.arange(command_length))
    row_first = np.concatenate(([0], ends[:-1] + 1))
    x = edges[rows, 0].astype(np.int64)[row_of] + before - before[row_first][row_of]
    if np.any(x + advance > width):
        raise SMXError("MAIN commands overrun their row")

    drawn = (kind == CMD_DRAW) | (kind == CMD_PLAYER)
    runs = count[drawn]
    total = int(runs.sum())
    if total == 0:
        return main, None
    run_start = np.cumsum(runs) - runs
    within = np.arange(total) - np.repeat(run_start, runs)
    ys = rows[np.repeat(row_of[drawn], runs)]
    xs = np.repeat(x[drawn], runs) + within
    player = np.repeat(kind[drawn] == CMD_PLAYER, runs)

    pixels = np.frombuffer(data, dtype=np.uint8, count=pixel_length, offset=pixels_at)
    k = np.arange(total)
    if frame.compressed_8to5:
        chunk, sub = k // 2, k % 2
        if (int(chunk[-1]) + 1) * CHUNK_BYTES > pixel_length:
            raise SMXError("MAIN pixel array is shorter than its draw commands")
        base = chunk * CHUNK_BYTES
        first = sub == 0
        b1 = pixels[base + 1].astype(np.int64)
        b2 = pixels[base + 2].astype(np.int64)
        index = np.where(first, pixels[base], ((b1 >> 2) | (b2 << 6)) & 0xFF).astype(np.int64)
        section = np.where(first, b1 & 3, (b2 >> 2) & 3).astype(np.int64)
    else:
        chunk, sub = k // 4, k % 4
        if (int(chunk[-1]) + 1) * CHUNK_BYTES > pixel_length:
            raise SMXError("MAIN pixel array is shorter than its draw commands")
        base = chunk * CHUNK_BYTES
        index = pixels[base + sub].astype(np.int64)
        section = ((pixels[base + 4] >> _SECTION_SHIFTS_4PLUS1[sub]) & 3).astype(np.int64)
    entry = section * SECTION_SIZE + index
    colour_pixels = ~player
    if np.any(entry[colour_pixels] >= len(palette)):
        raise SMXError(f"MAIN indexes past its {len(palette)}-entry palette")

    main[ys[colour_pixels], xs[colour_pixels]] = palette[entry[colour_pixels]]
    if not player.any():
        return main, None
    pc = np.zeros((height, width, 4), dtype=np.uint8)
    shade = shades[index[player] & 0xFF]
    main[ys[player], xs[player], :3] = shade[:, None]
    main[ys[player], xs[player], 3] = 255
    pc[ys[player], xs[player]] = (255, 0, 0, 255)
    return main, pc


def load_smx(path: str | Path, *, keep_data: bool = True) -> SMXFile | None:
    """Walks a .smx file, or None if it can't be read. Never raises, like
    sld_decoder.load_sld()."""
    try:
        return SMXFile.from_file(path, keep_data=keep_data)
    except (SLDError, OSError) as exc:
        debug_log.log(f"smx: {Path(path).name} not readable ({exc})")
        return None


def palette_array(colours: Sequence[Sequence[int]]) -> np.ndarray:
    """(n, 4) uint8 RGBA from (r, g, b) or (r, g, b, a) rows; a missing alpha is 255."""
    out = np.full((len(colours), 4), 255, dtype=np.uint8)
    for i, colour in enumerate(colours):
        out[i, : len(colour)] = colour
    return out
