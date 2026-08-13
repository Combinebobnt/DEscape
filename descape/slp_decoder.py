"""Decodes .slp files (the classic Genie-engine sprite format, still used for
UI icons in AoE2:DE's resources/_common/drs/interface/*.slp) to Pillow images.

No third-party SLP library was usable here: the one public Python
implementation found (fredreichbier/genie) is Python-2-only under the hood
(xrange, print-statements, an ancient `construct` API) and not a drop-in
dependency. Its documentation of the format is accurate, though, and this is a
compact, modern reimplementation of the same documented opcode stream --
verified against real AoE2:DE files (version tag "2.0N"), not just AoK-era
".slp"s the reference implementation targeted.

Palettes are the separate, plain-text JASC-PAL files AoE2:DE ships in
resources/_common/palettes/*.pal -- see load_palette().
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

HEADER_SIZE = 32
FRAME_ENTRY_SIZE = 32
TRANSPARENT = (0, 0, 0, 0)


def load_palette(path: str | Path) -> list[tuple[int, int, int]]:
    """Parses a JASC-PAL file into a list of (r, g, b) tuples."""
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].strip() != "JASC-PAL":
        raise ValueError(f"{path} is not a JASC-PAL file")
    count = int(lines[2].strip())
    colors = []
    for line in lines[3 : 3 + count]:
        parts = line.split()
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        colors.append((r, g, b))
    return colors


@dataclass
class _FrameHeader:
    cmd_table_offset: int
    outline_table_offset: int
    palette_offset: int
    properties: int
    width: int
    height: int
    hotspot_x: int
    hotspot_y: int


class SLPFile:
    def __init__(self, data: bytes):
        self.data = data
        version = data[0:4]
        if version not in (b"2.0N", b"2.0#"):
            raise ValueError(f"Unrecognized SLP version tag: {version!r}")
        (num_frames,) = struct.unpack_from("<I", data, 4)
        self.frame_headers: list[_FrameHeader] = []
        off = HEADER_SIZE
        for _ in range(num_frames):
            fh = _FrameHeader(*struct.unpack_from("<IIIIiiii", data, off))
            self.frame_headers.append(fh)
            off += FRAME_ENTRY_SIZE

    @classmethod
    def from_file(cls, path: str | Path) -> "SLPFile":
        return cls(Path(path).read_bytes())

    @property
    def num_frames(self) -> int:
        return len(self.frame_headers)

    def decode_frame(
        self, index: int, palette: list[tuple[int, int, int]], player: int = 1
    ) -> Image.Image:
        fh = self.frame_headers[index]
        d = self.data
        w, h = fh.width, fh.height
        img = Image.new("RGBA", (w, h), TRANSPARENT)
        px = img.load()

        # Row boundary (left/right transparent-padding) table.
        left_boundaries: list[int | None] = []
        pos = fh.outline_table_offset
        for _y in range(h):
            left, right = struct.unpack_from("<HH", d, pos)
            pos += 4
            if left == 0x8000 or right == 0x8000:
                left_boundaries.append(None)  # fully transparent row
            else:
                left_boundaries.append(left)

        # Per-row command-stream start offsets.
        command_offsets = []
        pos = fh.cmd_table_offset
        for _y in range(h):
            (o,) = struct.unpack_from("<I", d, pos)
            command_offsets.append(o)
            pos += 4

        def palette_color(idx: int) -> tuple[int, int, int]:
            if 0 <= idx < len(palette):
                return palette[idx]
            return (255, 0, 255)  # visibly-wrong magenta, not silently black

        def player_color(relindex: int) -> tuple[int, int, int]:
            return palette_color(player * 16 + relindex)

        y = 0
        while y < h and left_boundaries[y] is None:
            y += 1
        if y >= h:
            return img  # every row transparent
        x = left_boundaries[y]
        pos = command_offsets[y]

        def get_byte() -> int:
            nonlocal pos
            b = d[pos]
            pos += 1
            return b

        def draw(amount: int, color) -> None:
            nonlocal x
            if color is not None:
                for i in range(amount):
                    if 0 <= x + i < w:
                        px[x + i, y] = color + (255,)
            x += amount

        while y < h:
            opcode = get_byte()
            twobit = opcode & 0b11
            fourbit = opcode & 0b1111

            if fourbit == 0x0F:
                y += 1
                if y < h:
                    while y < h and left_boundaries[y] is None:
                        y += 1
                    if y < h:
                        x = left_boundaries[y]
                        pos = command_offsets[y]
                continue
            elif fourbit == 0x06:
                amount = (opcode >> 4) or get_byte()
                for _ in range(amount):
                    draw(1, player_color(get_byte()))
            elif fourbit == 0x0E:
                extended = opcode >> 4
                if extended in (4, 6):
                    idx = 0 if extended == 4 else player * 16
                    draw(1, palette_color(idx))
                elif extended in (5, 7):
                    amount = get_byte()
                    idx = 0 if extended == 5 else player * 16
                    draw(amount, palette_color(idx))
                # extended in (0, 1, 2, 3): shadow/flip-only markers, no pixels
            elif fourbit == 0x07:
                amount = (opcode >> 4) or get_byte()
                draw(amount, palette_color(get_byte()))
            elif fourbit == 0x0A:
                amount = (opcode >> 4) or get_byte()
                draw(amount, player_color(get_byte()))
            elif fourbit == 0x0B:
                amount = (opcode >> 4) or get_byte()
                draw(amount, None)
            elif twobit == 0:
                amount = opcode >> 2
                for _ in range(amount):
                    draw(1, palette_color(get_byte()))
            elif twobit == 1:
                amount = opcode >> 2
                if amount == 0:
                    amount = get_byte()
                draw(amount, None)
            elif twobit == 2:
                amount = ((opcode & 0xF0) << 4) + get_byte()
                for _ in range(amount):
                    draw(1, palette_color(get_byte()))
            elif twobit == 3:
                amount = ((opcode & 0xF0) << 4) + get_byte()
                draw(amount, None)
            else:
                raise ValueError(f"Unhandled SLP opcode {opcode:#x} at frame {index}")

        return img
