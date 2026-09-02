"""Decodes .sld files (AoE2:DE's block-compressed in-world sprite format, in
resources/_common/drs/graphics/*.sld) to RGBA numpy arrays.

Phase 3's P3-g0. Leaf module: numpy and stdlib only, no Qt, no genieutils, no
install-path logic -- resolving a unit_const to a file is P3-g1/g3's job, and
nothing here reads config. As with slp_decoder.py, no game assets are shipped
in this repo; every byte comes from the user's own install at runtime.

No third-party SLD library exists for Python. The format understanding here
comes from SFTtech's openage (GPL-3.0+, license-compatible): its doc
media/sld-files.md and its reference decoder sld.pyx, which this
reimplements from scratch in numpy. Verified against the real install's 7,372
.sld files, not just read off the documentation.

Layout, as actually laid out on disk: a file header whose size is the u16
layout tag at offset 10 (16 bytes, or 14 for the variant below), then per frame
a 12-byte frame header immediately followed by *that frame's own* layer data --
NOT a contiguous table of frame headers. Each present layer starts with a u32
length counting its own 4 bytes; the next layer or frame header sits at
start_offset + length, padded up as an ABSOLUTE offset to the next value
congruent to the header size mod 4, not a pad of the layer's own byte count.

Three things worth knowing before reading further:

- **Skip commands are not always transparent.** With flag0 & 0x80 set and
  frame_index > 0, a skip copies the corresponding block from the previous
  layer of the same kind -- an inter-frame delta, used by 23% of layers
  including the idle graphics this tool wants. Chains run long: the deepest
  measured is 313 consecutive delta layers, so decode_frame() on one frame can
  pay for hundreds of predecessor decodes. Resolution is iterative, never
  recursive. There is deliberately no cache here; an LRU over decoded frames
  belongs with the rendering integration (P3-g3/g4), not in the decoder.
- **BC1 endpoints expand by *8/*4/*8, not by 5-to-8-bit replication.** That is
  what openage does and what this matches, so pure white decodes as
  (248, 252, 248). Not a defect; noted so nobody loses an hour diffing this
  against a general-purpose DXT1 tool.
- **The `14` layout variant differs from `16` by the header size and nothing
  else.** The u16 at header offset 10 is 16 for 7,145 of the install's files
  and 14 for the other 226 (stables, several mills and universities, some
  flags and huntables). Variant 14's header is variant 16's with the two
  always-zero high bytes of the trailing u32 truncated, so the tag is literally
  the header's byte length -- and because layer boundaries pad to the header
  size mod 4 rather than to a plain multiple of 4, every boundary in a
  variant-14 file sits at offset % 4 == 2. That single 2-byte phase shift is
  the whole of the "desync further in" this module used to reject the variant
  for; frame headers, sub-headers, the command stream and the BC1/BC4 payloads
  are byte-for-byte identical. Measured over the full install: all 226
  variant-14 files walk to exact EOF across 270,044 layers with the
  length == header + commands + 8 * drawn_blocks identity holding on every
  one. openage hardcodes 16 and still cannot read these.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path

import numpy as np

from descape import debug_log

# magic, version, frame_count, unknown1, layout_tag. Everything after this is
# variant-specific trailing bytes nothing here reads, and layout_tag is their
# end: the tag IS the byte length of the whole header.
_HEADER_PREFIX = struct.Struct("<4s4H")
_FRAME_HEADER = struct.Struct("<4H2BH")
_LAYER_LENGTH = struct.Struct("<I")
_GRAPHICS_SUBHEADER = struct.Struct("<4H2B")
_MASK_SUBHEADER = struct.Struct("<2B")
_COMMAND_COUNT = struct.Struct("<H")

MAGIC = b"SLDX"

# The header layouts this decoder reads, which are also their header sizes in
# bytes. See the module docstring. Both walk with one rule; nothing else in the
# file differs between them.
SUPPORTED_LAYOUT_TAGS = (14, 16)

# Set in a layer's flag0 when its skip commands copy from the previous
# same-kind layer instead of writing transparent blocks.
DELTA_FLAG = 0x80

BLOCK_PX = 4
BLOCK_BYTES = 8
PIXELS_PER_BLOCK = BLOCK_PX * BLOCK_PX


class SLDError(ValueError):
    """A .sld file this decoder will not read: bad magic, the unsupported 14
    layout variant, truncation, or a walk that runs off the end."""


class LayerKind(IntFlag):
    """frame_type bit flags. The bit order is also the on-disk layer order."""

    MAIN = 0x01
    SHADOW = 0x02
    OUTLINE = 0x04
    DAMAGE = 0x08
    PLAYERCOLOR = 0x10


# In frame_type bit order, which is the order layers appear in the file.
_LAYER_ORDER = (
    LayerKind.MAIN,
    LayerKind.SHADOW,
    LayerKind.OUTLINE,
    LayerKind.DAMAGE,
    LayerKind.PLAYERCOLOR,
)

# MAIN and DAMAGE are real DXT1; SHADOW and PLAYERCOLOR are single-channel BC4.
_BC1_KINDS = (LayerKind.MAIN, LayerKind.DAMAGE)


@dataclass(frozen=True)
class SLDLayer:
    """One layer of one frame, located but not decoded.

    x1/y1 and width/height are the layer's own bounding box within the frame
    canvas. DAMAGE and PLAYERCOLOR carry no box of their own and reuse MAIN's,
    already resolved here. All four are multiples of 4 in every one of the
    install's 365,970 surveyed layers, so blocks tile the box exactly -- but
    the box itself can still extend past the canvas edge (165 surveyed
    layers do), which is why compositing clips.
    """

    kind: LayerKind
    x1: int
    y1: int
    width: int
    height: int
    flag0: int
    flag1: int
    frame_index: int
    command_count: int
    command_offset: int
    block_offset: int
    chain_pos: int

    @property
    def is_delta(self) -> bool:
        return bool(self.flag0 & DELTA_FLAG) and self.frame_index > 0

    @property
    def block_count(self) -> int:
        return (self.width // BLOCK_PX) * (self.height // BLOCK_PX)


@dataclass(frozen=True)
class SLDFrame:
    """One frame's canvas geometry plus its located layers.

    frame_index is the file's own recorded index for the frame, not its
    position in `SLDFile.frames` -- the delta rule keys off the recorded one.
    """

    width: int
    height: int
    hotspot_x: int
    hotspot_y: int
    frame_type: int
    frame_index: int
    layers: tuple[SLDLayer, ...]


@dataclass(frozen=True)
class DecodedFrame:
    """Every layer of one frame, composited into canvas space.

    Each array is (height, width, 4) uint8 RGBA covering the whole canvas, with
    the layer pasted at its own bounding box and transparent elsewhere -- so all
    four share one coordinate system, and one hotspot anchors the lot. None
    means the frame has no layer of that kind. OUTLINE is never decoded.
    """

    width: int
    height: int
    hotspot_x: int
    hotspot_y: int
    main: np.ndarray | None = None
    shadow: np.ndarray | None = None
    damage: np.ndarray | None = None
    playercolor: np.ndarray | None = None


class SLDFile:
    """A walked .sld file. Construction locates every frame and layer; pixels
    are only decoded by decode_frame()."""

    def __init__(self, data: bytes):
        self.data = data
        self.frames: list[SLDFrame] = []
        # Per kind, every layer of that kind in walk order. A delta layer's
        # predecessor is the previous entry here, which is NOT frames[i - 1]'s
        # layer: 960 frames of the surveyed sample carry SHADOW but no MAIN, so
        # a MAIN chain can step over several frames.
        self._chains: dict[LayerKind, list[SLDLayer]] = {k: [] for k in _LAYER_ORDER}
        self._walk()

    @classmethod
    def from_file(cls, path: str | Path) -> "SLDFile":
        return cls(Path(path).read_bytes())

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    # -- walk ------------------------------------------------------------

    def _walk(self) -> None:
        data = self.data
        try:
            magic, _version, frame_count, _unknown1, layout_tag = _HEADER_PREFIX.unpack_from(data, 0)
        except struct.error as exc:
            raise SLDError(f"too short to hold an SLD header ({len(data)} bytes)") from exc
        if magic != MAGIC:
            raise SLDError(f"not an SLD file: magic {magic!r}")
        if layout_tag not in SUPPORTED_LAYOUT_TAGS:
            readable = " or ".join(str(t) for t in SUPPORTED_LAYOUT_TAGS)
            raise SLDError(f"unsupported layout variant {layout_tag} (only {readable} is readable)")
        if len(data) < layout_tag:
            raise SLDError(f"too short to hold an SLD header ({len(data)} bytes)")
        self.layout_tag = layout_tag

        offset = layout_tag
        for position in range(frame_count):
            offset = self._walk_frame(offset, position)

    def _walk_frame(self, offset: int, position: int) -> int:
        try:
            width, height, hotspot_x, hotspot_y, frame_type, _unknown5, frame_index = _FRAME_HEADER.unpack_from(
                self.data, offset
            )
        except struct.error as exc:
            raise SLDError(f"frame {position} header runs past the end of the file") from exc
        offset += _FRAME_HEADER.size

        layers: list[SLDLayer] = []
        main_box: tuple[int, int, int, int] | None = None
        for kind in _LAYER_ORDER:
            if not frame_type & kind:
                continue
            offset, layer, main_box = self._walk_layer(offset, kind, frame_index, main_box, position)
            if layer is not None:
                layers.append(layer)

        self.frames.append(
            SLDFrame(
                width=width,
                height=height,
                hotspot_x=hotspot_x,
                hotspot_y=hotspot_y,
                frame_type=frame_type,
                frame_index=frame_index,
                layers=tuple(layers),
            )
        )
        return offset

    def _walk_layer(
        self,
        offset: int,
        kind: LayerKind,
        frame_index: int,
        main_box: tuple[int, int, int, int] | None,
        position: int,
    ) -> tuple[int, SLDLayer | None, tuple[int, int, int, int] | None]:
        data = self.data
        start = offset
        try:
            (length,) = _LAYER_LENGTH.unpack_from(data, offset)
        except struct.error as exc:
            raise SLDError(f"frame {position} {kind.name} layer length runs past the end of the file") from exc
        offset += _LAYER_LENGTH.size
        if length < _LAYER_LENGTH.size or start + length > len(data):
            raise SLDError(f"frame {position} {kind.name} layer length {length} is out of range")

        if kind is LayerKind.OUTLINE:
            # Read nothing but the length. openage has a bare TODO here and no
            # decode; it does still read a command count off these bytes, which
            # is garbage it happens to discard. Parsing them would only manufacture
            # a bounds failure on a perfectly good file.
            return self._advance(start, length), None, main_box

        try:
            if kind in (LayerKind.MAIN, LayerKind.SHADOW):
                x1, y1, x2, y2, flag0, flag1 = _GRAPHICS_SUBHEADER.unpack_from(data, offset)
                offset += _GRAPHICS_SUBHEADER.size
                box = (x1, y1, x2 - x1, y2 - y1)
                if kind is LayerKind.MAIN:
                    main_box = box
            else:
                flag0, flag1 = _MASK_SUBHEADER.unpack_from(data, offset)
                offset += _MASK_SUBHEADER.size
                # DAMAGE/PLAYERCOLOR reuse MAIN's box. A frame carrying one
                # without a MAIN layer has no geometry at all; openage leaves it
                # zero-sized rather than guessing, and so does this.
                box = main_box if main_box is not None else (0, 0, 0, 0)

            (command_count,) = _COMMAND_COUNT.unpack_from(data, offset)
        except struct.error as exc:
            raise SLDError(f"frame {position} {kind.name} layer header runs past the end of the file") from exc
        offset += _COMMAND_COUNT.size

        x1, y1, width, height = box
        if width < 0 or height < 0:
            raise SLDError(f"frame {position} {kind.name} layer has a negative bounding box {box}")

        command_offset = offset
        block_offset = command_offset + 2 * command_count
        if block_offset > len(data):
            raise SLDError(f"frame {position} {kind.name} command array runs past the end of the file")

        chain = self._chains[kind]
        layer = SLDLayer(
            kind=kind,
            x1=x1,
            y1=y1,
            width=width,
            height=height,
            flag0=flag0,
            flag1=flag1,
            frame_index=frame_index,
            command_count=command_count,
            command_offset=command_offset,
            block_offset=block_offset,
            chain_pos=len(chain),
        )
        chain.append(layer)
        return self._advance(start, length), layer, main_box

    def _advance(self, start: int, length: int) -> int:
        """Next layer or frame header: start + length, then the absolute offset
        padded up to the next one congruent to the header size mod 4. Padding
        the layer's own byte count instead silently desyncs the rest of the
        file, and so does padding to a plain multiple of 4 in variant 14.

        Not a variant-16 regression: 16 % 4 == 0, so this is algebraically the
        old `(BLOCK_PX - offset) % BLOCK_PX` for every tag-16 file.
        """
        offset = start + length
        return offset + (self.layout_tag - offset) % BLOCK_PX

    # -- decode ----------------------------------------------------------

    def decode_frame(self, index: int) -> DecodedFrame:
        """Decodes every layer of one frame into canvas-space RGBA.

        Raises SLDError if the frame's command stream overruns its own block
        grid or its block data runs past the end of the file -- callers on a
        rendering path should catch that the same way they catch load_sld()
        returning None.
        """
        frame = self.frames[index]
        arrays: dict[LayerKind, np.ndarray] = {}
        for layer in frame.layers:
            image = self._decode_layer_image(layer)
            if image is not None:
                arrays[layer.kind] = _paste_on_canvas(image, layer.x1, layer.y1, frame.width, frame.height)
        return DecodedFrame(
            width=frame.width,
            height=frame.height,
            hotspot_x=frame.hotspot_x,
            hotspot_y=frame.hotspot_y,
            main=arrays.get(LayerKind.MAIN),
            shadow=arrays.get(LayerKind.SHADOW),
            damage=arrays.get(LayerKind.DAMAGE),
            playercolor=arrays.get(LayerKind.PLAYERCOLOR),
        )

    def _decode_layer_image(self, layer: SLDLayer) -> np.ndarray | None:
        if layer.width <= 0 or layer.height <= 0:
            return None
        blocks = self._decode_layer_blocks(layer)
        return _blocks_to_image(blocks, layer.width // BLOCK_PX, layer.height // BLOCK_PX)

    def _decode_layer_blocks(self, layer: SLDLayer) -> np.ndarray:
        """Resolves the layer's delta chain iteratively and returns its blocks.

        Walks back to the nearest same-kind layer that isn't a delta, then
        decodes forward. Recursing instead would sit near Python's frame limit
        on the deepest real chain (313 layers in a stock unit's idle graphic).
        """
        chain = self._chains[layer.kind]
        start = layer.chain_pos
        while start > 0 and chain[start].is_delta:
            start -= 1

        # At `start` there is either no predecessor at all or one the delta rule
        # doesn't reach back to, so the first decode always stands alone.
        blocks = self._decode_blocks(chain[start], None, None)
        for position in range(start + 1, layer.chain_pos + 1):
            blocks = self._decode_blocks(chain[position], blocks, chain[position - 1])
        return blocks

    def _decode_blocks(
        self,
        layer: SLDLayer,
        previous_blocks: np.ndarray | None,
        previous_layer: SLDLayer | None,
    ) -> np.ndarray:
        """One layer's blocks as (block_count, 16, 4) uint8 RGBA."""
        out = np.zeros((layer.block_count, PIXELS_PER_BLOCK, 4), dtype=np.uint8)
        skip_runs, draw_runs, filled = _command_runs(self.data, layer)
        if filled > layer.block_count:
            raise SLDError(
                f"{layer.kind.name} command stream fills {filled} blocks, past this layer's {layer.block_count}"
            )
        # Under-filling is normal, not corruption: 137 surveyed layers stop
        # early and leave the tail transparent, which the zeros above already are.

        draw_dest = _run_indices(draw_runs)
        if draw_dest.size:
            end = layer.block_offset + BLOCK_BYTES * draw_dest.size
            if end > len(self.data):
                raise SLDError(f"{layer.kind.name} block data runs past the end of the file")
            raw = np.frombuffer(
                self.data, dtype=np.uint8, count=BLOCK_BYTES * draw_dest.size, offset=layer.block_offset
            ).reshape(-1, BLOCK_BYTES)
            decode = decode_bc1_blocks if layer.kind in _BC1_KINDS else decode_bc4_blocks
            out[draw_dest] = decode(raw)

        if layer.is_delta and previous_blocks is not None and previous_layer is not None:
            skip_dest = _run_indices(skip_runs)
            if skip_dest.size:
                source = _previous_block_indices(skip_dest, layer, previous_layer)
                inside = source >= 0
                out[skip_dest[inside]] = previous_blocks[source[inside]]
        return out


def load_sld(path: str | Path) -> SLDFile | None:
    """Walks a .sld file, or returns None if it can't be read at all.

    Never raises: an unreadable, missing or unparseable file must be exactly as
    safe for a caller as having no AoE2:DE install configured, which is what the
    colored-mark fallback already handles.
    """
    try:
        return SLDFile.from_file(path)
    except (SLDError, OSError) as exc:
        debug_log.log(f"sld: {Path(path).name} not readable ({exc})")
        return None


# -- block decoding ------------------------------------------------------


def decode_bc1_blocks(raw: np.ndarray) -> np.ndarray:
    """(n, 8) uint8 BC1 blocks -> (n, 16, 4) uint8 RGBA, row-major within a block.

    Real DXT1: c0 > c1 gives four opaque colors, c0 <= c1 gives three plus a
    fully transparent one. Endpoints expand by *8/*4/*8 (see the module
    docstring), and c2/c3 interpolate on the expanded values, matching openage.
    """
    n = raw.shape[0]
    low0 = raw[:, 0].astype(np.uint16)
    high0 = raw[:, 1].astype(np.uint16)
    low1 = raw[:, 2].astype(np.uint16)
    high1 = raw[:, 3].astype(np.uint16)
    c0_val = low0 | (high0 << 8)
    c1_val = low1 | (high1 << 8)

    c0 = np.empty((n, 3), dtype=np.int32)
    c0[:, 0] = ((raw[:, 1] & 0xF8) >> 3).astype(np.int32) * 8
    c0[:, 1] = (((raw[:, 1] & 0x07).astype(np.int32) << 3) + ((raw[:, 0] & 0xE0) >> 5)) * 4
    c0[:, 2] = (raw[:, 0] & 0x1F).astype(np.int32) * 8

    c1 = np.empty((n, 3), dtype=np.int32)
    c1[:, 0] = ((raw[:, 3] & 0xF8) >> 3).astype(np.int32) * 8
    c1[:, 1] = (((raw[:, 3] & 0x07).astype(np.int32) << 3) + ((raw[:, 2] & 0xE0) >> 5)) * 4
    c1[:, 2] = (raw[:, 2] & 0x1F).astype(np.int32) * 8

    opaque = (c0_val > c1_val)[:, None]
    c2 = np.where(opaque, (2 * c0 + c1 + 1) // 3, (c0 + c1) // 2)
    c3 = np.where(opaque, (c0 + 2 * c1 + 1) // 3, 0)

    palette = np.zeros((n, 4, 4), dtype=np.uint8)
    palette[:, 0, :3] = c0
    palette[:, 1, :3] = c1
    palette[:, 2, :3] = c2
    palette[:, 3, :3] = c3
    palette[:, 0:3, 3] = 255
    palette[:, 3, 3] = np.where(opaque[:, 0], 255, 0)

    # Four index bytes, one per pixel row, two bits per pixel, low bits first.
    shifts = np.arange(4, dtype=np.uint8) * 2
    indices = ((raw[:, 4:8, None] >> shifts) & 0b11).reshape(n, PIXELS_PER_BLOCK)
    return palette[np.arange(n)[:, None], indices]


def decode_bc4_blocks(raw: np.ndarray) -> np.ndarray:
    """(n, 8) uint8 BC4 blocks -> (n, 16, 4) uint8 RGBA, single channel in R.

    Two modes, and getting them backwards yields a plausible-looking gradient
    rather than an obvious failure: a0 > a1 interpolates all six intermediate
    values, a0 <= a1 interpolates four and reserves index 6 for fully
    transparent and index 7 for 255.
    """
    n = raw.shape[0]
    a0 = raw[:, 0].astype(np.int32)
    a1 = raw[:, 1].astype(np.int32)
    wide = raw[:, 0] > raw[:, 1]

    values = np.zeros((n, 8), dtype=np.int32)
    values[:, 0] = a0
    values[:, 1] = a1
    # Indices 2..6: six interpolants over sevenths in the wide mode, four over
    # fifths in the narrow one, whose index 6 is instead the transparent entry.
    for step in range(1, 6):
        seven = ((7 - step) * a0 + step * a1) // 7
        five = ((5 - step) * a0 + step * a1) // 5 if step < 5 else np.zeros(n, dtype=np.int32)
        values[:, step + 1] = np.where(wide, seven, five)
    values[:, 7] = np.where(wide, (a0 + 6 * a1) // 7, 255)

    alpha = np.full((n, 8), 255, dtype=np.uint8)
    alpha[:, 6] = np.where(wide, 255, 0)

    palette = np.zeros((n, 8, 4), dtype=np.uint8)
    palette[:, :, 0] = values
    palette[:, :, 3] = alpha

    # Two 3-byte groups of eight 3-bit indices each, low bits first.
    shifts = np.arange(8, dtype=np.uint32) * 3
    group0 = raw[:, 2].astype(np.uint32) | (raw[:, 3].astype(np.uint32) << 8) | (raw[:, 4].astype(np.uint32) << 16)
    group1 = raw[:, 5].astype(np.uint32) | (raw[:, 6].astype(np.uint32) << 8) | (raw[:, 7].astype(np.uint32) << 16)
    indices = np.concatenate(
        [(group0[:, None] >> shifts) & 0b111, (group1[:, None] >> shifts) & 0b111], axis=1
    )
    return palette[np.arange(n)[:, None], indices]


# -- helpers -------------------------------------------------------------


def _command_runs(data: bytes, layer: SLDLayer) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Walks the (skip, draw) command pairs into destination-block runs.

    Plain Python because commands are few (a few hundred per layer) while the
    blocks they address are many; the runs then feed one vectorized decode.
    """
    skip_runs: list[tuple[int, int]] = []
    draw_runs: list[tuple[int, int]] = []
    position = 0
    offset = layer.command_offset
    for _ in range(layer.command_count):
        skip = data[offset]
        draw = data[offset + 1]
        offset += 2
        if skip:
            skip_runs.append((position, skip))
            position += skip
        if draw:
            draw_runs.append((position, draw))
            position += draw
    return skip_runs, draw_runs, position


def _run_indices(runs: list[tuple[int, int]]) -> np.ndarray:
    if not runs:
        return np.empty(0, dtype=np.intp)
    return np.concatenate([np.arange(start, start + count) for start, count in runs]).astype(np.intp)


def _previous_block_indices(dest: np.ndarray, layer: SLDLayer, previous: SLDLayer) -> np.ndarray:
    """Maps this layer's block indices onto the previous same-kind layer's.

    The two layers sit at different offsets in the frame, and only their own
    boxes are stored, so a copied block moves by the difference between them.
    Blocks that fall outside the previous layer come back as -1. Every surveyed
    box offset is a multiple of 4, so the block-space difference is exact.
    """
    width_blocks = layer.width // BLOCK_PX
    previous_width_blocks = previous.width // BLOCK_PX
    previous_height_blocks = previous.height // BLOCK_PX
    if width_blocks == 0 or previous_width_blocks == 0 or previous_height_blocks == 0:
        return np.full(dest.shape, -1, dtype=np.intp)

    x = dest % width_blocks + (layer.x1 - previous.x1) // BLOCK_PX
    y = dest // width_blocks + (layer.y1 - previous.y1) // BLOCK_PX
    inside = (x >= 0) & (x < previous_width_blocks) & (y >= 0) & (y < previous_height_blocks)
    return np.where(inside, x + y * previous_width_blocks, -1).astype(np.intp)


def _blocks_to_image(blocks: np.ndarray, width_blocks: int, height_blocks: int) -> np.ndarray:
    """(n, 16, 4) blocks in row-major block order -> (h, w, 4) image."""
    grid = blocks.reshape(height_blocks, width_blocks, BLOCK_PX, BLOCK_PX, 4)
    return grid.transpose(0, 2, 1, 3, 4).reshape(height_blocks * BLOCK_PX, width_blocks * BLOCK_PX, 4)


def _paste_on_canvas(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Places a layer image at (x, y) on a transparent canvas, clipped.

    Clipping is load-bearing, not defensive: 165 surveyed layers have a box
    extending past their own frame's canvas edge.
    """
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    source_x, source_y = max(0, -x), max(0, -y)
    dest_x, dest_y = max(0, x), max(0, y)
    copy_w = min(image.shape[1] - source_x, width - dest_x)
    copy_h = min(image.shape[0] - source_y, height - dest_y)
    if copy_w > 0 and copy_h > 0:
        canvas[dest_y : dest_y + copy_h, dest_x : dest_x + copy_w] = image[
            source_y : source_y + copy_h, source_x : source_x + copy_w
        ]
    return canvas
