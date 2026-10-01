"""Decodes .sld files (AoE2:DE's block-compressed in-world sprite format, in
resources/_common/drs/graphics/*.sld) to RGBA numpy arrays.

Phase 3's P3-g0. numpy and stdlib, plus composite_backend for the optional
native kernel; no Qt, no genieutils, no install-path logic -- resolving a
unit_const to a file is P3-g1/g3's job, and nothing here reads config. As with
slp_decoder.py, no game assets are shipped in this repo; every byte comes from
the user's own install at runtime.

The walk and the per-layer delta-chain decode run in the native kernel
(_composite_native.pyx's sld_walk/sld_decode_layer) when
composite_backend.native is set, read at call time like render.py does. The
numpy code below stays the oracle and the no-compiler path. The kernel returns
a failure sentinel wherever the Python would raise, and the Python then re-runs
to raise the exact SLDError; a re-run that succeeds instead is a kernel bug,
counted in `native_disagreements` and answered from the Python result.

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

import operator
import struct
from array import array
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import IntFlag
from itertools import pairwise
from pathlib import Path

import numpy as np

from descape import composite_backend, debug_log

# magic, version, frame_count, unknown1, layout_tag. Everything after this is
# variant-specific trailing bytes nothing here reads, and layout_tag is their
# end: the tag IS the byte length of the whole header.
_HEADER_PREFIX = struct.Struct("<4s4H")
_FRAME_HEADER = struct.Struct("<4H2BH")
_LAYER_LENGTH = struct.Struct("<I")

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

# A layer's length, sub-header and command count in one unpack: MAIN/SHADOW
# carry a box (x1, y1, x2, y2) and two flags, DAMAGE/PLAYERCOLOR only the flags.
_GRAPHICS_LAYER_HEAD = struct.Struct("<I4H2BH")
_MASK_LAYER_HEAD = struct.Struct("<I2BH")
_GRAPHICS, _MASK, _SKIPPED = 0, 1, 2


def _layer_shape(kind: LayerKind) -> int:
    if kind is LayerKind.OUTLINE:
        return _SKIPPED
    return _GRAPHICS if kind in (LayerKind.MAIN, LayerKind.SHADOW) else _MASK


# frame_type & 0x1F -> (kind, index in _LAYER_ORDER, shape) for each present
# layer, in file order. Plain ints: `frame_type & kind` builds an IntFlag per test.
_LAYER_PLAN = tuple(
    tuple((kind, position, _layer_shape(kind)) for position, kind in enumerate(_LAYER_ORDER) if bits & kind.value)
    for bits in range(32)
)
_KIND_POSITION = {kind: position for position, kind in enumerate(_LAYER_ORDER)}

# Record widths of SLDFile's flat u32 index; the field order is in its docstring.
_LAYER_FIELDS = 11
_FRAME_FIELDS = 8
_INDEX_TYPECODE = "I"

# Each frame records at most four layers: OUTLINE is stepped over, not indexed.
_MAX_LAYERS_PER_FRAME = 4

# Times the native kernel refused a walk or decode the Python then accepted.
# Always 0 unless the kernel has a bug; tests/test_sld_native.py pins it.
native_disagreements = 0


def _native_disagreed(what: str) -> None:
    global native_disagreements
    native_disagreements += 1
    debug_log.log(f"sld: native kernel refused a {what} the numpy path accepts; using numpy's result")


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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
    means the frame has no layer of that kind, or the decode was not asked for
    it. OUTLINE is never decoded. Under decode_frame(window_to_main=True) the
    window is the canvas: width, height, hotspot and every array refer to it.
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
    are only decoded by decode_frame().

    keep_data=False drops the bytes once the walk is done, leaving only the
    index. A caller holding many indexes then passes each decode the file's
    bytes itself, so one index can serve any number of decodes without holding
    the file in memory. release_data() does the same after the walk, handing
    the bytes to whoever caches them; nothing else mutates an index.

    The index is flat unsigned 32-bit records, not an object per layer: one
    object per layer held 92.6 MB for the 341 files one big map touches
    (2026-09-28). A layer record is kind (position in _LAYER_ORDER), x1, y1,
    width, height, flag0, flag1, frame_index, command_count, command_offset,
    chain_pos, with block_offset derived. A frame record is width, height,
    hotspot_x, hotspot_y, frame_type, frame_index, first layer id, layer
    count. `frames` reads it back as SLDFrame/SLDLayer values built on each
    access, so compare those by value, never by identity.
    """

    def __init__(self, data: bytes, *, keep_data: bool = True):
        self.data: bytes | None = data
        self.byte_length = len(data)
        self._frames = array(_INDEX_TYPECODE)
        self._layers = array(_INDEX_TYPECODE)
        # Per kind (by _LAYER_ORDER position), every layer id of that kind in
        # walk order. A delta layer's predecessor is the previous entry here,
        # which is NOT frames[i - 1]'s layer: 960 frames of the surveyed sample
        # carry SHADOW but no MAIN, so a MAIN chain can step over several frames.
        self._chains = tuple(array(_INDEX_TYPECODE) for _ in _LAYER_ORDER)
        self._walk()
        if not keep_data:
            self.data = None

    @classmethod
    def from_file(cls, path: str | Path, *, keep_data: bool = True) -> SLDFile:
        return cls(Path(path).read_bytes(), keep_data=keep_data)

    def release_data(self) -> bytes | None:
        """Drops the bytes this index kept and returns them: keep_data=False
        after the walk, for a caller that holds the bytes somewhere else."""
        data, self.data = self.data, None
        return data

    @property
    def frame_count(self) -> int:
        return len(self._frames) // _FRAME_FIELDS

    @property
    def frames(self) -> Sequence[SLDFrame]:
        return _FrameView(self)

    def _frame(self, index: int) -> SLDFrame:
        base = index * _FRAME_FIELDS
        width, height, hotspot_x, hotspot_y, frame_type, frame_index, first, count = self._frames[
            base : base + _FRAME_FIELDS
        ]
        layers = tuple(self._layer(layer_id) for layer_id in range(first, first + count))
        return SLDFrame(width, height, hotspot_x, hotspot_y, frame_type, frame_index, layers)

    def _layer(self, layer_id: int) -> SLDLayer:
        base = layer_id * _LAYER_FIELDS
        kind, x1, y1, width, height, flag0, flag1, frame_index, command_count, command_offset, chain_pos = self._layers[
            base : base + _LAYER_FIELDS
        ]
        block_offset = command_offset + 2 * command_count
        return SLDLayer(
            _LAYER_ORDER[kind], x1, y1, width, height, flag0, flag1, frame_index,
            command_count, command_offset, block_offset, chain_pos,
        )

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
        if len(data) > 0xFFFFFFFF:
            # Offsets are stored as u32; no real file comes anywhere near this.
            raise SLDError(f"too large to index ({len(data)} bytes)")
        self.layout_tag = layout_tag

        native = composite_backend.native
        if native is None:
            self._walk_frames(data, layout_tag, frame_count)
        elif not self._walk_native(native, data, layout_tag, frame_count):
            self._walk_frames(data, layout_tag, frame_count)
            _native_disagreed("walk")

    def _walk_native(self, native, data: bytes, layout_tag: int, frame_count: int) -> bool:
        """_walk_frames() in the kernel. False, leaving the index empty, where
        _walk_frames() would raise."""
        frames = np.empty(frame_count * _FRAME_FIELDS, dtype=np.uint32)
        layers = np.empty(frame_count * _MAX_LAYERS_PER_FRAME * _LAYER_FIELDS, dtype=np.uint32)
        chains = np.empty((len(_LAYER_ORDER), frame_count), dtype=np.uint32)
        chain_len = np.zeros(len(_LAYER_ORDER), dtype=np.int64)
        layer_count = native.sld_walk(data, layout_tag, frame_count, frames, layers, chains, chain_len)
        if layer_count < 0:
            return False
        self._frames.frombytes(frames.tobytes())
        self._layers.frombytes(layers[: layer_count * _LAYER_FIELDS].tobytes())
        for chain, row, count in zip(self._chains, chains, chain_len, strict=True):
            chain.frombytes(row[:count].tobytes())
        return True

    def _walk_frames(self, data: bytes, layout_tag: int, frame_count: int) -> None:
        # One flat loop, one unpack per layer, checks in the step-by-step order.
        # _layer_read_error() splits a failed unpack back into that order.
        size = len(data)
        add_frame = self._frames.extend
        add_layer = self._layers.extend
        chains = self._chains
        unpack_frame = _FRAME_HEADER.unpack_from
        unpack_graphics = _GRAPHICS_LAYER_HEAD.unpack_from
        unpack_mask = _MASK_LAYER_HEAD.unpack_from
        unpack_length = _LAYER_LENGTH.unpack_from
        layer_id = 0
        offset = layout_tag
        for position in range(frame_count):
            try:
                width, height, hotspot_x, hotspot_y, frame_type, _unknown5, frame_index = unpack_frame(data, offset)
            except struct.error as exc:
                raise SLDError(f"frame {position} header runs past the end of the file") from exc
            offset += _FRAME_HEADER.size

            first_layer = layer_id
            # DAMAGE/PLAYERCOLOR reuse MAIN's box. A frame carrying one without
            # a MAIN layer has no geometry at all; openage leaves it zero-sized.
            main_box = (0, 0, 0, 0)
            for kind, kind_position, shape in _LAYER_PLAN[frame_type & 0x1F]:
                start = offset
                if shape == _GRAPHICS:
                    try:
                        length, x1, y1, x2, y2, flag0, flag1, command_count = unpack_graphics(data, start)
                    except struct.error as exc:
                        raise _layer_read_error(data, start, kind, position) from exc
                    if length < _LAYER_LENGTH.size or start + length > size:
                        raise SLDError(f"frame {position} {kind.name} layer length {length} is out of range")
                    box = (x1, y1, x2 - x1, y2 - y1)
                    if box[2] < 0 or box[3] < 0:
                        raise SLDError(f"frame {position} {kind.name} layer has a negative bounding box {box}")
                    if kind is LayerKind.MAIN:
                        main_box = box
                    command_offset = start + _GRAPHICS_LAYER_HEAD.size
                elif shape == _MASK:
                    try:
                        length, flag0, flag1, command_count = unpack_mask(data, start)
                    except struct.error as exc:
                        raise _layer_read_error(data, start, kind, position) from exc
                    if length < _LAYER_LENGTH.size or start + length > size:
                        raise SLDError(f"frame {position} {kind.name} layer length {length} is out of range")
                    box = main_box
                    command_offset = start + _MASK_LAYER_HEAD.size
                else:
                    # OUTLINE: read nothing but the length. openage reads a
                    # command count off these bytes and discards the garbage.
                    try:
                        (length,) = unpack_length(data, start)
                    except struct.error as exc:
                        raise _layer_read_error(data, start, kind, position) from exc
                    if length < _LAYER_LENGTH.size or start + length > size:
                        raise SLDError(f"frame {position} {kind.name} layer length {length} is out of range")
                    offset = start + length
                    offset += (layout_tag - offset) % BLOCK_PX
                    continue

                block_offset = command_offset + 2 * command_count
                if block_offset > size:
                    raise SLDError(f"frame {position} {kind.name} command array runs past the end of the file")
                chain = chains[kind_position]
                x1, y1, layer_width, layer_height = box
                add_layer((
                    kind_position, x1, y1, layer_width, layer_height, flag0, flag1, frame_index,
                    command_count, command_offset, len(chain),
                ))
                chain.append(layer_id)
                layer_id += 1
                # Next layer or frame header: the absolute offset padded up to
                # the header size mod 4, never the layer's own byte count.
                offset = start + length
                offset += (layout_tag - offset) % BLOCK_PX

            add_frame((width, height, hotspot_x, hotspot_y, frame_type, frame_index, first_layer, layer_id - first_layer))

    # -- decode ----------------------------------------------------------

    def decode_frame(
        self,
        index: int,
        data: bytes | None = None,
        *,
        kinds: Collection[LayerKind] | None = None,
        window_to_main: bool = False,
    ) -> DecodedFrame:
        """Decodes every layer of one frame into canvas-space RGBA.

        `data` is the file's bytes, for an index walked with keep_data=False;
        None decodes from the bytes this object kept. `kinds` limits the decode
        to those layers, leaving the others' fields None; None decodes them all.

        window_to_main narrows the canvas to MAIN's box clipped to it: every
        array is sliced to that window, width/height are its size and the
        hotspot is rebased onto it. MAIN is transparent outside its box, so
        nothing of MAIN is lost. A frame without a drawable MAIN keeps the
        whole canvas; a box starting past the canvas edge gives zero-size arrays.

        Raises SLDError if the frame's command stream overruns its own block
        grid or its block data runs past the end of the file -- callers on a
        rendering path should catch that the same way they catch load_sld()
        returning None. Also raises it when there are no bytes to decode from,
        or when `data` is not the length the walk saw: the file changed on disk
        since, and every offset in this index may now point at the wrong bytes.
        """
        if data is None:
            data = self.data
            if data is None:
                raise SLDError("walked with keep_data=False and decoded without passing the file's bytes")
        elif len(data) != self.byte_length:
            raise SLDError(f"file is {len(data)} bytes but was walked at {self.byte_length}; it changed on disk")
        frame = self.frames[index]
        native = composite_backend.native
        arrays: dict[LayerKind, np.ndarray] = {}
        for layer in frame.layers:
            if kinds is not None and layer.kind not in kinds:
                continue
            if native is not None and layer.width > 0 and layer.height > 0:
                arrays[layer.kind] = self._decode_layer_native(native, layer, data, frame.width, frame.height)
                continue
            image = self._decode_layer_image(layer, data)
            if image is not None:
                arrays[layer.kind] = _paste_on_canvas(image, layer.x1, layer.y1, frame.width, frame.height)
        x0, y0, x1, y1 = 0, 0, frame.width, frame.height
        if window_to_main:
            main = next((layer for layer in frame.layers if layer.kind is LayerKind.MAIN), None)
            if main is not None and main.width > 0 and main.height > 0:
                x0, y0 = min(main.x1, frame.width), min(main.y1, frame.height)
                x1, y1 = min(main.x1 + main.width, frame.width), min(main.y1 + main.height, frame.height)
                # Views: _cropped_to_ink copies only what it keeps.
                arrays = {kind: pixels[y0:y1, x0:x1] for kind, pixels in arrays.items()}
        return DecodedFrame(
            width=x1 - x0,
            height=y1 - y0,
            hotspot_x=frame.hotspot_x - x0,
            hotspot_y=frame.hotspot_y - y0,
            main=arrays.get(LayerKind.MAIN),
            shadow=arrays.get(LayerKind.SHADOW),
            damage=arrays.get(LayerKind.DAMAGE),
            playercolor=arrays.get(LayerKind.PLAYERCOLOR),
        )

    def _decode_layer_native(self, native, layer: SLDLayer, data: bytes, width: int, height: int) -> np.ndarray:
        """The layer's whole delta chain, decoded and pasted on its canvas in
        one kernel call. Where the kernel refuses, the numpy path raises the
        exact SLDError."""
        chain = self._chains[_KIND_POSITION[layer.kind]]
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        if native.sld_decode_layer(data, self._layers, chain, chain[layer.chain_pos], layer.kind in _BC1_KINDS, canvas):
            return canvas
        image = self._decode_layer_image(layer, data)
        _native_disagreed("decode")
        return _paste_on_canvas(image, layer.x1, layer.y1, width, height)

    def _decode_layer_image(self, layer: SLDLayer, data: bytes) -> np.ndarray | None:
        if layer.width <= 0 or layer.height <= 0:
            return None
        blocks = self._decode_layer_blocks(layer, data)
        return _blocks_to_image(blocks, layer.width // BLOCK_PX, layer.height // BLOCK_PX)

    def _decode_layer_blocks(self, layer: SLDLayer, data: bytes) -> np.ndarray:
        """Resolves the layer's delta chain iteratively and returns its blocks.

        Walks back to the nearest same-kind layer that isn't a delta, then
        decodes forward. Recursing instead would sit near Python's frame limit
        on the deepest real chain (313 layers in a stock unit's idle graphic).
        """
        chain = self._chains[_KIND_POSITION[layer.kind]]
        links = [layer]
        position = layer.chain_pos
        while position > 0 and links[-1].is_delta:
            position -= 1
            links.append(self._layer(chain[position]))
        links.reverse()

        # links[0] has either no predecessor at all or one the delta rule
        # doesn't reach back to, so the first decode always stands alone.
        blocks = self._decode_blocks(links[0], None, None, data)
        for previous, current in pairwise(links):
            blocks = self._decode_blocks(current, blocks, previous, data)
        return blocks

    def _decode_blocks(
        self,
        layer: SLDLayer,
        previous_blocks: np.ndarray | None,
        previous_layer: SLDLayer | None,
        data: bytes,
    ) -> np.ndarray:
        """One layer's blocks as (block_count, 16, 4) uint8 RGBA."""
        out = np.zeros((layer.block_count, PIXELS_PER_BLOCK, 4), dtype=np.uint8)
        skip_runs, draw_runs, filled = _command_runs(data, layer)
        if filled > layer.block_count:
            raise SLDError(
                f"{layer.kind.name} command stream fills {filled} blocks, past this layer's {layer.block_count}"
            )
        # Under-filling is normal, not corruption: 137 surveyed layers stop
        # early and leave the tail transparent, which the zeros above already are.

        draw_dest = _run_indices(draw_runs)
        if draw_dest.size:
            end = layer.block_offset + BLOCK_BYTES * draw_dest.size
            if end > len(data):
                raise SLDError(f"{layer.kind.name} block data runs past the end of the file")
            raw = np.frombuffer(
                data, dtype=np.uint8, count=BLOCK_BYTES * draw_dest.size, offset=layer.block_offset
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


class _FrameView(Sequence):
    """SLDFile.frames: a read-only sequence building each SLDFrame on access."""

    __slots__ = ("_sld",)

    def __init__(self, sld: SLDFile):
        self._sld = sld

    def __len__(self) -> int:
        return self._sld.frame_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._sld._frame(i) for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError("frame index out of range")
        return self._sld._frame(index)


def _layer_read_error(data: bytes, start: int, kind: LayerKind, position: int) -> SLDError:
    """The error for a layer whose one-shot head unpack ran off the end: the
    one reading its length, then checking it, then its sub-header would hit."""
    if start + _LAYER_LENGTH.size > len(data):
        return SLDError(f"frame {position} {kind.name} layer length runs past the end of the file")
    (length,) = _LAYER_LENGTH.unpack_from(data, start)
    if length < _LAYER_LENGTH.size or start + length > len(data):
        return SLDError(f"frame {position} {kind.name} layer length {length} is out of range")
    return SLDError(f"frame {position} {kind.name} layer header runs past the end of the file")


def load_sld(path: str | Path, *, keep_data: bool = True) -> SLDFile | None:
    """Walks a .sld file, or returns None if it can't be read at all.

    Never raises: an unreadable, missing or unparseable file must be exactly as
    safe for a caller as having no AoE2:DE install configured, which is what the
    colored-mark fallback already handles. keep_data: see SLDFile.
    """
    try:
        return SLDFile.from_file(path, keep_data=keep_data)
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
