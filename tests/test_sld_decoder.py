"""Verifies descape.sld_decoder -- phase 3's P3-g0.

Every .sld here is built byte-by-byte in this file. The repo ships no game
assets and tests/conftest.py deliberately points CONFIG_PATH at a file that
doesn't exist, so the suite never sees a real AoE2:DE install; a synthetic
builder is the only way to cover this at the default tier.

Two checks carry most of the weight:

**The scalar oracle.** decode_bc1_blocks/decode_bc4_blocks expand every block
at once with numpy, which is easy to get subtly wrong in a way that still
produces a plausible image. _scalar_bc1/_scalar_bc4 below are transcribed from
openage's sld.pyx branch-for-branch -- the `if index == 0b00` chains, the
per-byte right shifts, the push order -- rather than re-derived from the
vectorized code, so agreement between them means something. Same shape
tests/test_unit_pick.py's ID-plane oracle already uses.

**Navigation.** Frame headers are interleaved with layer data, so one wrong
step anywhere silently desyncs everything after it rather than raising. The
builder pads with 0xAA instead of zeros precisely so a mis-step reads as
garbage, and several tests assert on a *later* frame's contents, which is the
only thing that actually proves the earlier step landed right.

The sub-header size for DAMAGE/PLAYERCOLOR needs its own test for the opposite
reason: navigation jumps to an absolute offset, so getting that size wrong stays
contained inside the layer and produces wrong pixels with a perfectly clean
walk. No real-file check can catch it.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import struct

import numpy as np
import pytest

from descape import asset_source, composite_backend, sld_decoder, unit_sprites
from descape.sld_decoder import (
    MAGIC,
    DecodedFrame,
    LayerKind,
    SLDError,
    SLDFile,
    SLDLayer,
    decode_bc1_blocks,
    decode_bc4_blocks,
    load_sld,
)

RNG_SEED = 20260820

# Non-zero filler for the absolute pad-to-4 between layers. Zeros would make a
# navigation error decode as innocuous transparency.
PAD_BYTE = 0xAA

_HEADER = struct.Struct("<4s4HI")
_HEADER_14 = struct.Struct("<4s4HH")
_FRAME_HEADER = struct.Struct("<4H2BH")

_LAYER_ORDER = (
    LayerKind.MAIN,
    LayerKind.SHADOW,
    LayerKind.OUTLINE,
    LayerKind.DAMAGE,
    LayerKind.PLAYERCOLOR,
)


# -- synthetic .sld builder ----------------------------------------------


def bc1_block(color0: int, color1: int, indices: list[int]) -> bytes:
    """One BC1 block: two RGB565 endpoints, then 16 two-bit indices, four to a
    byte, low bits first, one byte per pixel row."""
    packed = bytes(sum(indices[row * 4 + i] << (2 * i) for i in range(4)) for row in range(4))
    return struct.pack("<HH", color0, color1) + packed


def bc4_block(a0: int, a1: int, indices: list[int]) -> bytes:
    """One BC4 block: two endpoint bytes, then 16 three-bit indices in two
    little-endian 3-byte groups of eight."""
    out = bytes([a0, a1])
    for group in (indices[:8], indices[8:]):
        bits = sum(value << (3 * i) for i, value in enumerate(group))
        out += bits.to_bytes(3, "little")
    return out


def solid_bc1(color0: int = 0xF800) -> bytes:
    """A block that is entirely endpoint 0. 0xF800 is pure red, which the
    decoder's *8/*4/*8 endpoint expansion turns into (248, 0, 0)."""
    return bc1_block(color0, 0x0000, [0] * 16)


def solid_bc4(value: int = 200) -> bytes:
    return bc4_block(value, 0, [0] * 16)


class Layer:
    """One layer of a synthetic frame, in the shape the builder wants."""

    def __init__(
        self,
        kind: LayerKind,
        *,
        box: tuple[int, int, int, int] = (0, 0, 4, 4),
        flag0: int = 0,
        flag1: int = 0,
        commands: list[tuple[int, int]] | None = None,
        blocks: list[bytes] | None = None,
        length_slack: int = 0,
    ):
        self.kind = kind
        self.box = box
        self.flag0 = flag0
        self.flag1 = flag1
        self.blocks = blocks if blocks is not None else [solid_bc1()]
        self.commands = commands if commands is not None else [(0, len(self.blocks))]
        # Trailing bytes inside the layer's own declared length, so a test can
        # drive the layer length to any residue mod 4 it likes.
        self.length_slack = length_slack

    def to_bytes(self) -> bytes:
        if self.kind is LayerKind.OUTLINE:
            body = b""
        elif self.kind in (LayerKind.MAIN, LayerKind.SHADOW):
            x1, y1, x2, y2 = self.box
            body = struct.pack("<4H2B", x1, y1, x2, y2, self.flag0, self.flag1)
        else:
            body = struct.pack("<2B", self.flag0, self.flag1)
        if self.kind is not LayerKind.OUTLINE:
            body += struct.pack("<H", len(self.commands))
            body += b"".join(bytes(pair) for pair in self.commands)
            body += b"".join(self.blocks)
        body += bytes([PAD_BYTE]) * self.length_slack
        return struct.pack("<I", 4 + len(body)) + body


class Frame:
    def __init__(
        self,
        layers: list[Layer],
        *,
        canvas: tuple[int, int] = (8, 8),
        hotspot: tuple[int, int] = (4, 4),
        frame_index: int | None = None,
        frame_type: int | None = None,
    ):
        self.layers = layers
        self.canvas = canvas
        self.hotspot = hotspot
        self.frame_index = frame_index
        self.frame_type = frame_type


def build_sld(
    frames: list[Frame],
    *,
    magic: bytes = MAGIC,
    layout_tag: int = 16,
    version: int = 4,
) -> bytes:
    """Assembles a whole .sld, padding each absolute layer boundary up to the
    next offset congruent to the header size mod 4.

    layout_tag is the header size, so variant 14 shifts every boundary in the
    file by 2 relative to variant 16 -- that phase shift is the only structural
    difference between the two on disk.
    """
    header = _HEADER_14 if layout_tag == 14 else _HEADER
    out = bytearray(header.pack(magic, version, len(frames), 0, layout_tag, 0))
    for position, frame in enumerate(frames):
        frame_type = frame.frame_type
        if frame_type is None:
            frame_type = 0
            for layer in frame.layers:
                frame_type |= int(layer.kind)
        frame_index = position if frame.frame_index is None else frame.frame_index
        out += _FRAME_HEADER.pack(*frame.canvas, *frame.hotspot, frame_type, 0, frame_index)
        for kind in _LAYER_ORDER:
            for layer in frame.layers:
                if layer.kind is not kind:
                    continue
                out += layer.to_bytes()
                out += bytes([PAD_BYTE]) * ((layout_tag - len(out)) % 4)
    return bytes(out)


def independent_walk(data: bytes) -> tuple[list[int], int]:
    """Where each frame header actually starts, and the offset just past the
    final layer's declared length -- walked independently of the decoder so a
    test can assert against it rather than against itself.

    That second number is where the file stops being load-bearing: any byte
    before it is structure the walk reads, everything after it is trailing pad.
    """
    offsets = []
    frame_count, layout_tag = struct.unpack_from("<H", data, 6)[0], struct.unpack_from("<H", data, 10)[0]
    offset = layout_tag
    end = offset
    for _ in range(frame_count):
        offsets.append(offset)
        frame_type = _FRAME_HEADER.unpack_from(data, offset)[4]
        offset += _FRAME_HEADER.size
        end = offset
        for kind in _LAYER_ORDER:
            if not frame_type & kind:
                continue
            start = offset
            (length,) = struct.unpack_from("<I", data, offset)
            end = start + length
            offset = end + (layout_tag - end) % 4
    return offsets, end


# -- the scalar oracle, transcribed from openage's sld.pyx ---------------


def _scalar_bc1(data: bytes, offset: int) -> list[list[int]]:
    c0_val = data[offset] + (data[offset + 1] << 8)
    c1_val = data[offset + 2] + (data[offset + 3] << 8)

    c0 = [
        ((data[offset + 1] & 0b1111_1000) >> 3) * 8,
        (((data[offset + 1] & 0b0000_0111) << 3) + ((data[offset] & 0b1110_0000) >> 5)) * 4,
        (data[offset] & 0b0001_1111) * 8,
        255,
    ]
    c1 = [
        ((data[offset + 3] & 0b1111_1000) >> 3) * 8,
        (((data[offset + 3] & 0b0000_0111) << 3) + ((data[offset + 2] & 0b1110_0000) >> 5)) * 4,
        (data[offset + 2] & 0b0001_1111) * 8,
        255,
    ]

    if c0_val > c1_val:
        c2 = [(2 * c0[i] + c1[i] + 1) // 3 for i in range(3)] + [255]
        c3 = [(c0[i] + 2 * c1[i] + 1) // 3 for i in range(3)] + [255]
    else:
        c2 = [(c0[i] + c1[i]) // 2 for i in range(3)] + [255]
        c3 = [0, 0, 0, 0]

    block = []
    position = offset + 4
    for _ in range(4):
        byte_val = data[position]
        for _ in range(4):
            index = byte_val & 0b11
            if index == 0b00:
                block.append(c0)
            elif index == 0b01:
                block.append(c1)
            elif index == 0b10:
                block.append(c2)
            elif index == 0b11:
                block.append(c3)
            byte_val = byte_val >> 2
        position += 1
    return block


def _scalar_bc4(data: bytes, offset: int) -> list[list[int]]:
    a0 = data[offset]
    a1 = data[offset + 1]
    c0 = [a0, 0, 0, 255]
    c1 = [a1, 0, 0, 255]
    if a0 > a1:
        table = [c0, c1] + [[((7 - i) * a0 + i * a1) // 7, 0, 0, 255] for i in range(1, 7)]
    else:
        table = (
            [c0, c1]
            + [[((5 - i) * a0 + i * a1) // 5, 0, 0, 255] for i in range(1, 5)]
            + [[0, 0, 0, 0], [255, 0, 0, 255]]
        )

    block = []
    position = offset + 2
    for _ in range(2):
        pixel_indices = data[position] | (data[position + 1] << 8) | (data[position + 2] << 16)
        for _ in range(8):
            block.append(table[pixel_indices & 0b111])
            pixel_indices = pixel_indices >> 3
        position += 3
    return block


def scalar_blocks(raw: np.ndarray, decoder) -> np.ndarray:
    data = raw.tobytes()
    return np.array([decoder(data, i * 8) for i in range(raw.shape[0])], dtype=np.uint8)


# -- walk ----------------------------------------------------------------


def test_frame_headers_are_interleaved_with_their_own_layer_data():
    """The bug the planning spike actually hit: reading the format as one table
    of frame headers followed by all layer data. Frame 0 carries three layers
    here, so frame 1's header only lands right if all three were stepped over
    individually."""
    data = build_sld(
        [
            Frame(
                [
                    Layer(LayerKind.MAIN, box=(0, 0, 4, 4)),
                    Layer(LayerKind.SHADOW, box=(0, 0, 8, 8), blocks=[solid_bc4()] * 4),
                    Layer(LayerKind.PLAYERCOLOR, blocks=[solid_bc4(64)]),
                ],
                canvas=(8, 8),
            ),
            Frame([Layer(LayerKind.MAIN, blocks=[solid_bc1(0x001F)])], canvas=(12, 12), hotspot=(6, 6)),
        ]
    )
    sld = SLDFile(data)

    assert sld.frame_count == 2
    assert [layer.kind for layer in sld.frames[0].layers] == [
        LayerKind.MAIN,
        LayerKind.SHADOW,
        LayerKind.PLAYERCOLOR,
    ]
    # Frame 1's own geometry, which is only readable from the right offset.
    assert (sld.frames[1].width, sld.frames[1].height) == (12, 12)
    assert (sld.frames[1].hotspot_x, sld.frames[1].hotspot_y) == (6, 6)
    assert sld.decode_frame(1).main[0, 0, 2] == 248  # 0x001F is pure blue


@pytest.mark.parametrize("slack", [0, 1, 2, 3])
def test_pad_to_four_applies_to_the_absolute_offset_not_the_layer_size(slack):
    """A layer's own byte count and its absolute end offset have different
    residues mod 4 in general. Padding the wrong one desyncs every later frame,
    so this asserts on frame 1, not frame 0."""
    data = build_sld(
        [
            Frame([Layer(LayerKind.MAIN, length_slack=slack)]),
            Frame([Layer(LayerKind.MAIN, blocks=[solid_bc1(0x07E0)])], canvas=(16, 16)),
        ]
    )
    assert independent_walk(data)[0][1] % 4 == 0

    sld = SLDFile(data)
    assert (sld.frames[1].width, sld.frames[1].height) == (16, 16)
    assert sld.decode_frame(1).main[0, 0, 1] == 252  # 0x07E0 is pure green


def test_outline_layer_is_skipped_but_still_stepped_over():
    """OUTLINE has no decode even upstream. Its bytes are deliberately not
    parsed at all -- openage does read a command count off them and gets
    garbage, which it only survives because the jump is absolute."""
    data = build_sld(
        [
            Frame(
                [
                    Layer(LayerKind.MAIN),
                    Layer(LayerKind.OUTLINE, length_slack=17),
                    Layer(LayerKind.PLAYERCOLOR, blocks=[solid_bc4(128)]),
                ]
            ),
            Frame([Layer(LayerKind.MAIN)], canvas=(20, 20)),
        ]
    )
    sld = SLDFile(data)

    kinds = [layer.kind for layer in sld.frames[0].layers]
    assert LayerKind.OUTLINE not in kinds
    assert kinds == [LayerKind.MAIN, LayerKind.PLAYERCOLOR]
    decoded = sld.decode_frame(0)
    assert decoded.playercolor is not None
    assert decoded.playercolor[0, 0, 0] == 128
    assert sld.frames[1].width == 20


def test_mask_layers_use_a_two_byte_subheader_and_reuse_mains_box():
    """DAMAGE and PLAYERCOLOR carry only two flag bytes, no box of their own.
    Reading a 10-byte box here instead would consume the command count as box
    bytes and still navigate perfectly -- only the pixels would be wrong."""
    data = build_sld(
        [
            Frame(
                [
                    Layer(LayerKind.MAIN, box=(4, 8, 12, 16), blocks=[solid_bc1()] * 4),
                    Layer(LayerKind.DAMAGE, blocks=[solid_bc1(0x001F)] * 4),
                    Layer(LayerKind.PLAYERCOLOR, blocks=[solid_bc4(77)] * 4),
                ],
                canvas=(24, 24),
            ),
            Frame([Layer(LayerKind.MAIN)], canvas=(28, 28)),
        ]
    )
    sld = SLDFile(data)

    damage, playercolor = sld.frames[0].layers[1], sld.frames[0].layers[2]
    assert (damage.x1, damage.y1, damage.width, damage.height) == (4, 8, 8, 8)
    assert (playercolor.x1, playercolor.y1, playercolor.width, playercolor.height) == (4, 8, 8, 8)

    decoded = sld.decode_frame(0)
    assert decoded.damage[8, 4, 2] == 248 and decoded.damage[0, 0, 3] == 0
    assert decoded.playercolor[8, 4, 0] == 77
    assert sld.frames[1].width == 28


def test_a_frame_with_no_layers_parses_to_an_empty_frame():
    """frame_type 0x00 is real: 1,322 frames of the surveyed sample have it."""
    data = build_sld(
        [
            Frame([], frame_type=0),
            Frame([Layer(LayerKind.MAIN)], canvas=(32, 32)),
        ]
    )
    sld = SLDFile(data)

    assert sld.frames[0].layers == ()
    decoded = sld.decode_frame(0)
    assert isinstance(decoded, DecodedFrame)
    assert (decoded.main, decoded.shadow, decoded.damage, decoded.playercolor) == (None, None, None, None)
    assert sld.frames[1].width == 32


def test_layers_of_one_kind_are_chained_in_walk_order_across_frames():
    """chain_pos is what the delta rule resolves a predecessor through, and it
    counts layers of one kind, not frames: a MAIN chain steps over every frame
    that has no MAIN layer at all."""
    frames = [
        Frame([Layer(LayerKind.MAIN), Layer(LayerKind.SHADOW, blocks=[solid_bc4()])]),
        Frame([Layer(LayerKind.SHADOW, blocks=[solid_bc4()])]),
        Frame([Layer(LayerKind.SHADOW, blocks=[solid_bc4()])]),
        Frame([Layer(LayerKind.MAIN)]),
    ]
    sld = SLDFile(build_sld(frames))

    assert [layer.chain_pos for layer in sld.frames[0].layers] == [0, 0]
    assert sld.frames[3].layers[0].chain_pos == 1
    assert [layer.frame_index for layer in sld.frames[3].layers] == [3]


def test_the_delta_flag_and_frame_index_survive_the_walk():
    """is_delta needs both halves of the rule: the bit, and a frame_index past
    zero. A flagged frame 0 has no predecessor to copy from."""
    data = build_sld(
        [
            Frame([Layer(LayerKind.MAIN, flag0=0x81, flag1=0x01)], frame_index=0),
            Frame([Layer(LayerKind.MAIN, flag0=0x81)]),
            Frame([Layer(LayerKind.MAIN, flag0=0x01)]),
        ]
    )
    sld = SLDFile(data)

    assert (sld.frames[0].layers[0].flag0, sld.frames[0].layers[0].flag1) == (0x81, 0x01)
    assert not sld.frames[0].layers[0].is_delta
    assert sld.frames[1].layers[0].is_delta
    assert not sld.frames[2].layers[0].is_delta


# -- rejection -----------------------------------------------------------


@pytest.mark.parametrize("layout_tag", [14, 16])
def test_both_layout_variants_walk_to_the_same_decoded_frame(layout_tag):
    """226 of the install's 7,372 files carry 14 in the u16 at header offset 10
    instead of 16. The tag is the header's byte length and nothing else in the
    file differs, so the same frames must decode identically under either."""
    frames = [
        Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4))], canvas=(8, 8)),
        Frame([Layer(LayerKind.MAIN, blocks=[solid_bc1(0x001F)])], canvas=(12, 12), hotspot=(6, 6)),
    ]
    sld = SLDFile(build_sld(frames, layout_tag=layout_tag))

    assert sld.frame_count == 2
    assert (sld.frames[1].width, sld.frames[1].height) == (12, 12)
    assert (sld.frames[1].hotspot_x, sld.frames[1].hotspot_y) == (6, 6)
    assert sld.decode_frame(1).main[0, 0, 2] == 248  # 0x001F is pure blue


@pytest.mark.parametrize("slack", [0, 1, 2, 3])
def test_variant_fourteen_pads_boundaries_to_two_mod_four_not_zero(slack):
    """The whole of variant 14's "desync further in". Its 14-byte header puts
    every later boundary at offset % 4 == 2, so padding up to a plain multiple
    of 4 -- correct for variant 16 -- overshoots frame 1's header by two bytes
    and reads a garbage layer length.

    The slack sweep is what makes this non-vacuous: only the arms where frame
    0's layer happens to end on a multiple of 4 require the pad at all, and an
    unswept fixture would pass under both rules while measuring nothing.
    """
    frames = [
        Frame([Layer(LayerKind.MAIN, length_slack=slack)]),
        Frame([Layer(LayerKind.MAIN, blocks=[solid_bc1(0x07E0)])], canvas=(16, 16)),
    ]
    data = build_sld(frames, layout_tag=14)
    assert independent_walk(data)[0][1] % 4 == 2

    sld = SLDFile(data)
    assert (sld.frames[1].width, sld.frames[1].height) == (16, 16)
    assert sld.decode_frame(1).main[0, 0, 1] == 252  # 0x07E0 is pure green


def test_a_layout_variant_that_is_neither_fourteen_nor_sixteen_is_rejected():
    """openage hardcodes 16; this reads 14 as well. Anything else is a file
    shape nobody has measured, so it takes the caller's fallback rather than
    being walked on the assumption that the tag is always the header size."""
    frames = [Frame([Layer(LayerKind.MAIN)])]
    with pytest.raises(SLDError, match="layout variant 12"):
        SLDFile(build_sld(frames, layout_tag=12))


def test_bad_magic_is_rejected():
    with pytest.raises(SLDError, match="magic"):
        SLDFile(build_sld([Frame([Layer(LayerKind.MAIN)])], magic=b"SLPX"))


@pytest.mark.parametrize("size", [0, 4, 8, 15])
def test_a_file_too_short_for_a_header_is_rejected(size):
    """One file in the real install is zero bytes long."""
    with pytest.raises(SLDError):
        SLDFile(build_sld([Frame([Layer(LayerKind.MAIN)])])[:size])


def test_truncation_mid_walk_is_rejected_rather_than_read_past_the_end():
    """Every cut inside the structure raises. Cuts past it fall in the trailing
    pad, which the walk never reads, so they are not truncation at all."""
    data = build_sld([Frame([Layer(LayerKind.MAIN)]), Frame([Layer(LayerKind.MAIN)])])
    for cut in range(_HEADER.size + 1, independent_walk(data)[1]):
        with pytest.raises(SLDError):
            SLDFile(data[:cut])


def test_a_command_stream_that_overruns_its_own_block_grid_is_rejected():
    """Under-filling is normal and pads transparent; over-filling is corruption
    with nowhere to put the surplus."""
    data = build_sld(
        [Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), commands=[(0, 2)], blocks=[solid_bc1()] * 2)])]
    )
    with pytest.raises(SLDError, match="past this layer"):
        SLDFile(data).decode_frame(0)


def test_load_sld_returns_none_instead_of_raising(tmp_path):
    """An unreadable .sld must be exactly as safe for a caller as having no
    install configured at all -- including the likeliest real failure, a path
    that isn't there."""
    good = tmp_path / "good.sld"
    good.write_bytes(build_sld([Frame([Layer(LayerKind.MAIN)])]))
    assert load_sld(good) is not None

    corrupt = tmp_path / "corrupt.sld"
    corrupt.write_bytes(build_sld([Frame([Layer(LayerKind.MAIN)])], magic=b"XXXX"))
    assert load_sld(corrupt) is None

    empty = tmp_path / "empty.sld"
    empty.write_bytes(b"")
    assert load_sld(empty) is None

    assert load_sld(tmp_path / "not_there.sld") is None
    assert load_sld(tmp_path) is None  # a directory: OSError, not SLDError


# -- block codecs --------------------------------------------------------


def test_bc1_four_color_mode_expands_endpoints_and_interpolates():
    """c0 > c1: four opaque colors. Endpoints expand by *8/*4/*8, matching
    openage rather than 5-to-8-bit replication, so 0xFFFF is (248, 252, 248)."""
    white, black = 0xFFFF, 0x0000
    out = decode_bc1_blocks(np.frombuffer(bc1_block(white, black, [0, 1, 2, 3] * 4), dtype=np.uint8).reshape(1, 8))

    assert out.shape == (1, 16, 4)
    assert list(out[0, 0]) == [248, 252, 248, 255]
    assert list(out[0, 1]) == [0, 0, 0, 255]
    assert list(out[0, 2]) == [165, 168, 165, 255]  # (2*c0 + c1 + 1) // 3
    assert list(out[0, 3]) == [83, 84, 83, 255]  # (c0 + 2*c1 + 1) // 3


def test_bc1_three_color_mode_makes_index_three_fully_transparent():
    white, black = 0xFFFF, 0x0000
    out = decode_bc1_blocks(np.frombuffer(bc1_block(black, white, [0, 1, 2, 3] * 4), dtype=np.uint8).reshape(1, 8))

    assert list(out[0, 2]) == [124, 126, 124, 255]  # (c0 + c1) // 2
    assert list(out[0, 3]) == [0, 0, 0, 0]


def test_bc1_indices_are_two_bits_low_first_one_byte_per_row():
    """Row-major within the block: index byte j is row j, and the low bits of
    each byte are the leftmost pixels."""
    block = bc1_block(0xFFFF, 0x0000, [0, 1, 0, 1] + [1] * 4 + [0] * 8)
    out = decode_bc1_blocks(np.frombuffer(block, dtype=np.uint8).reshape(1, 8))

    assert [int(px[0]) for px in out[0, 0:4]] == [248, 0, 248, 0]
    assert [int(px[0]) for px in out[0, 4:8]] == [0, 0, 0, 0]
    assert [int(px[0]) for px in out[0, 8:12]] == [248] * 4


def test_bc4_wide_mode_interpolates_six_values_over_sevenths():
    out = decode_bc4_blocks(np.frombuffer(bc4_block(210, 70, list(range(8)) * 2), dtype=np.uint8).reshape(1, 8))

    expected = [210, 70] + [((7 - i) * 210 + i * 70) // 7 for i in range(1, 7)]
    assert [int(px[0]) for px in out[0, 0:8]] == expected
    assert all(int(px[3]) == 255 for px in out[0])
    assert all(int(px[1]) == 0 and int(px[2]) == 0 for px in out[0])


def test_bc4_narrow_mode_reserves_index_six_for_transparent_and_seven_for_full():
    """The mode that is easy to get backwards: it yields a plausible gradient
    either way, and only index 6 and 7 give it away."""
    out = decode_bc4_blocks(np.frombuffer(bc4_block(70, 210, list(range(8)) * 2), dtype=np.uint8).reshape(1, 8))

    expected = [70, 210] + [((5 - i) * 70 + i * 210) // 5 for i in range(1, 5)]
    assert [int(px[0]) for px in out[0, 0:6]] == expected
    assert list(out[0, 6]) == [0, 0, 0, 0]
    assert list(out[0, 7]) == [255, 0, 0, 255]


def test_bc4_indices_split_into_two_three_byte_groups_of_eight():
    # The `+` is the point: it shows the two groups of eight this test is
    # named for. Merged into one 16-list they stop reading as groups at all.
    indices = [7, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 7]  # noqa: RUF005
    out = decode_bc4_blocks(np.frombuffer(bc4_block(255, 0, indices), dtype=np.uint8).reshape(1, 8))

    assert int(out[0, 0, 0]) == (1 * 255 + 6 * 0) // 7
    assert int(out[0, 15, 0]) == (1 * 255 + 6 * 0) // 7
    assert int(out[0, 1, 0]) == 255


@pytest.mark.parametrize("decoder", [decode_bc1_blocks, decode_bc4_blocks])
def test_vectorized_block_decode_matches_the_scalar_oracle(decoder):
    """The check the vectorized path exists to be trusted by. The oracle is
    transcribed from openage's own branch structure, not derived from this
    module, so a shared misreading cannot make both agree."""
    rng = np.random.default_rng(RNG_SEED)
    raw = rng.integers(0, 256, size=(4000, 8), dtype=np.uint8)
    scalar = _scalar_bc1 if decoder is decode_bc1_blocks else _scalar_bc4

    assert np.array_equal(decoder(raw), scalar_blocks(raw, scalar))


# -- commands and the delta chain ----------------------------------------


def test_a_skip_writes_transparent_when_the_layer_is_not_a_delta():
    data = build_sld(
        [
            Frame(
                [Layer(LayerKind.MAIN, box=(0, 0, 8, 8), commands=[(3, 1)], blocks=[solid_bc1()])],
                canvas=(8, 8),
            )
        ]
    )
    main = SLDFile(data).decode_frame(0).main

    assert main[0:4, 0:8, 3].max() == 0  # blocks 0 and 1: skipped
    assert main[4:8, 0:4, 3].max() == 0  # block 2: skipped
    assert main[4, 4, 0] == 248  # block 3: drawn


def test_an_under_filled_command_stream_leaves_the_tail_transparent():
    """137 surveyed real layers stop short of their own block grid. That is not
    corruption, and must not be treated as such."""
    data = build_sld(
        [Frame([Layer(LayerKind.MAIN, box=(0, 0, 8, 8), commands=[(0, 1)], blocks=[solid_bc1()])], canvas=(8, 8))]
    )
    main = SLDFile(data).decode_frame(0).main

    assert main[0, 0, 0] == 248
    assert main[4:8, 4:8, 3].max() == 0


def _delta_chain(depth: int, *, kind: LayerKind = LayerKind.MAIN, block: bytes | None = None) -> bytes:
    """Frame 0 draws one block; every later frame skips it as a delta."""
    block = block if block is not None else solid_bc1()
    frames = [Frame([Layer(kind, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[block])], canvas=(4, 4))]
    frames.extend(
        Frame(
            [Layer(kind, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])],
            canvas=(4, 4),
        )
        for _ in range(depth)
    )
    return build_sld(frames)


def test_a_delta_skip_copies_the_previous_same_kind_layers_block():
    sld = SLDFile(_delta_chain(1))

    assert np.array_equal(sld.decode_frame(1).main, sld.decode_frame(0).main)
    assert sld.decode_frame(1).main[0, 0, 0] == 248


def test_the_delta_flag_is_ignored_on_the_first_frame():
    """The rule is flag0 & 0x80 AND frame_index > 0. A flagged frame 0 has no
    predecessor to copy from and must come out transparent, not crash."""
    data = build_sld(
        [
            Frame(
                [Layer(LayerKind.MAIN, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])],
                canvas=(4, 4),
                frame_index=0,
            )
        ]
    )
    assert SLDFile(data).decode_frame(0).main[..., 3].max() == 0


def test_the_same_kind_predecessor_is_not_always_the_previous_frame():
    """960 frames of the surveyed sample carry SHADOW with no MAIN at all, so a
    MAIN delta's predecessor can sit several frames back. Resolving it as
    frames[index - 1] would copy from a SHADOW layer -- a different codec and a
    different block grid."""
    frames = [
        Frame(
            [
                Layer(LayerKind.MAIN, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[solid_bc1(0x001F)]),
                Layer(LayerKind.SHADOW, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[solid_bc4(90)]),
            ],
            canvas=(4, 4),
        )
    ]
    frames.extend(
        Frame(
            [Layer(LayerKind.SHADOW, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[solid_bc4(30)])],
            canvas=(4, 4),
        )
        for _ in range(3)
    )
    frames.append(
        Frame(
            [Layer(LayerKind.MAIN, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])],
            canvas=(4, 4),
        )
    )
    decoded = SLDFile(build_sld(frames)).decode_frame(4)

    assert list(decoded.main[0, 0]) == [0, 0, 248, 255]  # frame 0's blue MAIN block
    assert decoded.shadow is None


def test_a_delta_copy_remaps_block_indices_when_the_boxes_differ():
    """Only each layer's own box is stored, so a block copied between layers
    sitting at different offsets in the frame moves with the difference. Blocks
    landing outside the previous layer come back transparent."""
    frames = [
        Frame(
            [
                Layer(
                    LayerKind.MAIN,
                    box=(0, 0, 8, 4),
                    commands=[(0, 2)],
                    blocks=[solid_bc1(0x001F), solid_bc1(0xF800)],
                )
            ],
            canvas=(16, 16),
        ),
        # Shifted one block right: its block 0 maps onto the predecessor's
        # block 1, and its block 1 falls off the right-hand edge.
        Frame(
            [Layer(LayerKind.MAIN, box=(4, 0, 12, 4), flag0=0x80, commands=[(2, 0)], blocks=[])],
            canvas=(16, 16),
        ),
    ]
    main = SLDFile(build_sld(frames)).decode_frame(1).main

    assert list(main[0, 4]) == [248, 0, 0, 255]  # the predecessor's red block
    assert main[0, 8:12, 3].max() == 0  # off the edge: transparent


def test_a_chain_hundreds_deep_resolves_without_recursion():
    """The deepest measured real chain is 313 consecutive delta layers, in a
    stock unit's idle graphic. Recursing one Python frame per link would sit
    close to the interpreter's own limit on a file scenarios actually use."""
    sld = SLDFile(_delta_chain(400))

    assert np.array_equal(sld.decode_frame(400).main, sld.decode_frame(0).main)
    assert sld.decode_frame(400).main[0, 0, 0] == 248


def test_decoding_a_frame_twice_gives_the_same_answer():
    """There is deliberately no cache in this module, so this is guarding
    against chain resolution mutating shared state rather than against a stale
    cache -- the block arrays a delta copies from are reused by every later
    link."""
    sld = SLDFile(_delta_chain(5))
    first = sld.decode_frame(5).main.copy()

    sld.decode_frame(2)
    assert np.array_equal(sld.decode_frame(5).main, first)


# -- canvas placement ----------------------------------------------------


def test_a_layer_is_pasted_at_its_own_box_offset_on_a_canvas_sized_array():
    """Every layer of a frame comes back in one coordinate system, so the frame
    hotspot anchors all of them at once."""
    data = build_sld(
        [
            Frame(
                [Layer(LayerKind.MAIN, box=(8, 12, 12, 16), commands=[(0, 1)], blocks=[solid_bc1()])],
                canvas=(32, 24),
                hotspot=(16, 12),
            )
        ]
    )
    decoded = SLDFile(data).decode_frame(0)

    assert (decoded.width, decoded.height) == (32, 24)
    assert (decoded.hotspot_x, decoded.hotspot_y) == (16, 12)
    assert decoded.main.shape == (24, 32, 4)
    assert list(decoded.main[12, 8]) == [248, 0, 0, 255]
    assert decoded.main[0:12, :, 3].max() == 0
    assert int((decoded.main[..., 3] > 0).sum()) == 16


def test_a_box_reaching_past_the_canvas_edge_is_clipped_not_wrapped():
    """165 surveyed real layers have a box extending past their own canvas."""
    data = build_sld(
        [
            Frame(
                [
                    Layer(
                        LayerKind.MAIN,
                        box=(4, 4, 12, 12),
                        commands=[(0, 4)],
                        blocks=[solid_bc1()] * 4,
                    )
                ],
                canvas=(8, 8),
            )
        ]
    )
    main = SLDFile(data).decode_frame(0).main

    assert main.shape == (8, 8, 4)
    assert list(main[4, 4]) == [248, 0, 0, 255]
    assert main[0:4, :, 3].max() == 0
    assert main[:, 0:4, 3].max() == 0


# -- released data (an index that holds no bytes) ------------------------


def _every_layer_kind() -> bytes:
    """All four decoded kinds plus a skipped OUTLINE in one frame, then a
    frame whose MAIN is a delta over it."""
    return build_sld(
        [
            Frame(
                [
                    Layer(LayerKind.MAIN, box=(0, 0, 8, 4), blocks=[solid_bc1(0x001F), solid_bc1()]),
                    Layer(LayerKind.SHADOW, box=(0, 0, 8, 8), blocks=[solid_bc4()] * 4),
                    Layer(LayerKind.OUTLINE),
                    Layer(LayerKind.DAMAGE, blocks=[solid_bc1(0x07E0)] * 2),
                    Layer(LayerKind.PLAYERCOLOR, blocks=[solid_bc4(64)] * 2),
                ],
                canvas=(8, 8),
            ),
            Frame(
                [Layer(LayerKind.MAIN, box=(0, 0, 8, 4), flag0=0x80, commands=[(1, 1)], blocks=[solid_bc1(0x0800)])],
                canvas=(8, 8),
            ),
        ]
    )


def _released_fixtures() -> dict[str, bytes]:
    return {
        "delta_chain": _delta_chain(5),
        "every_layer_kind": _every_layer_kind(),
        "variant_14_delta": build_sld(
            [
                Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[solid_bc1()])]),
                Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])]),
            ],
            layout_tag=14,
        ),
    }


def _assert_same_frame(got: DecodedFrame, want: DecodedFrame) -> None:
    assert (got.width, got.height, got.hotspot_x, got.hotspot_y) == (
        want.width,
        want.height,
        want.hotspot_x,
        want.hotspot_y,
    )
    for kind in ("main", "shadow", "damage", "playercolor"):
        a, b = getattr(got, kind), getattr(want, kind)
        assert (a is None) == (b is None), kind
        if a is not None:
            assert np.array_equal(a, b), kind


@pytest.mark.parametrize("name", sorted(_released_fixtures()))
def test_decoding_with_released_data_matches_decoding_with_the_bytes_held(name):
    """unit_sprites caches indexes walked with keep_data=False and passes each
    decode the file's bytes. Every pixel has to come out as it did when the
    index held them, and decoding must not stash the bytes back on the shared
    index."""
    data = _released_fixtures()[name]
    held = SLDFile(data)
    released = SLDFile(data, keep_data=False)

    assert released.data is None
    assert released.byte_length == len(data) == held.byte_length
    assert released.frame_count == held.frame_count > 1
    for index in range(held.frame_count):
        _assert_same_frame(released.decode_frame(index, bytes(data)), held.decode_frame(index))
    assert released.data is None


def test_load_sld_can_release_the_bytes(tmp_path):
    path = tmp_path / "t.sld"
    path.write_bytes(_delta_chain(2))

    sld = load_sld(path, keep_data=False)

    assert sld is not None and sld.data is None and sld.frame_count == 3
    assert load_sld(path).data == path.read_bytes()


def test_a_released_index_decoded_without_bytes_raises_sld_error():
    """SLDError, not a TypeError from inside numpy: a rendering caller only
    catches the former."""
    with pytest.raises(SLDError, match="keep_data=False"):
        SLDFile(_delta_chain(1), keep_data=False).decode_frame(1)


@pytest.mark.parametrize("change", ["grown", "shrunk"])
def test_a_file_whose_length_changed_since_its_walk_raises_sld_error(change):
    """The cached index's offsets belong to the bytes it walked. A file
    replaced on disk (a game update) is refused rather than decoded from
    offsets into the wrong bytes. Grown by pad bytes on purpose: the frame
    still decodes cleanly from those bytes, so only the length check stops it."""
    data = _delta_chain(3)
    changed = data + bytes([PAD_BYTE]) * 4 if change == "grown" else data[:-1]
    sld = SLDFile(data, keep_data=False)

    with pytest.raises(SLDError, match="changed on disk"):
        sld.decode_frame(3, changed)
    with pytest.raises(SLDError, match="changed on disk"):
        sld.decode_frame(3, changed, kinds=())
    assert sld.decode_frame(3, data).main[0, 0, 0] == 248


# -- decoding only some layers -------------------------------------------

_DECODED_KINDS = {
    "main": LayerKind.MAIN,
    "shadow": LayerKind.SHADOW,
    "damage": LayerKind.DAMAGE,
    "playercolor": LayerKind.PLAYERCOLOR,
}


def _kinds_fixtures() -> dict[str, bytes]:
    """Every decoded kind in one frame (plus a MAIN delta), and a delta SHADOW."""
    return {
        "every_layer_kind": _every_layer_kind(),
        "shadow_delta": _delta_chain(3, kind=LayerKind.SHADOW, block=solid_bc4(90)),
    }


def _refusing_kernel():
    return type("Refusing", (), {"sld_decode_layer": staticmethod(lambda *_args: False)})()


@pytest.mark.parametrize("backend", ["numpy", "native", "refused"])
@pytest.mark.parametrize("name", sorted(_kinds_fixtures()))
def test_a_kinds_filter_leaves_the_other_layers_undecoded_and_the_kept_ones_unchanged(
    backend, name, monkeypatch
):
    """unit_sprites reads only MAIN and PLAYERCOLOR, so it asks for only those.
    Every subset of the four kinds, on every frame: a filtered field is None, a
    kept one equals the unfiltered decode's. The kernel-refusal fallback counts
    one disagreement per layer it decodes, so it also shows a filtered layer is
    skipped before any decode, not decoded and dropped."""
    if backend != "numpy" and not composite_backend.available():
        if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
            pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
        pytest.skip(composite_backend.unavailable_reason)
    data = _kinds_fixtures()[name]
    with composite_backend.use_backend("numpy"):
        sld = SLDFile(data)
        want = [sld.decode_frame(index) for index in range(sld.frame_count)]
    fields = tuple(_DECODED_KINDS)
    subsets = [c for n in range(len(fields) + 1) for c in itertools.combinations(fields, n)]

    if backend == "refused":
        # Not inside use_backend: monkeypatch's undo would then put back the real
        # kernel use_backend swapped in, leaking native into every later test.
        monkeypatch.setattr(composite_backend, "native", _refusing_kernel())
        backend_context = contextlib.nullcontext()
    else:
        backend_context = composite_backend.use_backend(backend)
    with backend_context:
        kept_seen = 0
        for index, full in enumerate(want):
            for subset in subsets:
                before = sld_decoder.native_disagreements
                got = sld.decode_frame(index, kinds=[_DECODED_KINDS[f] for f in subset])
                assert (got.width, got.height, got.hotspot_x, got.hotspot_y) == (
                    full.width, full.height, full.hotspot_x, full.hotspot_y
                )
                kept = 0
                for field in fields:
                    a, b = getattr(got, field), getattr(full, field)
                    if field not in subset:
                        assert a is None, (index, subset, field)
                        continue
                    assert (a is None) == (b is None), (index, subset, field)
                    if b is not None:
                        assert np.array_equal(a, b), (index, subset, field)
                        kept += 1
                if backend == "refused":
                    assert sld_decoder.native_disagreements - before == kept, (index, subset)
                kept_seen += kept
    assert kept_seen > 0
    assert any(full.shadow is not None for full in want)


# -- the walk against a step-by-step reference ---------------------------


def reference_walk(data: bytes) -> list[tuple]:
    """The walk as it stood before it was flattened for speed (2026-09-28),
    transcribed step by step: one unpack per field, IntFlag tests, checks in
    their original order. The decoder's walk must agree with it on every field
    of every layer and on every error message.

    Each frame comes back as (width, height, hotspot_x, hotspot_y, frame_type,
    frame_index, layers), each layer as _layer_fields() spells it out.
    """
    try:
        magic, _version, frame_count, _unknown1, layout_tag = struct.unpack_from("<4s4H", data, 0)
    except struct.error as exc:
        raise SLDError(f"too short to hold an SLD header ({len(data)} bytes)") from exc
    if magic != MAGIC:
        raise SLDError(f"not an SLD file: magic {magic!r}")
    if layout_tag not in (14, 16):
        raise SLDError(f"unsupported layout variant {layout_tag} (only 14 or 16 is readable)")
    if len(data) < layout_tag:
        raise SLDError(f"too short to hold an SLD header ({len(data)} bytes)")

    chains = dict.fromkeys(_LAYER_ORDER, 0)
    frames = []
    offset = layout_tag
    for position in range(frame_count):
        try:
            width, height, hotspot_x, hotspot_y, frame_type, _unknown5, frame_index = _FRAME_HEADER.unpack_from(data, offset)
        except struct.error as exc:
            raise SLDError(f"frame {position} header runs past the end of the file") from exc
        offset += _FRAME_HEADER.size
        layers = []
        main_box = None
        for kind in _LAYER_ORDER:
            if not frame_type & kind:
                continue
            start = offset
            try:
                (length,) = struct.unpack_from("<I", data, offset)
            except struct.error as exc:
                raise SLDError(f"frame {position} {kind.name} layer length runs past the end of the file") from exc
            offset += 4
            if length < 4 or start + length > len(data):
                raise SLDError(f"frame {position} {kind.name} layer length {length} is out of range")
            if kind is not LayerKind.OUTLINE:
                try:
                    if kind in (LayerKind.MAIN, LayerKind.SHADOW):
                        x1, y1, x2, y2, flag0, flag1 = struct.unpack_from("<4H2B", data, offset)
                        offset += 10
                        box = (x1, y1, x2 - x1, y2 - y1)
                        if kind is LayerKind.MAIN:
                            main_box = box
                    else:
                        flag0, flag1 = struct.unpack_from("<2B", data, offset)
                        offset += 2
                        box = main_box if main_box is not None else (0, 0, 0, 0)
                    (command_count,) = struct.unpack_from("<H", data, offset)
                except struct.error as exc:
                    raise SLDError(f"frame {position} {kind.name} layer header runs past the end of the file") from exc
                offset += 2
                if box[2] < 0 or box[3] < 0:
                    raise SLDError(f"frame {position} {kind.name} layer has a negative bounding box {box}")
                block_offset = offset + 2 * command_count
                if block_offset > len(data):
                    raise SLDError(f"frame {position} {kind.name} command array runs past the end of the file")
                layers.append((kind, *box, flag0, flag1, frame_index, command_count, offset, block_offset, chains[kind]))
                chains[kind] += 1
            end = start + length
            offset = end + (layout_tag - end) % 4
        frames.append((width, height, hotspot_x, hotspot_y, frame_type, frame_index, tuple(layers)))
    return frames


def _layer_fields(layer) -> tuple:
    return (
        layer.kind,
        layer.x1,
        layer.y1,
        layer.width,
        layer.height,
        layer.flag0,
        layer.flag1,
        layer.frame_index,
        layer.command_count,
        layer.command_offset,
        layer.block_offset,
        layer.chain_pos,
    )


def decoder_walk(data: bytes) -> list[tuple]:
    """The decoder's walk in reference_walk()'s shape, read back through the
    public SLDFrame/SLDLayer views."""
    sld = SLDFile(data)
    for frame in sld.frames:
        for layer in frame.layers:
            assert type(layer.kind) is LayerKind
    return [
        (f.width, f.height, f.hotspot_x, f.hotspot_y, f.frame_type, f.frame_index, tuple(map(_layer_fields, f.layers)))
        for f in sld.frames
    ]


def _outcome(walk, data: bytes):
    try:
        return "walked", walk(data)
    except SLDError as exc:
        return "refused", str(exc)


def _mixed_frames() -> list[Frame]:
    """Every shape the walk branches on: all five kinds with slack, masks with
    no MAIN (a zero box), an empty frame, frame_type high bits the layer loop
    must ignore, a box off the origin, a delta, a lone OUTLINE."""
    return [
        Frame(
            [
                Layer(LayerKind.MAIN, box=(4, 8, 12, 16), blocks=[solid_bc1()] * 4, length_slack=3),
                Layer(LayerKind.SHADOW, box=(0, 4, 16, 12), blocks=[solid_bc4()] * 8, length_slack=1),
                Layer(LayerKind.OUTLINE, length_slack=6),
                Layer(LayerKind.DAMAGE, flag0=0x02, blocks=[solid_bc1(0x07E0)] * 4, length_slack=2),
                Layer(LayerKind.PLAYERCOLOR, flag1=0x09, blocks=[solid_bc4(64)] * 4),
            ],
            canvas=(24, 24),
            hotspot=(12, 20),
        ),
        Frame([Layer(LayerKind.DAMAGE, blocks=[]), Layer(LayerKind.PLAYERCOLOR, commands=[], blocks=[])]),
        Frame([], frame_type=0),
        Frame([Layer(LayerKind.MAIN, box=(8, 0, 12, 4), length_slack=1)], frame_type=0xE1, frame_index=9),
        Frame([Layer(LayerKind.SHADOW, box=(0, 0, 8, 4), blocks=[solid_bc4(9)] * 2)]),
        Frame([Layer(LayerKind.MAIN, box=(8, 0, 12, 4), flag0=0x81, commands=[(1, 0)], blocks=[])], frame_index=12),
        Frame([Layer(LayerKind.OUTLINE, length_slack=2)], canvas=(4, 4)),
        Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])], frame_index=0),
    ]


def _walk_fixtures() -> dict[str, bytes]:
    return {
        "mixed_16": build_sld(_mixed_frames()),
        "mixed_14": build_sld(_mixed_frames(), layout_tag=14),
        "every_layer_kind": _every_layer_kind(),
        "delta_chain": _delta_chain(40),
        "variant_14_delta": _released_fixtures()["variant_14_delta"],
    }


def reference_decoder(data: bytes):
    """decode_frame() as it stood over one SLDLayer object per layer, built
    from reference_walk(): per-kind lists of those objects, resolved back to
    the nearest non-delta and decoded forward. It reuses the module's own
    per-layer block decode, which the scalar oracle above pins, so what this
    checks is the chain resolution and every layer field fed into it."""
    frames = reference_walk(data)
    chains = {kind: [] for kind in _LAYER_ORDER}
    frame_layers = []
    for frame in frames:
        layers = tuple(SLDLayer(*fields) for fields in frame[6])
        for layer in layers:
            chains[layer.kind].append(layer)
        frame_layers.append(layers)

    def decode(index: int) -> DecodedFrame:
        width, height, hotspot_x, hotspot_y = frames[index][:4]
        arrays = {}
        for layer in frame_layers[index]:
            if layer.width <= 0 or layer.height <= 0:
                continue
            chain = chains[layer.kind]
            start = layer.chain_pos
            while start > 0 and chain[start].is_delta:
                start -= 1
            blocks = SLDFile._decode_blocks(None, chain[start], None, None, data)
            for position in range(start + 1, layer.chain_pos + 1):
                blocks = SLDFile._decode_blocks(None, chain[position], blocks, chain[position - 1], data)
            image = sld_decoder._blocks_to_image(blocks, layer.width // 4, layer.height // 4)
            arrays[layer.kind] = sld_decoder._paste_on_canvas(image, layer.x1, layer.y1, width, height)
        return DecodedFrame(
            width,
            height,
            hotspot_x,
            hotspot_y,
            main=arrays.get(LayerKind.MAIN),
            shadow=arrays.get(LayerKind.SHADOW),
            damage=arrays.get(LayerKind.DAMAGE),
            playercolor=arrays.get(LayerKind.PLAYERCOLOR),
        )

    return decode


def test_frames_reads_back_as_the_sequence_the_frame_list_was():
    """`frames` used to be a list. It is now a view over the flat index, so the
    list behaviour callers (tools/scan_sprite_reach.py) rely on is pinned."""
    sld = SLDFile(_walk_fixtures()["mixed_16"])
    count = sld.frame_count

    assert len(sld.frames) == count == len(_mixed_frames())
    assert list(sld.frames) == [sld.frames[i] for i in range(count)]
    assert sld.frames[-1] == sld.frames[count - 1]
    assert sld.frames[1:3] == [sld.frames[1], sld.frames[2]]
    with pytest.raises(IndexError):
        sld.frames[count]
    with pytest.raises(IndexError):
        sld.frames[-count - 1]


@pytest.mark.parametrize("name", sorted(_walk_fixtures()))
def test_every_frame_decodes_as_the_object_per_layer_index_did(name):
    """The flat index hands decode_frame() layers rebuilt from its arrays and
    resolves delta chains through arrays of layer ids. Every frame, from held
    and from released bytes, must match the object-per-layer decode."""
    data = _walk_fixtures()[name]
    decode = reference_decoder(data)
    held, released = SLDFile(data), SLDFile(data, keep_data=False)

    drawn = 0
    for index in range(held.frame_count):
        want = decode(index)
        _assert_same_frame(held.decode_frame(index), want)
        _assert_same_frame(released.decode_frame(index, data), want)
        drawn += sum(getattr(want, kind) is not None for kind in ("main", "shadow", "damage", "playercolor"))
    assert drawn >= held.frame_count - 2


@pytest.mark.parametrize("name", sorted(_walk_fixtures()))
def test_the_walk_records_every_field_the_step_by_step_reference_does(name):
    """Every frame header field and every located layer's twelve fields, chain
    positions included, against the reference transcription."""
    data = _walk_fixtures()[name]
    reference = reference_walk(data)

    assert sum(len(frame[6]) for frame in reference) > 1
    assert decoder_walk(data) == reference


# -- decoding into MAIN's box --------------------------------------------


def _main_window(fields: tuple, width: int, height: int) -> tuple[int, int, int, int]:
    """MAIN's box clipped to the canvas, from reference_walk's layer fields, or
    the whole canvas when the frame has no drawable MAIN."""
    for layer in (SLDLayer(*f) for f in fields):
        if layer.kind is LayerKind.MAIN and layer.width > 0 and layer.height > 0:
            return (
                min(layer.x1, width), min(layer.y1, height),
                min(layer.x1 + layer.width, width), min(layer.y1 + layer.height, height),
            )
    return 0, 0, width, height


def _window_fixtures() -> dict[str, bytes]:
    """The walk fixtures, plus MAIN boxes past the right/bottom edge (clipped)
    and starting at or past it (zero-size), each with a SHADOW reaching
    outside MAIN's box so the slice has something to cut."""
    shadow = Layer(LayerKind.SHADOW, box=(0, 0, 8, 8), blocks=[solid_bc4()] * 4)

    def edge(box):
        cols, rows = (box[2] - box[0]) // 4, (box[3] - box[1]) // 4
        return build_sld(
            [Frame([Layer(LayerKind.MAIN, box=box, blocks=[solid_bc1()] * (cols * rows)), shadow,
                    Layer(LayerKind.PLAYERCOLOR, blocks=[solid_bc4(64)] * (cols * rows))], canvas=(8, 8))]
        )

    return {
        **_walk_fixtures(),
        "clipped": edge((4, 4, 12, 12)),
        "off_right": edge((8, 0, 12, 4)),
        "off_bottom": edge((0, 8, 4, 12)),
    }


@pytest.mark.parametrize("name", sorted(_window_fixtures()))
def test_a_windowed_decode_is_the_whole_canvas_decode_sliced_to_mains_box(name):
    """Every frame: each array is the unwindowed one sliced to MAIN's box
    clipped to the canvas, width/height are the window's and the hotspot is
    rebased onto it, so the hotspot still marks the same pixel."""
    data = _window_fixtures()[name]
    reference = reference_walk(data)
    sld = SLDFile(data)

    narrowed = 0
    for index, frame in enumerate(reference):
        width, height, hotspot_x, hotspot_y = frame[:4]
        x0, y0, x1, y1 = _main_window(frame[6], width, height)
        full = sld.decode_frame(index)
        got = sld.decode_frame(index, window_to_main=True)
        assert (got.width, got.height) == (x1 - x0, y1 - y0), index
        assert (got.hotspot_x, got.hotspot_y) == (hotspot_x - x0, hotspot_y - y0), index
        for field in ("main", "shadow", "damage", "playercolor"):
            a, b = getattr(got, field), getattr(full, field)
            assert (a is None) == (b is None), (index, field)
            if b is not None:
                assert a.shape == (y1 - y0, x1 - x0, 4), (index, field)
                assert np.array_equal(a, b[y0:y1, x0:x1]), (index, field)
        narrowed += (x0, y0, x1, y1) != (0, 0, width, height)
    if name in ("mixed_16", "mixed_14", "clipped", "off_right", "off_bottom"):
        assert narrowed, "no frame's window differs from its canvas"


def test_a_main_box_past_the_canvas_edge_windows_to_the_clipped_box_or_to_nothing():
    fixtures = _window_fixtures()
    clipped = SLDFile(fixtures["clipped"]).decode_frame(0, window_to_main=True)
    assert (clipped.width, clipped.height, clipped.hotspot_x, clipped.hotspot_y) == (4, 4, 0, 0)
    assert clipped.main[..., 3].all() and clipped.shadow.shape == (4, 4, 4)
    for name, shape in (("off_right", (4, 0, 4)), ("off_bottom", (0, 4, 4))):
        empty = SLDFile(fixtures[name]).decode_frame(0, window_to_main=True)
        assert empty.main.shape == empty.playercolor.shape == empty.shadow.shape == shape, name
        assert unit_sprites._cropped_to_ink(empty.main, empty.playercolor, empty.hotspot_x, empty.hotspot_y) is None


# Bytes from a layer's start to its command array: the u32 length, then a box
# and two flags (MAIN, SHADOW) or just the flags (DAMAGE, PLAYERCOLOR), then a u16 count.
_GRAPHICS_HEAD_SIZE = 4 + 10 + 2
_MASK_HEAD_SIZE = 4 + 2 + 2

# The six ways a layer or frame can fail the walk, as reference_walk() words them.
_WALK_ERRORS = (
    r"^frame \d+ header runs past the end of the file",
    "layer length runs past the end of the file",
    "is out of range",
    "layer header runs past the end of the file",
    "negative bounding box",
    "command array runs past the end of the file",
)


def walk_variants(data: bytes) -> list[bytes]:
    """Every prefix of `data`, every byte forced to 0x00 and to 0xFF, and each
    layer shortened to a bare 4-byte length right at EOF."""
    variants = [data[:cut] for cut in range(len(data))]
    variants.extend(
        data[:position] + bytes([value]) + data[position + 1 :]
        for position in range(len(data))
        for value in (0x00, 0xFF)
        if data[position] != value
    )
    for frame in reference_walk(data):
        for layer in frame[6]:
            start = layer[9] - (_MASK_HEAD_SIZE if layer[0] in (LayerKind.DAMAGE, LayerKind.PLAYERCOLOR) else _GRAPHICS_HEAD_SIZE)
            variants.extend(data[:start] + struct.pack("<I", 4) + data[start + 4 : start + 4 + tail] for tail in range(12))
    return variants


@pytest.mark.parametrize("name", ["mixed_16", "mixed_14"])
def test_every_truncation_and_single_byte_corruption_fails_as_the_reference_does(name):
    """The flat walk reads a layer's length, sub-header and command count in
    one unpack, so its failure has to be split back into the error the
    step-by-step reads would have raised. Every prefix of the file, and every
    byte forced to 0x00 and to 0xFF, must walk to the same fields or be
    refused with the same message. So must each layer shortened to a bare
    4-byte length right at EOF, the one shape where the length checks out but
    the sub-header after it is missing."""
    data = _walk_fixtures()[name]
    seen = set()
    for variant in walk_variants(data):
        want = _outcome(reference_walk, variant)
        assert _outcome(decoder_walk, variant) == want, variant.hex()
        if want[0] == "refused":
            seen.update(error for error in _WALK_ERRORS if re.search(error, want[1]))
    # Non-vacuity: the sweep reached every error the walk can raise.
    assert seen == set(_WALK_ERRORS)


@pytest.mark.corpus
def test_real_install_files_walk_exactly_as_the_reference_does():
    """The synthetic fixtures cover every branch; this covers the real shapes
    those branches were measured against. A fixed sample, every 5th file by
    name, plus a variant-14 Stable and the trees a big map re-walked most."""
    root = asset_source.get_install_path()
    if root is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    directory = root / unit_sprites.GRAPHICS_SUBPATH
    names = sorted(path.name for path in directory.glob("*.sld"))
    if not names:
        pytest.skip(f"no .sld files under {directory}")
    wanted = {"b_west_stable_age3_x1.sld", "n_tree_oak_x1.sld", "n_tree_autumn_oak_x1.sld", "n_tree_pine_x1.sld"}
    sample = sorted(set(names[::5]) | (wanted & set(names)))

    walked = 0
    for file_name in sample:
        data = (directory / file_name).read_bytes()
        want = _outcome(reference_walk, data)
        assert _outcome(decoder_walk, data) == want, file_name
        walked += want[0] == "walked"
    assert walked > 0.9 * len(sample)


@pytest.mark.corpus
def test_real_install_frames_decode_as_the_object_per_layer_index_did():
    """Real delta chains, including the deep tree ones, through the flat index
    against the object-per-layer decode: first, middle and last frame of every
    50th file by name, plus the same named files as above."""
    root = asset_source.get_install_path()
    if root is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    directory = root / unit_sprites.GRAPHICS_SUBPATH
    names = sorted(path.name for path in directory.glob("*.sld"))
    if not names:
        pytest.skip(f"no .sld files under {directory}")
    wanted = {"b_west_stable_age3_x1.sld", "n_tree_oak_x1.sld", "n_tree_autumn_oak_x1.sld", "n_tree_pine_x1.sld"}
    sample = sorted(set(names[::50]) | (wanted & set(names)))

    decoded = 0
    for file_name in sample:
        data = (directory / file_name).read_bytes()
        sld = load_sld(directory / file_name, keep_data=False)
        if sld is None or sld.frame_count == 0:
            continue
        decode = reference_decoder(data)
        for index in sorted({0, sld.frame_count // 2, sld.frame_count - 1}):
            try:
                want = decode(index)
            except SLDError as exc:
                with pytest.raises(SLDError, match=re.escape(str(exc))):
                    sld.decode_frame(index, data)
                continue
            _assert_same_frame(sld.decode_frame(index, data), want)
            decoded += want.main is not None
    assert decoded > len(sample)
