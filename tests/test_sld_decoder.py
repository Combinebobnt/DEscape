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

import struct

import numpy as np
import pytest

from descape.sld_decoder import (
    MAGIC,
    DecodedFrame,
    LayerKind,
    SLDError,
    SLDFile,
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
    indices = [7, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 7]
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
    for _ in range(depth):
        frames.append(
            Frame(
                [Layer(kind, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])],
                canvas=(4, 4),
            )
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
    for _ in range(3):
        frames.append(
            Frame(
                [Layer(LayerKind.SHADOW, box=(0, 0, 4, 4), commands=[(0, 1)], blocks=[solid_bc4(30)])],
                canvas=(4, 4),
            )
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
