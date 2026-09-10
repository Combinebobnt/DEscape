"""Verifies descape.unit_sprites -- resolution, fallbacks, tint and anchoring.

**Why the .sld bytes here are hand-built and minimal.** tests/test_sld_decoder.py
already has a much richer synthetic-file builder, but it exists to drive the
format's edge cases; this module only ever needs one valid file, and keeping
its fixture independent is deliberate -- a test that shares its input builder
with the code path it is checking can pass for the wrong reason. No game assets
are involved either way, matching the suite's standing posture (conftest points
CONFIG_PATH at a missing file, so no test ever sees a real install).

The rules pinned here were settled by measurement against the real install
rather than read off a spec, so each one records what would break:

- **Angle-major indexing.** frames[angle * frame_count]. Consecutive frames are
  animation steps of ONE facing, not adjacent angles, so an off-by-frame_count
  error yields a plausible-looking sprite of the wrong unit pose.
- **mirroring_mode is ignored.** The usual AoE2 convention stores half the
  angles and mirrors the rest; these files do not.
- **Player colour is a multiply masked by coverage**, never a palette lookup.
  The 256-entry playercolor_*.pal files have all-black entries above 127 while
  MAIN's luminance under the mask reaches 247, so a lookup blacks out most
  sprites -- a defect that is invisible on the minority of sprites dark enough
  to survive it.
- **The hotspot anchors at the FOOTPRINT'S centre**, which differs from the
  anchor tile's diamond centre by half a tile on an even span.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from descape import asset_source, unit_sprites

MAGIC = b"SLDX"
_HEADER = struct.Struct("<4s4HI")
_FRAME_HEADER = struct.Struct("<4H2BH")

MAIN = 0x01
PLAYERCOLOR = 0x10

CANVAS = 8
HOTSPOT = 4
FILE_NAME = "t_synthetic_x1"
CONST = 4242


def bc1_solid(rgb565: int = 0xF800) -> bytes:
    """One 4x4 block, every texel endpoint 0. rgb565 > 0 keeps c0 > c1, which
    is the opaque four-colour mode."""
    return struct.pack("<HH", rgb565, 0) + bytes(4)


def bc4_solid(value: int = 255) -> bytes:
    """One 4x4 mask block at full coverage: a0 > a1 is the wide mode, and every
    index 0 selects a0."""
    return struct.pack("<BB", value, 0) + bytes(6)


def _layer(kind: int, blocks: list[bytes], box) -> bytes:
    if kind == MAIN:
        body = struct.pack("<4H2B", *box, 0, 1)
    else:
        body = struct.pack("<2B", 0, 1)
    # A command's draw count is a single byte, so a block grid wider than 255
    # blocks needs several back-to-back draws rather than one -- real files do
    # the same thing.
    runs = [min(255, len(blocks) - i) for i in range(0, len(blocks), 255)]
    body += struct.pack("<H", len(runs))
    body += b"".join(bytes((0, n)) for n in runs)
    body += b"".join(blocks)
    return struct.pack("<I", 4 + len(body)) + body


def build_sld(frame_count: int, *, playercolor: bool = True, layout_tag: int = 16,
              magic: bytes = MAGIC, canvas: int = CANVAS) -> bytes:
    """A file of `frame_count` identical frames, each a solid MAIN block grid
    plus (optionally) a full-coverage PLAYERCOLOR mask. The hotspot sits at the
    canvas centre, as it does on every real frame."""
    n_blocks = (canvas // 4) ** 2
    hotspot = canvas // 2
    out = bytearray(_HEADER.pack(magic, 4, frame_count, 0, layout_tag, 0))
    for i in range(frame_count):
        ftype = MAIN | (PLAYERCOLOR if playercolor else 0)
        out += _FRAME_HEADER.pack(canvas, canvas, hotspot, hotspot, ftype, 0, i)
        # Frame index varies the colour so a wrong frame is detectable.
        out += _layer(MAIN, [bc1_solid(0x0800 * (i % 30 + 1))] * n_blocks, (0, 0, canvas, canvas))
        out += bytes([0xAA]) * ((4 - len(out)) % 4)
        if playercolor:
            out += _layer(PLAYERCOLOR, [bc4_solid()] * n_blocks, (0, 0, canvas, canvas))
            out += bytes([0xAA]) * ((4 - len(out)) % 4)
    return bytes(out)


@pytest.fixture
def install(tmp_path, monkeypatch):
    """A tmp directory shaped like an AoE2:DE install, with one .sld in it."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)

    def write(data: bytes, name: str = FILE_NAME) -> None:
        (graphics / f"{name}.sld").write_bytes(data)

    def register(*, angle_count=4, frame_count=3, const=CONST, file_name=FILE_NAME,
                 rotation_is_variant=False) -> None:
        entry = {"graphic_id": 1, "file_name": file_name,
                 "angle_count": angle_count, "mirroring_mode": 6,
                 "frame_count": frame_count}
        if rotation_is_variant:
            entry["rotation_is_variant"] = True
        monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {const: entry})

    def register_composite(const: int, pieces: list[dict]) -> None:
        """`pieces` is a list of unit_graphic_map.json-shaped piece dicts
        (unit_id/file_name/angle_count/frame_count/dx/dy). The first is
        emitted as `const`'s own five fields too, matching the generator's
        own contract that the parent's own piece IS the entry's own graphic."""
        parent = pieces[0]
        monkeypatch.setattr(
            unit_sprites, "graphic_map",
            lambda: {const: {
                "graphic_id": 1, "file_name": parent["file_name"],
                "angle_count": parent["angle_count"], "mirroring_mode": 6,
                "frame_count": parent["frame_count"], "pieces": pieces,
            }},
        )

    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield type("Install", (), {"write": staticmethod(write),
                               "register": staticmethod(register),
                               "register_composite": staticmethod(register_composite),
                               "graphics": graphics})
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# --- angle resolution -------------------------------------------------


@pytest.mark.parametrize(
    ("rotation", "angle_count", "expected"),
    [
        # All the angle_count=16 rows carry ANGLE_ZERO_OFFSET_DEG's -45 degrees
        # as -2 steps -- these were 0/4/8/0/2/12/3 before that offset was
        # measured against the running game on 2026-08-24.
        (0.0, 16, 14),
        (math.pi / 2, 16, 2),
        (math.pi, 16, 6),
        (2 * math.pi, 16, 14),  # a full turn wraps back to where 0 lands
        (7.0, 16, 0),  # outside [0, 2pi) -- 574 real corpus units carry 7.0
        (-math.pi / 2, 16, 10),  # negative wraps too
        (1.0, 16, 1),  # a hand-typed value off the grid still rounds
        (0.0, 1, 0),
        (5.0, 1, 0),  # a single-angle graphic ignores rotation entirely
    ],
)
def test_angle_index_rounds_and_wraps(rotation, angle_count, expected):
    assert unit_sprites.angle_index(rotation, angle_count) == expected


@pytest.mark.parametrize("angle_count", [2, 3, 4, 5, 6, 25])
def test_the_zero_offset_is_skipped_when_it_is_not_a_whole_stored_frame(angle_count):
    """Walls. They store 5 angles, so a 45-degree offset is 0.625 of a frame
    and cannot select a stored one -- applying it re-quantized every facing to
    a neighbour and rotated every wall by a full 72-degree frame, which is
    exactly what a live window showed on 2026-08-24 once units were fixed.

    So the offset is skipped unless it lands on a whole frame, and these
    graphics must behave EXACTLY as they did before ANGLE_ZERO_OFFSET_DEG
    existed. Pinned against a local re-implementation of the old expression
    rather than against hardcoded indices: the point is byte-for-byte
    preservation of the old mapping, and a hardcoded table would not say that.

    The rotations are real values measured from the example corpus, not
    invented: stone walls carry integers 0..4 AND exact k*2pi/5 radians.
    """
    def old_mapping(rotation: float) -> int:
        return round(rotation / (2 * math.pi) * angle_count) % angle_count

    for rotation in (0.0, 1.0, 2.0, 3.0, 4.0, 1.2566, 2.5133, 3.7699, 5.0265):
        assert unit_sprites.angle_index(rotation, angle_count) == old_mapping(rotation), (
            f"angle_count={angle_count} rotation={rotation} moved. The zero offset is "
            f"{unit_sprites.ANGLE_ZERO_OFFSET_DEG / (360.0 / angle_count):.3f} frames here, "
            f"not a whole one, so it must not be applied at all"
        )


@pytest.mark.parametrize("angle_count", [8, 16, 32, 72])
def test_the_zero_offset_is_the_same_ANGLE_whatever_angle_count_is(angle_count):
    """The property the old index-space constant could not express, and the
    reason it was wrong rather than merely differently-written.

    `ANGLE_ZERO_INDEX = 0` was a fixed INDEX offset. Index 0 is a fixed compass
    direction for every graphic, so correcting the zero point by a real-world
    45 degrees is 1 step at angle_count 8 but 2 at 16 and 4 at 32 -- and
    angle_count is nowhere near uniform in the real data (16 dominates at 1,163
    graphics, 8 is rare at 76). Any index-space constant therefore mis-rotates
    every graphic that does not share the angle_count it was tuned on.

    Chosen angle_counts are all divisors of 360/ANGLE_ZERO_OFFSET_DEG's step so
    the expected index is exact, which keeps this an equality rather than a
    tolerance that could absorb the very error it is checking.
    """
    step_deg = 360.0 / angle_count
    offset_steps = unit_sprites.ANGLE_ZERO_OFFSET_DEG / step_deg
    assert offset_steps == int(offset_steps), "pick angle_counts that divide the offset exactly"

    # Compared modulo a full turn, because the offset is signed and the index
    # is not: at -45 degrees, angle_count 16 wraps to index 14, which IS -45
    # degrees, just expressed as +315.
    index = unit_sprites.angle_index(0.0, angle_count)
    assert (index * step_deg) % 360.0 == unit_sprites.ANGLE_ZERO_OFFSET_DEG % 360.0, (
        f"rotation 0 lands on index {index} of {angle_count}, i.e. "
        f"{(index * step_deg) % 360.0} degrees, but the offset is "
        f"{unit_sprites.ANGLE_ZERO_OFFSET_DEG % 360.0} degrees. The offset must be angular, "
        f"not a fixed number of index steps"
    )


# --- Flat's own zero point (FLAT_ANGLE_ZERO_OFFSET_DEG) ---------------


def _angle_entry(angle_count: int) -> dict:
    """A graphic_map()-shaped record for a plain facing graphic. CONST is not
    in the committed unit_graphic_map.json, so rotation_is_variant() reads
    False for it and _frame_for() takes the angle branch."""
    return {"graphic_id": 1, "file_name": FILE_NAME, "angle_count": angle_count,
            "mirroring_mode": 6, "frame_count": 1}


def test_the_flat_offset_is_the_iso_offset_plus_the_grid_rotation():
    """Not a hardcoded 0.0. iso_geometry projects world +x to screen up-right
    and +y to down-right, so the iso screen is the world rotated -45 degrees
    while Flat's top-down grid puts +x at 0 and +y at +90. Flat is therefore
    iso plus a 45-degree apparent facing, which is the ONE relationship worth
    pinning: re-measure either constant against the game and this says whether
    the pair still describes the same two projections."""
    delta = (unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG - unit_sprites.ANGLE_ZERO_OFFSET_DEG) % 360.0
    assert delta == 45.0, (
        f"the two zero points differ by {delta} degrees, not the 45 the iso grid "
        f"rotation accounts for"
    )


@pytest.mark.parametrize("angle_count", [8, 16, 32, 72])
def test_the_flat_offset_moves_the_frame_by_an_eighth_of_a_turn(angle_count):
    """45 degrees expressed in THIS graphic's own steps, which is the property
    an index-space constant could not carry (see the iso sibling above). Only
    angle_counts that are multiples of 8 land the offset on a whole stored
    step, so only these can show the shift at all."""
    entry = _angle_entry(angle_count)
    assert not unit_sprites.rotation_is_variant(CONST), "this must take the angle branch"
    iso = unit_sprites._frame_for(CONST, entry, 0.0)
    flat = unit_sprites._frame_for(CONST, entry, 0.0, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
    assert (flat - iso) % angle_count == angle_count // 8, (
        f"angle_count={angle_count}: iso resolves frame {iso} and flat frame {flat}, "
        f"a shift of {(flat - iso) % angle_count} steps rather than the "
        f"{angle_count // 8} that 45 degrees is here"
    )


@pytest.mark.parametrize("angle_count", [3, 5, 25])
def test_the_two_offsets_agree_where_neither_lands_on_a_stored_step(angle_count):
    """And the agreement is a COINCIDENCE of the two values, not an invariant.
    angle_index() skips an offset that does not land on a whole stored step,
    so the iso -45 is discarded here; Flat's offset is genuinely zero and so
    changes nothing. Two different reasons reaching the same index. Re-measure
    FLAT_ANGLE_ZERO_OFFSET_DEG to something non-zero and this stops holding,
    which is why the failure message says why rather than just what."""
    entry = _angle_entry(angle_count)
    iso = unit_sprites._frame_for(CONST, entry, 1.0)
    flat = unit_sprites._frame_for(CONST, entry, 1.0, unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
    assert iso == flat, (
        f"angle_count={angle_count} is not a multiple of 8, so angle_index() drops the "
        f"iso offset entirely and Flat's own offset is "
        f"{unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG}. They agree only while that is 0.0; "
        f"a re-measured flat offset belongs in this test's expectation, not in a skip"
    )


@pytest.mark.parametrize("unit_const", [117, 264])
def test_the_flat_offset_never_reaches_the_variant_path(unit_const):
    """A real wall const and a real cliff const. By construction, not by luck:
    a variant index is a SHAPE, not a facing, so no camera zero point applies
    to it, and threading the offset as a parameter is what keeps that true.
    Pre-rotating `rotation` at icon_for()'s door instead would land a mutated
    value in variant_index(), which AGENTS.md's hard rules forbid.

    A third, deliberately absurd offset of exactly one stored step is probed
    alongside the two real ones, and that is what keeps this non-vacuous:
    both real offsets happen to agree on the angle branch too at these
    angle_counts (5 and 25, neither a multiple of 8), so comparing only those
    would pass without the variant branch ignoring anything.
    """
    entry = unit_sprites.graphic_map()[unit_const]
    angle_count = int(entry["angle_count"])
    assert unit_sprites.rotation_is_variant(unit_const), "this must take the variant branch"

    one_step = 360.0 / angle_count
    assert unit_sprites.angle_index(0.0, angle_count, one_step) != unit_sprites.angle_index(
        0.0, angle_count
    ), "the probe offset must move the ANGLE branch, or this proves nothing"

    for rotation in (0.0, 1.0, 2.0, 3.0, 4.0, 7.0):
        iso = unit_sprites._frame_for(unit_const, entry, rotation)
        for offset in (unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG, one_step):
            got = unit_sprites._frame_for(unit_const, entry, rotation, offset)
            assert got == iso, (
                f"const {unit_const} rotation {rotation}: offset {offset} moved the "
                f"variant frame from {iso} to {got}"
            )


# --- variant resolution: walls, whose frames are shapes not facings ----


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        # The two the bug report and the frame decode pin as currently wrong:
        # angle_index() gives 2 and 3 here, so an isolated post drew the tall
        # tower and then a wall run instead of the narrow column.
        (3.0, 3),
        (4.0, 4),
        # Right by luck under the radians reading, pinned so an edit cannot
        # flip them silently.
        (0.0, 0),
        (1.0, 1),
        (2.0, 2),
        # The other convention real files ship for the SAME graphic: the index
        # re-encoded as k*2pi/5 radians. 2.513274 is the one a naive
        # round-the-raw-value would misfile as 3 -- it sits 0.4867 from 3 and
        # 0.5133 from 2.
        (1.256637, 1),
        (2.513274, 2),
        (3.769911, 3),
        (5.026548, 4),
    ],
)
def test_variant_index_reads_both_stored_conventions(rotation, expected):
    assert unit_sprites.variant_index(rotation, 5) == expected


@pytest.mark.parametrize("rotation", [5.969026, 3.298672, 4.084071, 5.0, -1.5])
def test_variant_index_is_in_range_for_hand_rotated_outliers(rotation):
    """The three measured real-world outliers (one 5.969026 among 1361
    Palisade Wall placements; one each of 3.298672 and 4.084071 among 4437
    Stone Wall placements) plus two synthetic edges: 5.0, which is an integer
    but OUT of [0, 5) so it must fall to the angle branch, and a negative.

    Only in-range determinism is asserted, not a specific frame. There is no
    ground truth for a freely-rotated wall piece -- the premise is that the
    frames are discrete shapes, not a continuum -- so pinning a "correct"
    answer here would be inventing one.
    """
    got = unit_sprites.variant_index(rotation, 5)
    assert got in range(5)
    assert got == unit_sprites.variant_index(rotation, 5)


@pytest.mark.parametrize("rotation", [0.0, 3.0, 7.0, -2.5])
def test_variant_index_ignores_rotation_for_a_single_frame_graphic(rotation):
    assert unit_sprites.variant_index(rotation, 1) == 0


def test_the_zero_offset_never_reaches_the_variant_path():
    """By construction, not because it is skipped at angle_count 5.

    ANGLE_ZERO_OFFSET_DEG corrects a camera-relative facing; a variant-index
    graphic has no facing to correct. Checked at angle_count 8, where the
    offset IS a whole stored step and angle_index() therefore applies it, so
    the two functions provably disagree -- at angle_count 5 they would agree
    for the wrong reason and this test would pass vacuously.

    The expected angle_index is derived from the constant rather than
    hardcoded, so re-measuring the offset against the game does not redden a
    test about something else.
    """
    offset_steps = unit_sprites.ANGLE_ZERO_OFFSET_DEG / (360.0 / 8)
    assert offset_steps == int(offset_steps), "pick an angle_count the offset divides exactly"
    assert unit_sprites.angle_index(0.0, 8) == int(offset_steps) % 8
    assert unit_sprites.angle_index(0.0, 8) != 0, "otherwise the disagreement below is vacuous"
    assert unit_sprites.variant_index(0.0, 8) == 0


def test_the_real_wall_consts_are_gated_onto_the_variant_path():
    """The gate itself, not variant_index(). 117 is the direct-decode case;
    the rest come from the plan's corpus profiling. Aqueduct (231) and Granary
    (1089) are generated-VARIANT too (2026-09-06 Tier B plan settled the TODO
    item that had left them unresolved -- thin corpus evidence for either
    reading on their own, but they share the same unit.type != 70 discriminator
    as the confirmed wall family). 4 and 74 are ordinary creatable units and
    must stay on angle_index()."""
    for const in (72, 117, 119, 155, 370, 788, 1062, 2678, 231, 1089):
        assert unit_sprites.rotation_is_variant(const), const
    for const in (4, 74):
        assert not unit_sprites.rotation_is_variant(const), const


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (0, None),  # isolated -- no derivable answer, caller must fall through
        (unit_sprites.WEST | unit_sprites.EAST, 0),
        (unit_sprites.NORTH | unit_sprites.SOUTH, 1),
        (unit_sprites.WEST, 2),
        (unit_sprites.NORTH, 2),
        (unit_sprites.WEST | unit_sprites.NORTH, 2),
        (unit_sprites.WEST | unit_sprites.EAST | unit_sprites.NORTH, 2),
        (unit_sprites.WEST | unit_sprites.EAST | unit_sprites.NORTH | unit_sprites.SOUTH, 2),
    ],
)
def test_wall_variant_from_neighbours(mask, expected):
    assert unit_sprites.wall_variant_from_neighbours(mask) == expected


@pytest.mark.parametrize(
    ("rotation", "angle_count", "expected"),
    [
        (0.0, 5, True),
        (4.0, 5, True),
        (3.9999999, 5, True),  # inside the 1e-6 window
        (5.0, 5, False),  # an integer, but out of [0, angle_count)
        (1.256637, 5, False),  # index 1's radian encoding
        (2.513274, 5, False),  # the value closest to a DIFFERENT integer --
        # variant_index()'s own hard case, checked here too since a wrong
        # classification would misfile precondition 3.
    ],
)
def test_is_literal_variant_index(rotation, angle_count, expected):
    assert unit_sprites.is_literal_variant_index(rotation, angle_count) == expected


def test_rotation_variant_eligible_requires_variant_const_and_five_angles(install):
    """Preconditions 1-2 only -- rotation itself plays no part here."""
    install.register(const=117, angle_count=5)
    assert unit_sprites.rotation_variant_eligible(117)

    install.register(const=117, angle_count=4)
    assert not unit_sprites.rotation_variant_eligible(117), "angle_count != 5 must not transfer the mask table"

    install.register(const=4, angle_count=5)
    assert not unit_sprites.rotation_variant_eligible(4), "an ordinary (non-variant) const stays excluded"


def test_a_gated_const_resolves_a_different_frame_than_an_ungated_one(install):
    """The wiring in sprite_for(), which neither the pure-function tests nor
    the gate test above can reach.

    Rotation 3.0 at angle_count 5 is the reported bug: angle_index() collapses
    it onto frame 2, the same frame rotation 2.0 selects, so a lone post drew
    the tall tower. Asserted through real decoded frame colours rather than
    through either function's return value, so the test measures what actually
    got drawn. GAIA (player 0) keeps the tint identity so the frame's own
    colour survives.

    Non-vacuity: emptying _ROTATION_VARIANT_CONSTS turns this red, verified by
    mutation rather than assumed.
    """
    install.write(build_sld(15))
    install.register(angle_count=5, frame_count=3, const=117, rotation_is_variant=True)

    def colour(rotation):
        got = unit_sprites.sprite_for(117, rotation, 0, 32)
        assert got is not None
        return tuple(int(v) for v in got.rgba[0, 0, :3])

    assert colour(3.0) != colour(2.0)

    def colour_of(frame_index):
        native = unit_sprites._native_frame(FILE_NAME, frame_index)
        return tuple(int(v) for v in native[0][0, 0, :3])

    # Exact frames, not just "they differ": frame_count=3 makes the stride
    # visible, so a variant-3 read lands on 9 where the old angle read landed
    # on 6.
    assert colour(3.0) == colour_of(3 * 3)
    assert colour(2.0) == colour_of(2 * 3)


def test_frame_index_is_angle_major(install):
    """frames[angle * frame_count], not frames[angle]. Consecutive frames are
    animation steps of one facing, so the two differ by a whole pose."""
    install.write(build_sld(12))
    install.register(angle_count=4, frame_count=3)
    quarter = 2 * math.pi / 4
    # GAIA, so the tint is identity and the frame's own colour survives to be
    # compared -- a real player's multiply would flatten these synthetic
    # single-channel colours to black.
    got = [unit_sprites.sprite_for(CONST, quarter * a, 0, 32) for a in range(4)]
    assert all(g is not None for g in got)
    # Frame colours vary per frame index; angle-major picks 0, 3, 6, 9, so four
    # distinct colours. Frame-major would pick 0, 1, 2, 3 -- also four, but the
    # wrong ones, which is why the exact frames are checked below rather than
    # just their count.
    colours = [tuple(int(v) for v in g.rgba[0, 0, :3]) for g in got]
    assert len(set(colours)) == 4, colours

    def colour_of(frame_index):
        native = unit_sprites._native_frame(FILE_NAME, frame_index)
        return tuple(int(v) for v in native[0][0, 0, :3])

    # Resolved through angle_index rather than assuming angle a lands on stored
    # index a: this test's subject is the STRIDE (frames[index * frame_count]),
    # and hardcoding the identity mapping would make it fail whenever
    # ANGLE_ZERO_OFFSET_DEG changes -- which says nothing about the stride.
    expected = [unit_sprites.angle_index(quarter * a, 4) for a in range(4)]
    assert sorted(expected) == [0, 1, 2, 3], f"the four facings must stay distinct, got {expected}"
    assert colours == [colour_of(i * 3) for i in expected]


def test_mirroring_mode_does_not_halve_the_stored_angles(install):
    """All angle_count angles are stored; mirroring_mode is not consulted.
    Measured over 400 real graphics: total_frames - angle_count*frame_count is
    0 or 1, never half."""
    install.write(build_sld(12))
    install.register(angle_count=4, frame_count=3)
    last = unit_sprites.sprite_for(CONST, 2 * math.pi * 3 / 4, 1, 32)
    assert last is not None  # angle 3 exists, i.e. frame 9 was reachable


# --- fallbacks: every failure returns None, never raises ---------------


def test_unknown_unit_const_falls_back(install):
    install.write(build_sld(4))
    install.register()
    assert unit_sprites.sprite_for(CONST + 1, 0.0, 1, 32) is None


def test_no_install_configured_falls_back(install):
    install.write(build_sld(4))
    install.register()
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


def test_missing_file_falls_back(install):
    install.register(file_name="t_absent_x1")
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


def test_bad_magic_falls_back(install):
    install.write(build_sld(4, magic=b"NOPE"))
    install.register()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


def test_zero_byte_file_falls_back(install):
    """One real file in the install is zero bytes long."""
    install.write(b"")
    install.register()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


def test_the_fourteen_layout_variant_falls_back(install):
    """226 real files use it and no known decoder reads them."""
    install.write(build_sld(4, layout_tag=14))
    install.register()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


def test_a_frame_index_past_the_files_real_end_falls_back(install):
    """angle_count * frame_count overruns the true frame count on a minority of
    real graphics, so the FILE's count is what gets trusted, not the .dat's
    arithmetic. Without the guard this is a wrong frame or a crash."""
    install.write(build_sld(4))  # only 4 frames on disk
    install.register(angle_count=4, frame_count=3)  # claims 12
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is not None  # angle 0 -> 0
    assert unit_sprites.sprite_for(CONST, math.pi, 1, 32) is None  # angle 2 -> 6


def test_a_frame_that_fails_to_decode_falls_back(install, monkeypatch):
    """decode_frame() RAISES SLDError; only load_sld() is the never-raises
    half. An uncaught one would crash the whole render."""
    install.write(build_sld(4))
    install.register()

    def boom(self, index):
        raise unit_sprites.SLDError("synthetic")

    monkeypatch.setattr("descape.sld_decoder.SLDFile.decode_frame", boom)
    unit_sprites.clear_caches()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None


# --- player colour ----------------------------------------------------


def test_gaia_is_the_untinted_sprite(install):
    """TEAM_COLORS[0] is (255, 255, 255), so GAIA multiplies to identity and
    needs no special case anywhere in the render path."""
    install.write(build_sld(4))
    # angle_count=1 so rotation is ignored and frame 0 is the one resolved,
    # whatever ANGLE_ZERO_OFFSET_DEG happens to be. This test is about the
    # TINT; leaving the angle mapping in the loop just made it fail whenever
    # the zero offset moved, which says nothing about tinting.
    install.register(angle_count=1, frame_count=1)
    gaia = unit_sprites.sprite_for(CONST, 0.0, 0, 32)
    plain = unit_sprites._tinted(
        unit_sprites._native_frame(FILE_NAME, 0)[0], None, (255, 255, 255)
    )
    assert np.array_equal(gaia.rgba[..., :3], unit_sprites._resize_rgba(plain, *gaia.rgba.shape[1::-1])[..., :3])


def test_the_tint_is_a_multiply_and_only_inside_the_mask():
    main = np.zeros((4, 4, 4), np.uint8)
    main[..., :3] = 200
    main[..., 3] = 255
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[:2, :, 0] = 255  # full coverage on the top half only
    mask[:2, :, 3] = 255

    red = unit_sprites._tinted(main, mask, (255, 0, 0))
    assert tuple(red[0, 0, :3]) == (200, 0, 0)  # 200 * (1, 0, 0)
    assert tuple(red[3, 0, :3]) == (200, 200, 200)  # outside the mask, untouched

    blue = unit_sprites._tinted(main, mask, (0, 0, 255))
    differ = np.any(red[..., :3] != blue[..., :3], axis=-1)
    assert differ[:2, :].all() and not differ[2:, :].any()


def test_partial_coverage_blends_rather_than_switching():
    main = np.zeros((1, 1, 4), np.uint8)
    main[..., :3] = 200
    mask = np.zeros((1, 1, 4), np.uint8)
    mask[..., 0] = 128
    mask[..., 3] = 255
    out = unit_sprites._tinted(main, mask, (255, 0, 0))
    assert out[0, 0, 0] == 200
    assert 90 < out[0, 0, 1] < 110  # halfway between 200 and 0


# --- anchoring and scale ----------------------------------------------


def test_scale_follows_the_projection():
    assert unit_sprites.sprite_scale(unit_sprites.NATIVE_TILE_W // 2) == 1.0
    assert unit_sprites.sprite_scale(32) == 64 / unit_sprites.NATIVE_TILE_W


@pytest.mark.parametrize(
    ("x0", "y0", "span_x", "span_y", "expected"),
    [
        (5, 7, 1, 1, (5.5, 7.5)),  # a 1x1 anchors at its own tile's centre
        (4, 4, 2, 2, (5.0, 5.0)),  # an even span anchors on a tile CORNER
        (4, 4, 3, 3, (5.5, 5.5)),
        (2, 6, 4, 1, (4.0, 6.5)),  # a gate strip: per-axis, not square
    ],
)
def test_anchor_is_the_footprint_centre(x0, y0, span_x, span_y, expected):
    assert unit_sprites.anchor_tile_coords(x0, y0, span_x, span_y) == expected


def test_anchor_tile_is_the_footprints_depth_latest(unit_span=None):
    """Maximal d = y - x, tie-broken on ascending x -- the same key
    composite_rect_iso's own np.lexsort((xs, ys - xs)) uses. An ad-hoc
    max(x + y) would diverge, and the ID-plane pick oracle cannot catch that:
    picking still goes through the diamond path, untouched by sprites."""
    for x0, y0, sx, sy in [(3, 3, 1, 1), (3, 3, 2, 2), (10, 2, 4, 4), (0, 0, 4, 1)]:
        tiles = [(tx, ty) for ty in range(y0, y0 + sy) for tx in range(x0, x0 + sx)]
        got = unit_sprites.sprite_anchor_tile(tiles)
        expected = max(tiles, key=lambda t: (t[1] - t[0], t[0]))
        assert got == expected


def test_anchor_tile_takes_the_occupied_set_not_the_bbox():
    """A diagonal gate's sprite anchor must resolve from its real sparse
    footprint, not its 4x4 bbox -- Part 2's reported wrong-depth-position
    defect. The two real shapes (tools/gen_unit_render_data.py's
    building_tiles docs; local offsets, own tile always at (2, 2)):

       e gates                    n gates
          #...                       ...#
          .##.                       .##.
          .##.                       .##.
          ...#                       #...

    For the "e" shape, the bbox's own depth-latest tile (local (0, 3), the
    bbox corner opposite the own-tile pillar) is not in the occupied set at
    all -- the sparse-aware anchor must land on a real pillar instead."""
    x0, y0 = 10, 2
    bbox_local = [(ox, oy) for ox in range(4) for oy in range(4)]
    bbox_tiles = [(x0 + ox, y0 + oy) for ox, oy in bbox_local]
    e_local = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
    n_local = [(0, 3), (1, 1), (1, 2), (2, 1), (2, 2), (3, 0)]

    bbox_winner = unit_sprites.sprite_anchor_tile(bbox_tiles)
    assert bbox_winner == (x0 + 0, y0 + 3)
    assert (bbox_winner[0] - x0, bbox_winner[1] - y0) not in e_local

    e_tiles = [(x0 + ox, y0 + oy) for ox, oy in e_local]
    assert unit_sprites.sprite_anchor_tile(e_tiles) == (x0 + 1, y0 + 2)

    n_tiles = [(x0 + ox, y0 + oy) for ox, oy in n_local]
    assert unit_sprites.sprite_anchor_tile(n_tiles) == (x0 + 0, y0 + 3)


def test_the_hotspot_scales_with_the_sprite(install):
    # A 96px canvas, not this module's default 8px one: the hotspot is a
    # rounded fraction of the size, so on a tiny canvas a single rounding step
    # is an eighth of the sprite and swamps what is being checked.
    install.write(build_sld(4, canvas=96))
    install.register()
    at32 = unit_sprites.sprite_for(CONST, 0.0, 1, 32)
    at64 = unit_sprites.sprite_for(CONST, 0.0, 1, 64)
    assert at64.rgba.shape[0] == pytest.approx(2 * at32.rgba.shape[0], abs=1)
    assert at64.hotspot_x == pytest.approx(2 * at32.hotspot_x, abs=1)
    # The hotspot stays in the same relative spot, which is the property that
    # actually keeps a sprite anchored as the mip ladder changes tile_px.
    assert at64.hotspot_x / at64.rgba.shape[1] == pytest.approx(
        at32.hotspot_x / at32.rgba.shape[1], abs=0.02
    )


# --- caching ----------------------------------------------------------


def test_the_scaled_cache_evicts_rather_than_growing(install, monkeypatch):
    """sld_decoder holds no cache and a delta chain can be 313 links deep, so
    an unbounded cache here is a real memory risk on a unit-dense map. P3-g4
    sets the capacity; this only pins that it IS bounded."""
    install.write(build_sld(4))
    install.register(angle_count=4, frame_count=1)
    monkeypatch.setattr(unit_sprites._scaled_cache, "capacity", 2)
    for a in range(4):
        unit_sprites.sprite_for(CONST, 2 * math.pi * a / 4, 1, 32)
    assert len(unit_sprites._scaled_cache) == 2


def test_an_unresolvable_key_is_cached_so_it_is_not_re_walked(install, monkeypatch):
    """Caching the MISS is not an optimization, it is what makes the cache
    work at all: discovering that a unit resolves nowhere costs a full SLDFile
    walk. Measured on the 32k-unit corpus outlier before this existed, a WARM
    rebuild spent 1.59s of 1.74s re-walking files for the ~1% of units that
    resolve to nothing."""
    install.write(build_sld(2))
    install.register(angle_count=4, frame_count=3)  # claims 12 frames, 2 exist
    calls = []
    real = unit_sprites.load_sld
    monkeypatch.setattr(unit_sprites, "load_sld", lambda p: (calls.append(p), real(p))[1])

    for _ in range(5):
        assert unit_sprites.sprite_for(CONST, math.pi, 1, 32) is None
    assert len(calls) == 1, f"re-walked the file {len(calls)} times"


def test_a_missing_file_is_also_cached_as_a_miss(install, monkeypatch):
    install.register(file_name="t_absent_x1")
    calls = []
    real = unit_sprites.load_sld
    monkeypatch.setattr(unit_sprites, "load_sld", lambda p: (calls.append(p), real(p))[1])
    for _ in range(5):
        assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is None
    assert len(calls) == 1


def test_the_sprite_is_cropped_to_its_ink_and_the_hotspot_follows(install):
    """A frame's canvas is a median 6% ink; keeping the margin costs cache
    bytes and an alpha blend per transparent pixel. The hotspot has to be
    rebased with the crop or every sprite lands a margin's width off its
    tile."""
    canvas = 32
    # A file whose MAIN box is a 8x8 corner of a 32x32 canvas, so there is a
    # real margin to trim.
    n = (8 // 4) ** 2
    body = struct.pack("<4H2B", 4, 4, 12, 12, 0, 1)
    body += struct.pack("<H", 1) + bytes((0, n)) + b"".join([bc1_solid()] * n)
    layer = struct.pack("<I", 4 + len(body)) + body
    data = bytearray(_HEADER.pack(MAGIC, 4, 1, 0, 16, 0))
    data += _FRAME_HEADER.pack(canvas, canvas, 16, 16, MAIN, 0, 0)
    data += layer
    install.write(bytes(data))
    install.register(angle_count=1, frame_count=1)

    got = unit_sprites.sprite_for(CONST, 0.0, 0, unit_sprites.NATIVE_TILE_W // 2)
    assert got is not None
    # scale is 1.0 at half_w = NATIVE_TILE_W/2, so native pixels map 1:1.
    assert got.rgba.shape[:2] == (8, 8), "kept the transparent margin"
    assert (got.hotspot_x, got.hotspot_y) == (16 - 4, 16 - 4)
    assert (got.rgba[..., 3] > 0).all()


def test_repointing_the_install_drops_the_sprite_caches(install):
    """asset_source.set_install_path_override() must reach these caches. A
    cached MISS surviving a first-time install setup would leave every unit a
    coloured dot until the app restarted."""
    install.write(build_sld(4))
    install.register()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is not None
    assert len(unit_sprites._native_cache) == 1
    asset_source.set_install_path_override(None)
    assert len(unit_sprites._native_cache) == 0
    assert len(unit_sprites._scaled_cache) == 0


def test_clear_caches_drops_a_stale_install(install):
    """A cached array would otherwise outlive the file it came from when the
    configured install path changes."""
    install.write(build_sld(4))
    install.register()
    assert unit_sprites.sprite_for(CONST, 0.0, 1, 32) is not None
    assert len(unit_sprites._native_cache) == 1
    unit_sprites.clear_caches()
    assert len(unit_sprites._native_cache) == 0


# --- sprite_pieces_for (composite buildings) ---------------------------


def test_a_non_composite_unit_resolves_to_one_piece_at_zero_offset(install):
    """A unit_const with no "pieces" key must behave like sprite_for() itself,
    wrapped in a single-element list -- so a caller does not need two code
    paths for a composite vs. a plain unit."""
    install.write(build_sld(1))
    install.register(angle_count=1, frame_count=1)

    pieces = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, 32)
    direct = unit_sprites.sprite_for(CONST, 0.0, 0, 32)
    assert len(pieces) == 1
    assert pieces[0].dx == 0 and pieces[0].dy == 0
    assert np.array_equal(pieces[0].draw.rgba, direct.rgba)


def test_every_piece_resolves_at_its_own_native_pixel_offset(install):
    """Two distinct piece graphics, each at a different unit_graphic_map.json
    dx/dy -- resolved and scaled independently, in list order."""
    install.write(build_sld(1), name="t_piece_a_x1")
    install.write(build_sld(1), name="t_piece_b_x1")
    install.register_composite(CONST, [
        {"unit_id": CONST, "file_name": "t_piece_a_x1", "angle_count": 1,
         "frame_count": 1, "dx": 0, "dy": 0},
        {"unit_id": 999, "file_name": "t_piece_b_x1", "angle_count": 1,
         "frame_count": 1, "dx": 10, "dy": -20},
    ])

    half_w = unit_sprites.NATIVE_TILE_W // 2  # scale == 1.0, offsets pass through raw
    pieces = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, half_w)
    assert [(p.dx, p.dy) for p in pieces] == [(0, 0), (10, -20)]


def test_a_composite_scales_its_offsets_with_the_projection(install):
    install.write(build_sld(1), name="t_piece_a_x1")
    install.register_composite(CONST, [
        {"unit_id": CONST, "file_name": "t_piece_a_x1", "angle_count": 1,
         "frame_count": 1, "dx": 48, "dy": -24},
    ])
    half_w = unit_sprites.NATIVE_TILE_W  # scale == 2.0
    pieces = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, half_w)
    assert (pieces[0].dx, pieces[0].dy) == (96, -48)


def test_a_failing_parent_piece_drops_the_whole_composite(install):
    """The parent piece is identified by unit_id == unit_const, not list
    position -- depth order can put it anywhere in the list (a town centre's
    own back piece sorts to index 1, not 0), so this must not regress to an
    index check."""
    install.write(build_sld(1), name="t_piece_b_x1")  # only the non-parent file exists
    install.register_composite(CONST, [
        {"unit_id": 999, "file_name": "t_piece_b_x1", "angle_count": 1,
         "frame_count": 1, "dx": 5, "dy": 5},
        {"unit_id": CONST, "file_name": "t_missing_x1", "angle_count": 1,
         "frame_count": 1, "dx": 0, "dy": 0},
    ])
    assert unit_sprites.sprite_pieces_for(CONST, 0.0, 0, 32) == []


def test_a_failing_non_parent_piece_is_skipped_not_fatal(install):
    """Strictly better than today, never worse: a non-parent piece that fails
    to resolve drops out, but the rest of the composite still draws."""
    install.write(build_sld(1), name="t_piece_a_x1")
    install.register_composite(CONST, [
        {"unit_id": CONST, "file_name": "t_piece_a_x1", "angle_count": 1,
         "frame_count": 1, "dx": 0, "dy": 0},
        {"unit_id": 999, "file_name": "t_missing_x1", "angle_count": 1,
         "frame_count": 1, "dx": 5, "dy": 5},
    ])
    pieces = unit_sprites.sprite_pieces_for(CONST, 0.0, 0, 32)
    assert len(pieces) == 1
    assert (pieces[0].dx, pieces[0].dy) == (0, 0)


def test_each_piece_dispatches_rotation_through_its_own_resolving_unit_id(install, monkeypatch):
    """The whole reason sprite_pieces_for() threads a per-piece unit_id
    through to _draw_for_entry rather than the parent's unit_const: a piece's
    frame dispatch (angle_index vs. variant_index) is a property of its own
    graphic. Gate this const into _ROTATION_VARIANT_CONSTS under the PIECE's
    own id, not the parent's, and compare against _draw_for_entry called
    directly with that same piece id -- the one call the contract says is
    correct. At rotation=4.0, angle_count=5, variant_index reads the literal
    index 4 while angle_index resolves to 3 (floor(4.0/(2*pi)*5 + 0.5) == 3):
    a real, different frame, not just "some rotation happened to change
    something" -- so a regression to dispatching on unit_const (CONST is NOT
    in _ROTATION_VARIANT_CONSTS here) would resolve frame 3 and this would
    catch it as a pixel mismatch against the piece-id-correct frame 4.
    """
    install.write(build_sld(5), name="t_variant_piece_x1")
    piece_const = 999
    entry = {"file_name": "t_variant_piece_x1", "angle_count": 5, "frame_count": 1}
    install.register_composite(CONST, [
        {"unit_id": piece_const, "file_name": "t_variant_piece_x1",
         "angle_count": 5, "frame_count": 1, "dx": 0, "dy": 0},
    ])
    monkeypatch.setattr(unit_sprites, "_ROTATION_VARIANT_CONSTS", frozenset({piece_const}))
    assert unit_sprites.variant_index(4.0, 5) != unit_sprites.angle_index(4.0, 5)

    got = unit_sprites.sprite_pieces_for(CONST, 4.0, 0, 32)[0]
    expected = unit_sprites._draw_for_entry(piece_const, entry, 4.0, 0, 32)
    assert np.array_equal(got.draw.rgba, expected.rgba)


# --- icon_for (Flat's footprint-fitted icons, P3-g7) -------------------


def _register_tall_composite(install):
    """A composite whose assembly is deliberately NON-square: two 8x8 pieces
    stacked 8px apart, so the union ink is 8 wide by 16 tall. Every other
    fixture here is square, and a square ink cannot tell contain-fit apart
    from stretch-to-fill."""
    install.write(build_sld(1))
    install.register_composite(CONST, [
        {"unit_id": CONST, "file_name": FILE_NAME, "angle_count": 1,
         "frame_count": 1, "dx": 0, "dy": 0},
        {"unit_id": 777, "file_name": FILE_NAME, "angle_count": 1,
         "frame_count": 1, "dx": 0, "dy": 8},
    ])


def test_an_icon_contain_fits_rather_than_stretching(install):
    """8x16 ink into a 32x32 footprint is 16x32, not 32x32. Stretch-to-fill
    (the rejected alternative) would give the square, and a villager's real
    15x39 ink stretched into a square cell is a 4.3x distortion that reads as
    a blob rather than a unit."""
    _register_tall_composite(install)
    icon = unit_sprites.icon_for(CONST, 0.0, 0, 32, 32)
    assert icon.rgba.shape[:2] == (32, 16)


@pytest.mark.parametrize("fw,fh", [(16, 16), (17, 5), (5, 17), (1, 1), (128, 128), (64, 33)])
def test_an_icon_never_exceeds_its_footprint(install, fw, fh):
    """**The assertion P3-g7's whole no-widening argument rests on.** An icon
    that stays inside its own footprint rect is what lets Flat keep its
    existing per-tile edit rects and its exact canvas sizing, with none of the
    dirty-bbox widening, canvas headroom or MAX_SPRITE_REACH_* machinery
    Stepped and Sloped needed. One pixel of overhang and that is false.

    Includes tile_px=16 (the smallest shipped mip) and a 1x1 cell, where the
    max(1, ...) floor is what stops a zero-size array reaching _resize_rgba."""
    _register_tall_composite(install)
    icon = unit_sprites.icon_for(CONST, 0.0, 0, fw, fh)
    ih, iw = icon.rgba.shape[:2]
    assert 1 <= iw <= fw and 1 <= ih <= fh


def test_an_icon_is_placed_by_its_rect_not_a_hotspot(install):
    """A stale iso hotspot riding along on an icon is how it would silently
    get blitted off its own footprint -- the caller centres the rect."""
    install.write(build_sld(1))
    install.register(angle_count=1, frame_count=1)
    icon = unit_sprites.icon_for(CONST, 0.0, 0, 32, 32)
    assert (icon.hotspot_x, icon.hotspot_y) == (0, 0)


def test_a_composite_icon_is_the_whole_assembly_not_the_parent_piece(install):
    """Resolving only the entry's own graphic would give the parent's square
    8x8 ink (a 32x32 icon here) and drop the second piece entirely -- a town
    centre reduced to one of its four parts."""
    _register_tall_composite(install)
    composite = unit_sprites.icon_for(CONST, 0.0, 0, 32, 32)

    install.register(angle_count=1, frame_count=1)
    unit_sprites.clear_caches()
    parent_only = unit_sprites.icon_for(CONST, 0.0, 0, 32, 32)

    assert parent_only.rgba.shape[:2] == (32, 32)
    assert composite.rgba.shape[:2] != parent_only.rgba.shape[:2]


def test_an_unresolvable_icon_falls_back_to_none(install):
    install.register(angle_count=1, frame_count=1)  # no file written
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32) is None


def test_an_unknown_const_has_no_icon(install):
    install.register(angle_count=1, frame_count=1)
    assert unit_sprites.icon_for(CONST + 1, 0.0, 0, 32, 32) is None


def test_an_icon_is_cached_including_its_miss(install, monkeypatch):
    """Same _MISS rationale as the sprite path: re-deriving "this resolves
    nowhere" costs a whole SLDFile walk, and ~1% unresolvable units were most
    of a real file's warm rebuild cost before negative caching existed."""
    install.register(angle_count=1, frame_count=1)  # no file written
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32) is None

    calls = []
    real = unit_sprites._build_icon
    monkeypatch.setattr(unit_sprites, "_build_icon", lambda *a: calls.append(a) or real(*a))
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32) is None
    assert calls == []


def test_the_icon_cache_is_keyed_on_the_footprint_size(install):
    install.write(build_sld(1))
    install.register(angle_count=1, frame_count=1)
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32).rgba.shape[:2] == (32, 32)
    assert unit_sprites.icon_for(CONST, 0.0, 0, 16, 16).rgba.shape[:2] == (16, 16)


def test_clear_caches_drops_icons_too(install):
    """Missing this is how a stale icon survives an install-path change --
    the same first-time-configuration failure clear_caches() exists to stop
    for sprites."""
    install.register(angle_count=1, frame_count=1)
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32) is None
    install.write(build_sld(1))
    unit_sprites.clear_caches()
    assert unit_sprites.icon_for(CONST, 0.0, 0, 32, 32) is not None


def test_the_icon_cache_evicts_by_bytes_not_by_entry_count(install, monkeypatch):
    """An entry count cannot work here: a 1x1 unit at tile_px=16 is ~1KB and a
    4x4 Town Centre at tile_px=128 is ~1MB, so any single count is either
    wasteful at one end of that spread or thrashing at the other."""
    install.write(build_sld(1))
    install.register(angle_count=1, frame_count=1)
    monkeypatch.setattr(unit_sprites, "_icon_cache", unit_sprites._ByteLRU(8 * 1024))
    for size in (16, 17, 18, 19):  # ~1KB each, all four fit inside the budget
        unit_sprites.icon_for(CONST, 0.0, 0, size, size)
    assert len(unit_sprites._icon_cache) == 4
    # One 6.25KB entry then evicts SEVERAL of them. An entry count would have
    # held all five, which is the whole distinction being pinned here.
    unit_sprites.icon_for(CONST, 0.0, 0, 40, 40)
    assert len(unit_sprites._icon_cache) == 2


def test_a_native_miss_is_shared_across_scales(install, monkeypatch):
    """P3-g7 moved _MISS down from _scaled_cache (whose key carries a half_w)
    into _native_cache (keyed file+index, scale-independent). An icon's key
    carries a FOOTPRINT size instead of a half_w, so it could not have
    inherited the scaled-level misses -- it would have re-walked every
    unresolvable file once per footprint size per mip level."""
    install.register(angle_count=1, frame_count=1)  # no file written
    assert unit_sprites.sprite_for(CONST, 0.0, 0, 32) is None

    def boom(*_a, **_k):
        raise AssertionError("re-walked a file whose miss was already known")

    monkeypatch.setattr(unit_sprites, "load_sld", boom)
    assert unit_sprites.icon_for(CONST, 0.0, 0, 64, 64) is None
    assert unit_sprites.sprite_for(CONST, 0.0, 0, 16) is None
