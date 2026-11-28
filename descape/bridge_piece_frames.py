"""What each frame of a DE bridge piece shows, so a turn can remap it (TASK-705).

The twelve DE bridge pieces (`s_bridge_piece_*`, 1840-2041) with a (1, 2) or
(2, 1) span come in transposed pairs (`b` (1, 2), `c` (2, 1)), which
object_orientation.py pairs into two-groups. Unlike a classic bridge, whose
pieces step along their short axis and whose ends are separate consts, a DE
piece runs along its LONG axis (a `b` along y, a `c` along x), and which end of
the run it is lives in its `rotation`, a VARIANT frame index. So turning one
means taking its sibling const and remapping that frame.

Measured 2026-11-26 and re-rendered 2026-11-27 (unit_sprites._native_frame()):
an end piece's two frames are mirror images along the run, the side face tall
at one end and zero at the other. The tall, open end joins the next piece, so a
frame is recorded by its inward vector: the side the rest of the bridge is on.
AoE2:DE's own campaign scenarios place 1840 and 1841 that way (12 of 16 clear
cases, the 4 others where ends overlap ends); cd and wood rest on the rendered
frames alone. Middle and broken frames are cosmetic kinds, and a pair's `b` and
`c` list them in different orders, so an axis swap maps them by kind. A broken
frame's damage sits on one end or one side and no frame holds it reversed:
that side does not turn (accepted, as a reflected classic bridge's railings).

A row applies only while the active dataset's graphic for its const has the
measured file name and frame count (covers()), so a data mod that swaps or
reshapes the art keeps the piece as it was. Stdlib plus the dataset modules.
"""

from __future__ import annotations

from descape import games, unit_sprites, unit_variant

END = "end"
_N, _S, _E, _W = (0, -1), (0, 1), (1, 0), (-1, 0)

# const -> (file_name, frames): each frame ("end", inward vector) or (kind,).
_AOE2 = {
    1840: ("s_bridge_piece_ab_end_b_x1", ((END, _N), (END, _S))),
    1841: ("s_bridge_piece_ab_end_c_x1", ((END, _E), (END, _W))),
    1993: ("s_bridge_piece_cd_end_b_x1", ((END, _N), (END, _S))),
    1994: ("s_bridge_piece_cd_end_c_x1", ((END, _E), (END, _W))),
    2039: ("s_bridge_piece_wood_end_b_x1", ((END, _S), (END, _N))),
    2041: ("s_bridge_piece_wood_end_c_x1", ((END, _E), (END, _W))),
    1996: ("s_bridge_piece_cd_middle_b_x1", (("arched",), ("solid",))),
    1997: ("s_bridge_piece_cd_middle_c_x1", (("arched",), ("solid",))),
    1844: ("s_bridge_piece_ab_broken_b_x1", (("holed",), ("broken_deck",), ("gap",))),
    1845: ("s_bridge_piece_ab_broken_c_x1", (("gap",), ("holed",), ("broken_deck",))),
    1999: ("s_bridge_piece_cd_broken_b_x1", (("crumbled",), ("broken_side",), ("gap",))),
    2000: ("s_bridge_piece_cd_broken_c_x1", (("gap",), ("crumbled",), ("broken_side",))),
}

_TABLE = games.PerGame(
    "bridge_piece_frames: what each frame of a DE bridge piece shows",
    # ror: the same consts, file names and frame counts (tests/test_bridge_piece_frames.py).
    # aoc and hd: no DE bridge pieces.
    {"aoe2": _AOE2, "ror": _AOE2, "aoc": {}, "hd": {}},
)


def frames(unit_const: int) -> tuple[tuple, ...] | None:
    """The active game's measured frames for `unit_const`, or None."""
    row = _TABLE.active().get(unit_const)
    return None if row is None else row[1]


def covers(unit_const: int) -> bool:
    """Whether `unit_const` has a row that matches the active dataset's art:
    the same file name and as many frames as its variant_count."""
    row = _TABLE.active().get(unit_const)
    if row is None:
        return False
    file_name, row_frames = row
    graphic = unit_sprites.graphic_map().get(unit_const) or {}
    return graphic.get("file_name") == file_name and len(row_frames) == unit_variant.variant_count_for(unit_const)


def _row(unit_const: int) -> tuple[tuple, ...]:
    row = frames(unit_const)
    if row is None:
        raise ValueError(f"unit_const {unit_const} is not a measured DE bridge piece")
    return row


def image_frame(unit_const: int, frame: int, new_const: int, linear) -> int:
    """The frame of `new_const` that shows `unit_const`'s `frame` after the
    element whose linear part is `linear` (a function of a direction vector).
    An end frame takes the end frame whose inward vector is nearest the image
    of its own (exact for D4); a cosmetic frame the frame of the same kind.
    Raises for a const with no row, a frame outside its row, or a tie."""
    source, target = _row(unit_const), _row(new_const)
    if not 0 <= frame < len(source):
        raise ValueError(f"unit_const {unit_const} has no frame {frame} (it has {len(source)})")
    entry = source[frame]
    if entry[0] != END:
        matches = [k for k, other in enumerate(target) if other == entry]
        if len(matches) != 1:
            raise ValueError(f"unit_const {new_const} has no single {entry[0]} frame")
        return matches[0]
    vx, vy = linear(*entry[1])
    scored = sorted(
        ((vx * dx + vy * dy, k) for k, (kind, *rest) in enumerate(target) if kind == END for dx, dy in rest),
        reverse=True,
    )
    if not scored or (len(scored) > 1 and scored[0][0] - scored[1][0] < 1e-9):
        raise ValueError(f"unit_const {unit_const} frame {frame}'s image has no single nearest end of {new_const}")
    return scored[0][1]
