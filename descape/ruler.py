"""Qt-free click grammar and arithmetic for the two-point Ruler tool. The Qt
half lives in viewer.MapView, which owns four stock scene items and feeds this
session tile coordinates it has already resolved and bounds-checked.

Split the way descape.brush already is: this module decides WHAT the current
measurement is, the view decides how it looks on screen. Nothing here imports
PyQt5, so the whole interaction grammar is testable without a QApplication.

Distances come from tile INDICES alone. Elevation never enters, and that is a
decision rather than an omission: AoE2 resolves unit range on the ground
plane, so a unit standing on a hill is not further away. The view draws
endpoints at each tile's real rendered elevation, which is the opposite choice
and equally deliberate, because a marker pinned to elevation 0 would float
below the cursor on raised terrain and read as a bug. The two are meant to
disagree; do not reconcile them by making the distance three-dimensional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Tile = tuple[int, int]

# IDLE: nothing drawn. PENDING: the first point is placed and the second is
# still following the cursor. DONE: both points are fixed.
STATE_IDLE = "idle"
STATE_PENDING = "pending"
STATE_DONE = "done"

# One decimal on every readout, including axis-aligned runs that land on a
# whole number ("12.0 tiles"). A format that changes shape with the value is
# harder to read at a glance, not easier.
DISTANCE_DECIMALS = 1
DISTANCE_UNIT = "tiles"


@dataclass(frozen=True)
class Measurement:
    """A finished or in-progress span between two tiles. dx/dy are signed and
    read a-to-b, so they answer "which way does this run", not just how far:
    a bare magnitude tells you a rectangle is 12 wide without telling you
    which direction to draw it."""

    a: Tile
    b: Tile

    @property
    def dx(self) -> int:
        return self.b[0] - self.a[0]

    @property
    def dy(self) -> int:
        return self.b[1] - self.a[1]

    @property
    def distance(self) -> float:
        """Straight-line tile distance. math.hypot rather than a hand-rolled
        sqrt(dx*dx + dy*dy) because it avoids the intermediate overflow and
        precision loss that form is prone to."""
        return math.hypot(self.dx, self.dy)


def measure(a: Tile, b: Tile) -> Measurement:
    return Measurement(a, b)


def format_measurement(m: Measurement) -> str:
    """The one-line readout shared by the on-map label and the status log, so
    the two can never disagree about a number the user is comparing."""
    return f"{m.distance:.{DISTANCE_DECIMALS}f} {DISTANCE_UNIT}  (dx {m.dx:+d}, dy {m.dy:+d})"


class RulerSession:
    """The press/move/release grammar, as a three-state machine.

    | event   | IDLE          | PENDING                   | DONE          |
    |---------|---------------|---------------------------|---------------|
    | press   | a = b = tile  | b = tile, finish          | a = b = tile  |
    | move    | ignored       | b = tile                  | ignored       |
    | release | ignored       | a == b stays, else finish | ignored       |

    PENDING is what lets click-drag-release and click-move-click share one
    grammar instead of two code paths. A press and release inside a single
    tile is indistinguishable from a user arming a second click, so it stays
    PENDING rather than completing a zero-length measurement by accident; a
    release after a real drag completes. A deliberate second press on the
    start tile does complete, at distance 0, because an explicit click should
    never leave the tool stuck.
    """

    def __init__(self) -> None:
        self._state = STATE_IDLE
        self._a: Tile | None = None
        self._b: Tile | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def endpoints(self) -> tuple[Tile, Tile] | None:
        """Both points, or None when there is nothing to draw. Non-None for
        every state except IDLE, including PENDING, where b is still tracking
        the cursor."""
        if self._a is None or self._b is None:
            return None
        return self._a, self._b

    @property
    def measurement(self) -> Measurement | None:
        ends = self.endpoints
        return None if ends is None else measure(*ends)

    def press(self, tile: Tile) -> str:
        if self._state == STATE_PENDING:
            self._b = tile
            self._state = STATE_DONE
        else:
            self._a = tile
            self._b = tile
            self._state = STATE_PENDING
        return self._state

    def move(self, tile: Tile) -> bool:
        """True when the endpoint actually moved, so the caller can skip
        rebuilding scene items on the many mouse-move events that land on the
        tile the line already ends at."""
        if self._state != STATE_PENDING or tile == self._b:
            return False
        self._b = tile
        return True

    def release(self) -> str:
        """Takes no tile on purpose: move() has already placed b wherever the
        cursor last resolved to a real tile, and re-reading the release
        position would undo the off-map guard that kept b valid."""
        if self._state == STATE_PENDING and self._a != self._b:
            self._state = STATE_DONE
        return self._state

    def clear(self) -> None:
        self._state = STATE_IDLE
        self._a = None
        self._b = None
