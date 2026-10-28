"""The Unit Brush's planner (GH #170): which units one brush dab places.

A stroke (one press-drag-release) decides each tile it covers once: with
probability `density` the tile gets a unit, else it is skipped for the rest
of the stroke, so dwelling or re-crossing adds nothing. A unit sits on its
tile's snapped anchor for the const's span (Place Unit's tile_span(), so a
mountain or a House lands on the game's grid), plus a seeded in-tile offset
on each 1-wide axis when Off-grid is on. It is skipped when its footprint
would cross the map edge or touch a tile this stroke already filled (a
stroke never stacks on itself, Free placement or not), and, with the rules
on, when placement_rules.refused() refuses it. Every unit this stroke
places is learned by the stroke's BlockerGrid before the next candidate is
judged, so candidates in one dab see each other.

The rotation comes only from place_rotation: random_rotation() with Random
ticked, else rotation_for() of the pending placement rotation (AGENTS.md,
*Place Unit's placement rotation*).

Walls, gates and cliffs are refused at the press (is_refused_const()): each
is placed as a run or a chain by its own tool, and a per-tile copy would
write shapes the game re-derives. Qt-free; the viewer owns the model edit.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from descape import gate_orientation, placement_rules, unit_kind, unit_sprites
from descape.place_rotation import PlaceRotation, random_rotation, rotation_for
from descape.render import NON_BUILDING_SPAN, _span_start, occupied_tiles_for, snapped_anchor
from descape.scatter import MAX_JITTER
from descape.terrain_palette import tile_span


def is_refused_const(const: int) -> str | None:
    """Why the brush does not paint `const`, naming the tool that places it, or None."""
    if gate_orientation.is_gate(const):
        return "gates are placed one at a time with Place Unit"
    if const in unit_sprites.wall_family_consts():
        return "walls are placed as runs with Place Unit, Wall Rectangle or Wall Circle"
    if const in unit_kind.cliff_consts():
        return "cliffs are placed as chains with the Cliff tool"
    return None


@dataclass(frozen=True)
class Spec:
    """One unit to add: UnitEditModel.add()'s arguments."""

    player: int
    unit_const: int
    x: float
    y: float
    rotation: float


@dataclass
class BrushStroke:
    """One drag's state, built at press from widgets read once."""

    consts: tuple[int, ...]
    chosen_owner: int
    density: float  # 0.01..1.0
    rules: bool  # False with Free placement ticked
    randomize: bool
    pending: PlaceRotation
    rng: random.Random
    off_grid: bool = False
    # None until the first dab: Alt is read from its modifiers and held.
    jitter: float | None = None
    grid: placement_rules.BlockerGrid | None = None
    laid: dict | None = None
    rolled: set = field(default_factory=set)
    filled: set = field(default_factory=set)
    placed: list = field(default_factory=list)  # (player, unit)
    refused: int = 0
    refusal_reason: str | None = None  # the first refused candidate's rule message
    gaia_only: int = 0

    def owners(self) -> list[int]:
        """Every owner placement_owner() can give this stroke's consts."""
        return sorted({unit_kind.placement_owner(const, self.chosen_owner) for const in self.consts})


def _fits(x: float, y: float, span: tuple[int, int], width: int, height: int) -> bool:
    """Whether a `span` footprint anchored at (x, y) lies wholly on the map.
    occupied_tiles_for() clips instead, so it cannot answer this."""
    x0, y0 = _span_start(x, span[0]), _span_start(y, span[1])
    return x0 >= 0 and y0 >= 0 and x0 + span[0] <= width and y0 + span[1] <= height


def _ensure_rule_state(scenario, stroke: BrushStroke) -> None:
    """The grid and, for a stroke with a mobile const, the laid terrain, built
    on the first candidate the rules judge. The grid then learns every
    placement. The laid map needs no update: a bridge this stroke places
    fills its own tiles, so no later candidate of the stroke stands there."""
    if stroke.grid is None:
        stroke.grid = placement_rules.BlockerGrid(scenario)
    if stroke.laid is None and any(const in unit_kind.mobile_unit_consts() for const in stroke.consts):
        stroke.laid = placement_rules.laid_terrain(scenario)


def paint_dab(scenario, stroke: BrushStroke, tiles: Iterable[tuple[int, int]], add: Callable[[Spec], object]) -> list:
    """Places this dab's units through `add` (which returns the new unit) and
    returns them as [(player, unit)]. `tiles` is the brush footprint."""
    mm = scenario.map_manager
    width, height = mm.map_width, mm.map_height
    rng = stroke.rng
    jitter = min(stroke.jitter or 0.0, MAX_JITTER)
    placed = []
    for tile in sorted(set(tiles)):
        if tile in stroke.rolled:
            continue
        stroke.rolled.add(tile)
        if rng.random() >= stroke.density:
            continue
        const = stroke.consts[0] if len(stroke.consts) == 1 else rng.choice(stroke.consts)
        owner = unit_kind.placement_owner(const, stroke.chosen_owner)
        span = tile_span(const, NON_BUILDING_SPAN)
        x, y = snapped_anchor(tile, None, span, width, height)
        if jitter:
            # Both draws always happen, so the stream does not depend on the span.
            jx, jy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
            x += jx if span[0] == 1 else 0.0
            y += jy if span[1] == 1 else 0.0
        if not _fits(x, y, span, width, height):
            continue
        footprint = occupied_tiles_for(const, x, y, width, height) or []
        if any(t in stroke.filled for t in footprint):
            continue
        if stroke.rules:
            _ensure_rule_state(scenario, stroke)
            if placement_rules.refused(scenario, const, x, y, grid=stroke.grid, laid=stroke.laid):
                stroke.refused += 1
                if stroke.refusal_reason is None:
                    # Built now, once: a bridge or blocker this stroke adds later would change the answer.
                    stroke.refusal_reason = placement_rules.refusal(
                        scenario, const, x, y, grid=stroke.grid, laid=stroke.laid
                    )
                continue
        rotation = random_rotation(rng, const, owner) if stroke.randomize else rotation_for(stroke.pending, const, owner)
        unit = add(Spec(owner, const, x, y, rotation))
        # On the stroke at once: a raise later in this dab must still find it to commit.
        stroke.placed.append((owner, unit))
        placed.append((owner, unit))
        if owner != stroke.chosen_owner:
            stroke.gaia_only += 1
        # Learned before the next candidate, so one dab never overlaps itself.
        stroke.filled.update(footprint)
        if stroke.grid is not None:
            stroke.grid.add(owner, unit)
    return placed
