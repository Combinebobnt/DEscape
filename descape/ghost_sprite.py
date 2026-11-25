"""A unit ghost's sprite art per mip level (TASK-291), Qt-free.

The canvas paints each level from that level's own resolve: the anchor, the
hotspot and the size are each rounded at the level's own half_w. A ghost that
resolved level 0 only and let Qt scale it sat up to a level pixel off the
placed sprite at every other zoom. UnitGhostItem asks this module for the
level the canvas paints, so the two draw the same pixels in the same place.
"""

from __future__ import annotations

import math

from descape import render_sprites, unit_sprites

# How far a level's ghost rect may overhang level 0's, in that level's pixels: the hero glow ring,
# which is fixed in level pixels, plus the anchor's, the hotspot's and the size's rounding.
# Measured (TASK-291 step 0, every graphic_map() const, Stepped and hilly Sloped): 3.5 at the 16 px level.
LEVEL_OVERHANG_LEVEL_PX = unit_sprites.HERO_GLOW_RADIUS + 2


class GhostSprite:
    """One ghost unit's sprite draws, memoized per level.

    Captures everything unit_sprite_draws_at() reads besides the level's
    projection and corner_rise, so a level resolves on its first paint there
    with the same inputs level 0 resolved with. A level's memo is kept only
    while the cache hands back the same proj and corner_rise objects."""

    __slots__ = (
        "_memo",
        "elevations",
        "fold",
        "hero_glow",
        "player_id",
        "rotation_override",
        "scenario",
        "tree_scale",
        "unit",
        "wall_alpha",
    )

    def __init__(
        self,
        scenario,
        elevations,
        player_id: int,
        unit,
        *,
        rotation_override: float | None = None,
        tree_scale: float = 1.0,
        hero_glow: bool = False,
        wall_alpha: float = 1.0,
        fold=None,
    ) -> None:
        self.scenario = scenario
        self.elevations = elevations
        self.player_id = player_id
        self.unit = unit
        self.rotation_override = rotation_override
        self.tree_scale = tree_scale
        self.hero_glow = hero_glow
        self.wall_alpha = wall_alpha
        self.fold = fold
        # mip -> (proj, corner_rise, draws)
        self._memo: dict[int, tuple[object, object, list]] = {}

    def draws(self, mip: int, proj, corner_rise) -> list:
        """render_sprites.unit_sprite_draws_at()'s (draw, px, py) triples at
        level `mip`, in that level's pixels."""
        hit = self._memo.get(mip)
        if hit is not None and hit[0] is proj and hit[1] is corner_rise:
            return hit[2]
        found = render_sprites.unit_sprite_draws_at(
            self.scenario,
            proj,
            self.elevations,
            corner_rise,
            self.player_id,
            self.unit,
            rotation_override=self.rotation_override,
            tree_scale=self.tree_scale,
            hero_glow=self.hero_glow,
            wall_alpha=self.wall_alpha,
            fold=self.fold,
        )
        self._memo[mip] = (proj, corner_rise, found)
        return found

    def level0(self) -> list:
        """The level-0 draws the caller resolved, or [] before it did."""
        hit = self._memo.get(0)
        return [] if hit is None else hit[2]


def level_entries(sprites, cache, mip: int) -> list[tuple[object, float, float, float, str | None]]:
    """(draw, x0, y0, scale, look) per sprite piece the ghost draws at level
    `mip`: x0, y0 the piece's top-left in level pixels, scale the level's
    mip_scale(), which takes them to scene units. `sprites` are (GhostSprite,
    look) pairs. A member whose level resolves to nothing keeps its level-0
    draws, at scale 1."""
    proj, corner_rise = cache.level_sprite_inputs(mip)
    scale = cache.mip_scale(mip)
    out = []
    for member, look in sprites:
        draws, s = member.draws(mip, proj, corner_rise), scale
        if not draws:
            draws, s = member.level0(), 1.0
        for draw, ax, ay in draws:
            h, w = draw.rgba.shape[:2]
            if h and w:
                out.append((draw, float(ax - draw.hotspot_x), float(ay - draw.hotspot_y), s, look))
    return out


def level_pad(cache) -> float:
    """Scene px a level-0 ghost rect grows by per side so it holds every
    level's rect: the overhang bound at the coarsest level, plus one."""
    coarsest = cache.mip_levels()[0]
    return 1.0 + LEVEL_OVERHANG_LEVEL_PX * math.ceil(cache.mip_scale(coarsest))
