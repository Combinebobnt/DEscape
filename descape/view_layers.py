"""Which categories of rendering run at all -- the View > Layers registry.

A leaf module by construction, like unit_filter.py: its only intra-project
import is terrain_style, itself a leaf that imports nothing, so render.py,
render_cache.py and settings.py can all import it with no cycle risk and no
import-time cost (settings.py in particular is widely imported and
deliberately stays off terrain_palette, which parses two JSON files at
import -- see REBINDABLE_ACTIONS' own comment about that).

A table plus one loop rather than N hand-written menu blocks: a sibling
backlog item (the grid terrain overlay) plugs in as a row instead of
re-deriving the same action/gating/keybind pattern.

Layer visibility is a third axis, distinct from the two the existing menus
already own: Filters is WHICH units are drawn, View > Show sprites is WHETHER
unit art draws at all, and this is a render-category question -- which
categories of rendering happen, and (small_trees, translucent_walls) how
one category's art is drawn. That is why a row about tree or
wall ART still belongs here and not in Filters, which decides membership
rather than appearance.

Every row persists across launches, by user decision (GH #182), stored by
settings.get_view_layers()/set_view_layer() under the config key
`view_layers`. That departs from settings.py's content-hiding/passive-chrome
split, which still keeps the unit filter and Show sprites session-only; the
accepted cost is that a layer left hidden comes back hidden. Since all rows
persist, the old idea of a per-row `persist` field is moot.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import terrain_style

SMALL_TREE_SCALE = 0.6
"""The factor the Small Trees row draws tree art at. Fixed rather than
user-adjustable: the value is a readability tradeoff settled once, so only
the on/off bool is a setting, not the factor. Lives here because this
module is where the bool is defined; unit_sprites wants only the float."""

# Translucent Walls' opacity, Settings > Appearance (GH #35). Here rather than
# in settings.py so this module stays a leaf: settings.py imports these.
WALL_OPACITY_MIN_PCT = 10
WALL_OPACITY_MAX_PCT = 90
WALL_OPACITY_DEFAULT_PCT = 50


@dataclass(frozen=True)
class LayerState:
    """An immutable "render these layers" state, whose defaults are each row's
    own LayerSpec.default (on for the two visibility rows and Hero Glow, off
    for Small Trees and Translucent Walls).

    Frozen for exactly the reason UnitFilter is: it is stored on a chunk
    cache and compared BY VALUE to decide whether a change needs the cache
    evicting, and a mutable state edited in place would silently skip that
    eviction -- every field here is composited into cached chunk PIXELS, so
    the toggle would appear to do nothing until the user scrolled somewhere
    uncached.

    No `is_default` property, unlike UnitFilter: nothing here needs to
    collapse a combination of checkboxes into one canonical default-equal
    value, so there would be no consumer for it.
    """

    terrain_textures: bool = True
    terrain_blend: bool = True
    farm_overlay: bool = True
    small_trees: bool = False
    hero_glow: bool = True
    translucent_walls: bool = False
    # Not a registry row: Settings > Appearance owns it, and the viewer fills it in.
    wall_opacity_pct: int = WALL_OPACITY_DEFAULT_PCT

    @property
    def tree_scale(self) -> float:
        """The bool as the float every sprite call site actually wants, so
        the mapping is written once here rather than at each of the six
        cache re-resolve sites."""
        return SMALL_TREE_SCALE if self.small_trees else 1.0

    @property
    def wall_alpha(self) -> float:
        """Translucent Walls as the alpha factor wall art is drawn at. The
        caches compare this, not the raw pct, so moving the opacity slider
        with the row off rebuilds nothing."""
        return self.wall_opacity_pct / 100 if self.translucent_walls else 1.0


DEFAULT_LAYERS = LayerState()
"""The state every row's own default adds up to, used as the default argument
at every render.py and render_cache.py call site so existing callers and tests
stay byte-identical without naming this module at all."""


@dataclass(frozen=True)
class LayerSpec:
    """One row of the registry -- everything the menu, the gating and the
    keybind table need, so each of those is a loop over LAYERS rather than a
    hand-written block per layer."""

    layer_id: str
    label: str
    tooltip: str
    default: bool
    # None means every render style. Otherwise the styles where this layer
    # has anything to do at all -- a fact about the render, not policy.
    styles: frozenset[str] | None
    requires_sprites: bool
    keybind_label: str
    # Draws over the real textures, so greyed while Terrain Textures is off.
    requires_textures: bool = False


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec(
        layer_id="terrain_textures",
        label="&Terrain Textures",
        tooltip=(
            "Draw each tile as its flat terrain colour instead of the real "
            "terrain texture -- the same look as running with no AoE2DE "
            "install configured. Useful for reading elevation and unit "
            "placement without the texture detail."
        ),
        default=True,
        styles=None,
        requires_sprites=False,
        keybind_label="Terrain Textures Layer",
    ),
    LayerSpec(
        layer_id="terrain_blend",
        label="Terrain &Blending",
        tooltip=(
            "Blend each tile into higher-priority neighbouring terrain through "
            "the game's own transition masks, and draw a tile's second "
            "(layered) terrain over it. Off, every tile is a hard-edged crop "
            "of its own texture."
        ),
        default=True,
        styles=None,
        requires_sprites=False,
        keybind_label="Terrain Blending Layer",
        requires_textures=True,
    ),
    LayerSpec(
        layer_id="farm_overlay",
        label="&Farm Terrain Overlay",
        tooltip=(
            "Draw a farm's footprint as its own farm terrain, tinted with the "
            "owner's colour. With this off a farm is no longer skipped at "
            "paint time, so it draws as an ordinary coloured mark instead -- "
            "this hides the terrain override, not the farm."
        ),
        default=True,
        # All three caches drape: Stepped (P3 farm terrain), Sloped (Track C6,
        # over its footprint's warped terrain) and top-down Flat (GH #150,
        # FlatChunkCache's own farm tiles).
        styles=None,
        # Every cache builds the drape only when sprites are enabled; with
        # sprites off the plain coloured mark is already what draws.
        requires_sprites=True,
        keybind_label="Farm Terrain Overlay Layer",
    ),
    LayerSpec(
        layer_id="small_trees",
        label="&Small Trees",
        tooltip=(
            "Draw tree sprites at 60% size so the tiles behind and beside a "
            "tree stay readable. The tree still stands on the same tile and "
            "is still picked by its whole footprint -- this shrinks the art, "
            "not the unit."
        ),
        # The first row here to default off: it changes the map away from
        # what the game itself shows, where the other two only ever restore
        # detail.
        default=False,
        # Flat contain-fits every unit's art into its own footprint rect
        # (unit_sprites.icon_for), so a tree there already cannot overflow
        # onto a neighbouring tile and there is nothing behind it to reveal.
        styles=frozenset(terrain_style.ELEVATED_STYLES),
        # The factor is baked into the sprite the caches store, which they
        # only build with sprites enabled; with sprites off the coloured
        # mark is already tile-sized.
        requires_sprites=True,
        keybind_label="Small Trees Layer",
    ),
    LayerSpec(
        layer_id="hero_glow",
        label="&Hero Glow",
        tooltip=(
            "Draw a gold outline round hero units, like the glow the game "
            "draws round them. Turn it off to see a hero's own sprite edge."
        ),
        # On, for parity with the game, which always draws it.
        default=True,
        # Flat's icons go through unit_sprites.icon_for, a separate path and
        # cache this ring does not reach.
        styles=frozenset(terrain_style.ELEVATED_STYLES),
        # The ring is baked into the sprite the caches store, which they only
        # build with sprites enabled.
        requires_sprites=True,
        keybind_label="Hero Glow Layer",
    ),
    LayerSpec(
        layer_id="translucent_walls",
        label="T&ranslucent Walls",
        tooltip=(
            "Draw walls and gates see-through, at the opacity set in Settings "
            "> Appearance, so units behind a wall line show. A wall is still "
            "picked by its footprint -- this changes the art, not the unit."
        ),
        # Off, like Small Trees: it changes the map away from what the game shows.
        default=False,
        # Flat's icons are contain-fit into their own footprint, so a wall
        # there hides nothing behind it.
        styles=frozenset(terrain_style.ELEVATED_STYLES),
        # Baked into the sprite the caches store, which they only build with
        # sprites enabled.
        requires_sprites=True,
        keybind_label="Translucent Walls Layer",
    ),
)


def availability(
    spec: LayerSpec, *, style: str, sprites_enabled: bool, has_map: bool, terrain_textures: bool = True
) -> tuple[bool, str]:
    """Whether `spec`'s toggle is usable right now, plus the tooltip to show.

    Qt-free on purpose, so the whole gating matrix is unit-testable without
    building a window -- the reason this lives here rather than inline in
    ViewerWindow._update_tool_enabled().

    `style` is the RENDER style ("flat"/"stepped"/"sloped"), not the terrain
    style: Flat + Isometric View renders through IsoChunkCache, where small
    trees really do shrink, so keying on the terrain style would grey a live
    layer.

    Returns (enabled, tooltip). An unavailable layer keeps its checked state;
    the caller must never write the returned flag back into its LayerState.
    """
    if not has_map:
        return False, "Open a map first"
    if spec.styles is not None and style not in spec.styles:
        pretty = ", ".join(
            terrain_style.label_for(s) for s in terrain_style.TERRAIN_STYLES if s in spec.styles
        )
        return False, f"Only drawn in {pretty} ({terrain_style.label_for(style)} has no such layer)"
    if spec.requires_sprites and not sprites_enabled:
        return False, "Needs View > Show sprites, which is what builds this layer"
    if spec.requires_textures and not terrain_textures:
        return False, "Needs View > Layers > Terrain Textures, which is what this blends"
    return True, spec.tooltip
