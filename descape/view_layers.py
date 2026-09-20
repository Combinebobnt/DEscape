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
categories of rendering happen, and (small_trees) how big one category's art
is drawn. That is why a row about tree ART size still belongs here and not in
Filters, which decides membership rather than appearance.

Session-only, deliberately: every row changes what the map itself shows, so
they follow the unit filter and Show sprites rather than the persisted
passive chrome (settings.py's own content-hiding/passive-chrome split). A
future passive-chrome layer would add a `persist` field here plus the
config.example.yaml obligations that come with it; that is not added
speculatively.
"""

from __future__ import annotations

from dataclasses import dataclass

from descape import terrain_style

SMALL_TREE_SCALE = 0.6
"""The factor the Small Trees row draws tree art at. Fixed rather than
user-adjustable: the value is a readability tradeoff settled once, and a
settings entry would carry persistence and a config.example.yaml obligation
this session-only menu deliberately has none of. Lives here because this
module is where the bool is defined; unit_sprites wants only the float."""


@dataclass(frozen=True)
class LayerState:
    """An immutable "render these layers" state, whose defaults are each row's
    own LayerSpec.default (on for the two visibility rows, off for the
    Small Trees size row).

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
    farm_overlay: bool = True
    small_trees: bool = False

    @property
    def tree_scale(self) -> float:
        """The bool as the float every sprite call site actually wants, so
        the mapping is written once here rather than at each of the six
        cache re-resolve sites."""
        return SMALL_TREE_SCALE if self.small_trees else 1.0


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
        layer_id="farm_overlay",
        label="&Farm Terrain Overlay",
        tooltip=(
            "Draw a farm's footprint as its own farm terrain, tinted with the "
            "owner's colour. With this off a farm is no longer skipped at "
            "paint time, so it draws as an ordinary coloured mark instead -- "
            "this hides the terrain override, not the farm."
        ),
        default=True,
        # Flat has no farm path at all: FlatChunkCache composites through
        # unit_draws/icons, never through the SpriteLayer that carries
        # farm_by_tile. Stepped (P3 farm terrain) and Sloped (Track C6, which
        # drapes a farm over its footprint's warped terrain) both do -- which
        # is exactly terrain_style.ELEVATED_STYLES, not a coincidence worth
        # re-enumerating here.
        styles=frozenset(terrain_style.ELEVATED_STYLES),
        # The farm override rides SpriteLayer, which the caches only build
        # when sprites are enabled; with sprites off there is no farm_by_tile
        # to suppress and the plain coloured mark is already what draws.
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
)


def availability(
    spec: LayerSpec, *, style: str, sprites_enabled: bool, has_map: bool
) -> tuple[bool, str]:
    """Whether `spec`'s toggle is usable right now, plus the tooltip to show.

    Qt-free on purpose, so the whole gating matrix is unit-testable without
    building a window -- the reason this lives here rather than inline in
    ViewerWindow._update_tool_enabled().

    `style` is the RENDER style ("flat"/"stepped"/"sloped"), not the terrain
    style: Flat + Isometric View renders through IsoChunkCache, where farms
    really do drape, so keying on the terrain style would grey a live layer.

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
    return True, spec.tooltip
