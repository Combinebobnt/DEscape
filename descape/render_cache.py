"""Qt-free LRU caches of composited canvas chunks, one per terrain style.

This module keeps the three concrete caches, IsoChunkCache, FlatChunkCache
and SlopedChunkCache, with DEFAULT_CHUNK_PX, SPRITES_ENABLED and the pick-plane
constants. What they build on is one module per area:

- render_cache_common.py: layer-field tuples, warm, stamp and reduce
  constants, box_reduce2, _Reduced, ChunkJob, RectJob, LevelWarmJob,
  join_chunk_jobs.
- render_cache_splice.py: UnitSplice, splice eligibility and refusal,
  component walks, splice plans, elevation splices, unit re-anchoring.
- render_cache_base.py: _ChunkCacheBase's core: construction, the composite
  hooks, the chunk store and pooled build, terrain mirrors, the set_* calls.
  The rest of the class is three mixins:
  - render_cache_mip.py: _ChunkCacheMipMixin, the mip ladder, level warm
    jobs, residency and stamps, reduced chunks.
  - render_cache_units.py: _ChunkCacheUnitsMixin, sprite extents, splice
    plans and refusal traces, invalidate_units, flag hosts, the wall verdict.
  - render_cache_patch.py: _ChunkCachePatchMixin, patch() through
    invalidate_region().
- render_cache_leveled.py: _LeveledChunkCache, _IsoLevel, _SlopedLevel,
  WARM_NATIVE_INPUTS, NATIVE_PRIME_MAX_TILE_PX.

Rules of the split (tests/test_render_cache_split.py checks the first two):

- A render_cache module imports only the ones below it: render_cache_common
  and render_cache_splice import none; the three mixin modules may import
  those two; render_cache_base those five; render_cache_leveled those six.
  No render_cache module imports this one.
- A _ChunkCacheBase mixin defines no __init__, and no name is defined by two
  of the four classes.
- Tests and tools set a render_cache module's global only through
  testkit.render_family, which sets it in every module that binds it.

Split out of render.py, which had grown past 3700 lines. The dependency runs
one way only: these modules import the render modules (render_paint,
render_units, render_sprites, render_bbox, render_iso, render_sloped,
render_flat) and none of them imports one back, so the cluster is an acyclic
leaf.

Back-references reach a render module through ATTRIBUTE ACCESS
(render_bbox.foo(...)), never `from descape.render_bbox import foo`. Tests
patch a render global through testkit.render_family, which sets it in every
render module that binds it; a from-import here would bind the name at
import time and miss that patch.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from descape import (
    iso_geometry,
    native_composite,
    perf_trace,
    render_bbox,
    render_flat,
    render_iso,
    render_paint,
    render_sloped,
    render_sprites,
    render_units,
    settings,
    unit_sprites,
)
from descape.render_cache_base import _ChunkCacheBase
from descape.render_cache_common import LevelWarmJob
from descape.render_cache_leveled import _IsoLevel, _LeveledChunkCache, _SlopedLevel
from descape.render_cache_splice import (
    _SLOPED_ELEV_SPLICE_MAX_UNITS,
    UnitSplice,
    _const_splice_eligible,
    _corner_changed_tiles,
    _fold_changed_tiles,
    _splice_overrides,
    _write_dirty_elevations,
)
from descape.scenario_io import LoadedScenario
from descape.unit_filter import UnitFilter
from descape.view_layers import DEFAULT_LAYERS, LayerState, render_key

# Default chunk size for IsoChunkCache -- measured, not guessed: benched
# CHUNK_PX in {256, 512, 1024} via tools/verify_iso_chunks.py on this
# project's largest real map (480x480, tile_px=32). 1024 wins cold
# full-canvas assembly by only ~8% over 512 (4.7s vs 5.1s) while being a
# much coarser invalidation granularity for a moving viewport; 256 loses on
# both cold assembly (~6s) and warm single-tile patch cost (~44ms vs ~28ms).
DEFAULT_CHUNK_PX = 512


# P3-g3: whether IsoChunkCache composites real .sld sprites instead of
# coloured marks. OFF for now, deliberately -- P3-g4 is the measurement gate
# that decided an opt-in toggle over default-on, and defaulting off here means
# every existing byte-identity test keeps its meaning untouched while the new
# sprite tests opt in explicitly. The headless full-render path has its own
# with_sprites argument instead, so a test can render ground truth without
# touching global state.
#
# P3-g: this is the CONSTRUCTION DEFAULT for IsoChunkCache's `sprites`
# parameter, not a live switch. Nothing reads it at call time anymore -- the
# live flag is per-cache (_ChunkCacheBase.sprites_enabled), because a
# per-window toggle cannot be a process global: module state outlives a
# window's close(), so a GUI toggle would leak across windows and into other
# tests. Defined HERE, above IsoChunkCache, rather than beside SpriteLayer
# below, purely because a default argument is evaluated when the class body
# executes -- from below the class it would raise NameError at import.
SPRITES_ENABLED = False


class IsoChunkCache(_LeveledChunkCache):
    """Qt-free LRU cache of composited Stepped-mode canvas chunks, keyed by
    (mip, chunk_x, chunk_y) -- Phase B-B of Track B.
    chunk_x/chunk_y are chunk-GRID indices: canvas pixel
    (chunk_x*chunk_px, chunk_y*chunk_px) is that chunk's own origin. mip is
    real as of Phase B-D-a (previously always 0); Phase B-D-b (this
    version) enumerates the REAL per-level projection set via
    iso_geometry.mip_projections_for(), keyed off settings.get_elev_step_pct()
    -- the same function every real proj-construction site in this module
    already calls, so the cache reading it too reproduces exactly what
    built `proj`. Neither B-D-a nor B-D-b changes a single rendered pixel
    reachable from the running app: nothing outside this class and its
    tests ever asks for mip != 0 until Phase B-D-c wires MapCanvasItem.
    paint() to a real LOD signal.

    Each chunk is composited independently via composite_rect_iso() -- the
    SAME function render_terrain_iso_with_proj()'s full loop and
    refresh_region_iso()'s per-edit patch both reduce to -- so a chunk's
    pixels never depend on which OTHER chunks happen to be cached or in
    what order they were requested (this class's own load-bearing
    correctness bar, see tools/verify_iso_chunks.py). Grid/LRU mechanics
    (get_chunk/render_rect/patch/invalidate_region) live in _ChunkCacheBase,
    shared with Phase B-E's FlatChunkCache -- this class supplies only the
    Stepped-specific pieces: canvas_dims(), _composite_rect(), and
    _refresh_source_caches().

    Wired into viewer.py's Stepped mode as of Phase B-C, via MapCanvasItem;
    also exercised standalone by tools/verify_iso_chunks.py and its own
    bench.

    max_chunks defaults to covering the WHOLE canvas at chunk_px (see
    _ChunkCacheBase._init_max_chunks()'s own docstring for why).

    NOT free memory-wise: a fully-warmed cache holds roughly as many total
    pixel bytes as the old single canvas did (this project's biggest real
    map, 480x480 at chunk_px=512, needs 480 chunks), and render_rect() (see
    below) additionally allocates a fresh full-canvas-sized stitched output
    array on every full-viewport call -- so a full-viewport composite
    transiently needs the persistent cache AND that scratch buffer at once.
    Measured on that map: peak RSS for an open+warm+edit workflow went
    743MB -> 1127MB versus the pre-B-C single-buffer approach -- a real,
    accepted cost of avoiding thrashing at the default zoom, not a memory
    win. Pass an explicit
    smaller max_chunks (as tools/verify_iso_chunks.py's eviction tests do)
    to exercise real LRU eviction instead."""

    style = "stepped"
    _has_unit_pack = True

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.elevations = elevations
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since _level()
        # reads it to decide whether to build a level's sprite layer at all.
        self.sprites_enabled = sprites
        # Same ordering constraint: _level() reads the build-time layer
        # fields (_BUILD_TIME_LAYER_FIELDS) to build a level's sprite layer.
        self.layers = render_key(layers)
        # Phase B-D-b: the REAL per-level projection set. settings.
        # get_elev_step_pct() is read here rather than threaded through as
        # a parameter because it's exactly the same function every real
        # proj-construction site in this module already calls to build
        # `proj` itself -- so this reproduces what built `proj`, and
        # mip_projections_for's own identity-level self-check (an assert)
        # fails loudly if the two ever disagreed. Must happen before
        # _init_max_chunks(), which reads canvas_dims() -> self._levels[0].proj.
        mm = scenario.map_manager
        self._init_levels(
            iso_geometry.mip_projections_for(
                mm.map_width, mm.map_height, proj, settings.get_elev_step_pct(),
                max_tile_px=render_paint.finest_tile_pixels(), native_tier_max=render_paint.native_tier_max(),
            )
        )
        # canvas_dims()'s own exactness assert: is_exact_mip() proves every
        # IsoProjection field scales exactly, but canvas_dims() adds the
        # skirt-headroom term (_canvas_pixel_dims) on top of proj.canvas_h --
        # the value the blit actually trusts -- so assert it here too,
        # against the real function rather than duplicating its formula
        # into iso_geometry (a second place that formula could drift).
        ref_w, ref_h = render_bbox._canvas_pixel_dims(proj)
        for lvl in self._levels.values():
            lw, lh = render_bbox._canvas_pixel_dims(lvl.proj)
            assert lw * proj.tile_px == ref_w * lvl.tile_px and lh * proj.tile_px == ref_h * lvl.tile_px, (
                f"level tile_px={lvl.tile_px}'s canvas_dims (skirt headroom included) isn't an "
                f"exact mip of the reference's -- {(lw, lh)} vs reference {(ref_w, ref_h)}"
            )
        self._init_terrain_ids()
        self._record_flag_hosts()
        self._record_wall_verdict()
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """Rebuilds the SHARED, tile-space units_by_tile and bumps the
        source generation counter -- must run whenever elevations could
        have changed underneath this cache (construction, and every
        patch()). Deliberately does NOT rebuild any level's
        building_bboxes here: those are PROJECTION-dependent (a building's
        screen bbox depends on its center tile's elevation via
        _unit_screen_bbox_iso), so under mips they are per-level, and
        rebuilding every ENUMERATED level here would make a single edit
        pay a whole-map walk per level -- for levels that may hold no
        cached chunks at all. Each level's bboxes are instead rebuilt
        lazily, in _level(), the first time that level is actually
        composited after this bump -- see _level()'s own docstring.

        elevation_changed (draw-perf plan Step 3, 3a/3b): None means
        "unknown" -- construction, or a unit-filter/sprite-toggle refresh
        via _refresh_unit_sources() -- and gets today's unconditional
        wholesale behavior (units_by_tile rebuilds, gen always bumps),
        since THOSE changed and every level's bboxes/sprites must go stale.

        A patch() caller instead passes the elevation-changed subset of its
        own edit (empty for a terrain-paint-only edit). units_by_tile is
        NEVER rebuilt then (3a): no terrain or elevation edit can move a
        unit, so this dict cannot go stale from a patch(). The unit layers are
        re-anchored in place by _refresh_level_units(), seeded by
        elevation_changed itself: Stepped reads a unit's OWN tile only, radius
        0, unlike Sloped's corner-changed ring."""
        if elevation_changed is None:
            self._refresh_all_units()
            return
        if not self.with_units or not elevation_changed:
            return
        self._refresh_level_units(elevation_changed, splice_levels)

    def refresh_wholesale(self, dirty_indices, flatten: bool = False) -> bool | None:
        """The in-place counterpart of a full re-render, for an edit past
        STEPPED_FULL_RERENDER_THRESHOLD (TASK-031.61 A5): this cache, its
        elevations array and every level's SpriteMemo survive, so the next
        level build re-resolves only the units whose elevation key moved.

        Writes the dirty tiles' elevations into self.elevations in place
        (_write_dirty_elevations(); None, touching nothing, when one is out of
        range), bumps the gen directly when any moved (not through
        _refresh_source_caches(), whose elevation splice would seed and walk a
        component over every tile), then re-reads the terrain ids and evicts
        the whole canvas at every level. units_by_tile stays: no elevation or
        terrain edit moves a unit. Returns whether an elevation moved."""
        moved = _write_dirty_elevations(self.scenario, self.elevations, self.proj, dirty_indices, flatten)
        if moved is None:
            return None
        self._mutation_epoch += 1
        if moved:
            self._source_gen += 1
        self.invalidate_region((0, 0, *self.canvas_dims(0)))
        return moved

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in LEVEL `mip` canvas pixels, including skirt
        headroom -- see _canvas_pixel_dims(). Indexes self._levels[mip]
        DIRECTLY (never through _level()), so sizing the cache can never
        trigger a building_bboxes build."""
        return render_bbox._canvas_pixel_dims(self._levels[mip].proj)

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        lvl = self._level(mip)
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units(lvl, mip)
        with perf_trace.repaint_phase("composite"):
            return render_iso.composite_rect_iso(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                self.elevations,
                lvl.proj,
                lvl.tile_px,
                self.units_by_tile,
                lvl.building_bboxes,
                self.with_units,
                sprites=lvl.sprites,
                bystander_grid=lvl.bystander_grid,
                layers=self.layers,
                grid=self.grid,
                unit_pack=unit_pack,
                terrain_ids=self._terrain_ids,
                layer_ids=self._layer_ids,
                key_map=self._blend_key_map,
            )

    def _prepare_rect(
        self, mip: int, x0: int, y0: int, x1: int, y1: int, heights: np.ndarray | None = None
    ) -> tuple | None:
        lvl = self._level(mip)
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units(lvl, mip)
        with perf_trace.repaint_phase("prepare"):
            return render_bbox.prepare_rect_native(
                False, self.scenario, x0, y0, x1, y1, self.elevations, lvl.proj, lvl.tile_px, lvl.building_bboxes,
                self.with_units, lvl.bystander_grid, self.layers, self.grid, unit_pack,
                terrain_ids=self._terrain_ids, layer_ids=self._layer_ids, key_map=self._blend_key_map,
                snapshot=heights,
            )

    def _heights_snapshot(self, mip: int) -> np.ndarray | None:
        # The bbox functions write elevations in place, so the batch's jobs share one copy.
        return self.elevations.copy()


class FlatChunkCache(_ChunkCacheBase):
    """Flat mode's counterpart to IsoChunkCache -- Phase B-E of Track B. Same
    grid/LRU mechanics (_ChunkCacheBase), composited via composite_rect_flat()
    instead of composite_rect_iso()
    (see that function's own docstring for the correctness argument behind
    never needing refresh_units_over()'s fixed-point dirty-tile expansion
    here).

    max_chunks defaults to covering the WHOLE canvas at chunk_px, same as
    IsoChunkCache and for the same fitInView()-on-open reason -- unchanged
    here even though Flat's canvas is the LARGER of the two (675 MiB vs
    340 MiB at 480x480): capping this to something smaller would thrash
    at the app's own default zoom exactly
    the way IsoChunkCache's own docstring already argues against. Phase
    B-D's mip levels are the real fix for the fit-to-view memory cost, not
    a smaller cap here.

    _refresh_source_caches() is a documented no-op after construction:
    unlike Stepped's building bboxes (which move with their center tile's
    elevation, see IsoChunkCache._refresh_source_caches()), Flat's
    unit_draws (_flat_unit_draws()) derive only from unit.x/unit.y,
    unit_const, the owning player index, and map dimensions -- none of
    which any terrain or elevation edit touches. That makes patch() here
    genuinely cheaper than Stepped's per-edit re-anchor of the units on
    changed tiles, not just an equivalent no-op restated. A unit
    edit goes through invalidate_units() instead, which splices the edited
    rows or, failing that, forces a rebuild. Don't
    "fix" this no-op into an unconditional rebuild instead, since that
    would silently reintroduce the per-edit cost this class exists to
    avoid paying for edits that were never about units at all. The same
    argument covers P3-g7's per-level icon layers, which derive from the same
    unit data and so go stale through the same funnel.

    Real .sld icons as of P3-g7 (`sprites=`/set_sprites_enabled()), replacing
    the coloured rect per unit -- see _level_icons() and
    render_flat.composite_rect_flat(). With sprites on and the Farm Terrain
    Overlay layer on, a farm drapes as terrain too (GH #150): _farm_tiles,
    built with unit_draws, and the FLAT_DRAPED icon that drops its rect."""

    style = "flat"

    def __init__(
        self,
        scenario: LoadedScenario,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, the same ordering
        # trap IsoChunkCache.__init__ and SlopedChunkCache.__init__ document.
        self.sprites_enabled = sprites
        # terrain_textures/terrain_blend are read per composite; farm_overlay
        # at build time, by _refresh_source_caches() and the icon walks.
        self.layers = render_key(layers)
        # Phase B-D-b: the real candidate ladder, UNFILTERED -- "unfiltered"
        # here means the MIP ladder, nothing to do with unit_filter. Flat has no
        # projection (canvas_dims() is a bare multiply, exact at every
        # tile_px, no elev_step term to break exactness), so every power of
        # two mip_tile_px_candidates() finds is a real exact level, unlike
        # Stepped's mip_projections_for() which must filter through a real
        # exactness check. Must precede _init_max_chunks(), which reads
        # canvas_dims(). No native tier: top-down Flat draws footprint-fitted
        # icons, which have no native size (TASK-599 Decision 5).
        self._init_mip_levels(iso_geometry.mip_tile_px_candidates(tile_px, render_paint.finest_tile_pixels()))
        # Per-level unit_draws. Level 0's draws live in self.unit_draws (still
        # read directly by tests/test_flat_chunks.py); other levels are derived
        # lazily here from the mip-independent row columns (_derive_bboxes()).
        self._level_draws: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        # P3-g7's per-level icon layers. Level 0 lives in here TOO, unlike
        # _level_draws -- see _level_icons().
        self._level_icon_layers: dict[int, dict[int, unit_sprites.SpriteDraw] | None] = {}
        # Bumped by invalidate_units(), read only by level_warm_job(). Flat
        # needs a counter where the rest of this class does not, and the
        # reason is specific to the warm: every OTHER reader of the icon
        # layers goes through _level_icons(), which finds an absent dict entry
        # and rebuilds -- "absent" is a complete staleness test for a
        # synchronous caller. A warm spans event-loop turns, so it has to
        # distinguish "absent because nobody built it yet" from "absent
        # because invalidate_units() cleared it WHILE I was walking", and
        # those two states are byte-identical. The row-count assert doesn't
        # cover the gap either: a unit MOVED (not added or removed) leaves the
        # count intact while shifting the icons' meaning. A row splice leaves
        # the layer present but re-keyed, which the counter covers too.
        self._unit_gen = 0
        # _refresh_source_caches() also sets the row columns (_row_tiles,
        # _row_offsets, _row_colors, _row_uid, _row_player), one entry per row
        # of every level's draws (identical across levels).
        # _rows_unit_gen is the scenario.unit_gen they describe; _window_built
        # names each lazy level layer built while the model was ahead of it.
        self._rows_unit_gen = self._model_unit_gen()
        self._window_built: set[tuple[str, int]] = set()
        self._init_terrain_ids()
        self._record_flag_hosts()
        self._record_wall_verdict()
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """See this class's own docstring: a documented no-op once
        unit_draws already exists (nothing a terrain/elevation edit touches
        can make it stale), except at construction, where it must actually
        build unit_draws the first time. elevation_changed unused -- Flat
        has no elevation term at all, see this module's docstring.

        The farm drape (GH #150) is built in the same block, so construction
        and every wholesale invalidate_units() (set_layers, set_sprites_enabled,
        set_unit_filter) rebuild it with the draws."""
        if not hasattr(self, "unit_draws"):
            if self.with_units:
                self._row_tiles, self._row_offsets, self._row_colors, self._row_uid, self._row_player = (
                    render_flat._flat_unit_base_rows(self.scenario, self.unit_filter)
                )
                self.unit_draws = (self._derive_bboxes(0), self._row_colors)
            else:
                self.unit_draws = None
                self._row_tiles = self._row_offsets = self._row_colors = self._row_uid = self._row_player = None
            self._rebuild_farm_tiles()

    def _derive_bboxes(self, mip: int) -> np.ndarray:
        """Level mip's bboxes from the row columns, about 0.1 ms on 13k rows."""
        return render_flat.flat_unit_bboxes(self._row_tiles, self._row_offsets, self._mip_tile_px[mip])

    def _rebuild_farm_tiles(self) -> None:
        """_farm_tiles (flat_farm_tiles()' by_tile, mip-independent) and
        _farm_unit_ids, or None and an empty set while farms don't drape."""
        if self.farms_draped():
            self._farm_tiles, self._farm_unit_ids = render_sprites.flat_farm_tiles(self.scenario, self.unit_filter)
        else:
            self._farm_tiles, self._farm_unit_ids = None, set()

    def can_splice(self, changed: list[UnitSplice] | None) -> bool:
        """Whether invalidate_units(changed) would take its row splice."""
        return self.with_units and bool(changed) and self._flat_splice_eligible(changed)

    def sprite_extent_before(self, changed: list[UnitSplice], mip: int | None):
        """Flat patches rects sized from the touched tiles (its icons stay inside
        their footprint), so a unit edit here never asks for a sprite extent:
        raise, whatever mip."""
        raise NotImplementedError("FlatChunkCache has no unit-edit sprite extent")

    sprite_extent_after = sprite_extent_before

    def invalidate_units(
        self, changed: list[UnitSplice] | None = None, splice_levels: bool = True, cap_levels=None
    ) -> None:
        """Brings unit_draws, every other level's draws and every icon layer
        up to date after a unit edit. Phase 3.5b's unit-editing UI calls this
        via ViewerWindow._after_unit_mutation(); it overrides the base class
        because _refresh_source_caches()'s no-op would otherwise miss those
        edits.

        `changed` (a list of UnitSplice, in mutation order) takes the row
        splice when _flat_splice_eligible() allows it: Move/Nudge/Rotate/Set
        field rewrite their row in place, Delete and a reassign's source side
        drop their row, Add and a reassign's destination side insert one at
        the end of the owner's block (add() and reassign() both append). Only
        resident levels hold rows, and a non-resident level builds fresh from
        live scenario state in the same order. None, an empty list, or an
        ineligible batch takes the wholesale path below, which drops
        everything and rebuilds level 0's draws now and the rest lazily.
        splice_levels and cap_levels are Stepped/Sloped keywords, ignored here.

        A level layer built lazily inside a mutate-now, repaint-later window
        (the Eraser's, Convert's and the Unit Brush's, GH #170: a zoom onto a
        never-built level mid-drag) walked the live lists, which the row table
        does not describe yet: an add gives it an extra row, a reassign moves
        one. It is dropped here (_drop_window_built()) and rebuilds lazily
        from the lists this call brings the row table up to. A layer built
        outside such a window is kept, so the row splice stays the fast path.
        A batch that flips the wall radian verdict is taken as None (TASK-583)."""
        self._mutation_epoch += 1
        self._record_flag_hosts()
        changed = self._note_wall_verdict(changed)
        if self._window_built:
            self._drop_window_built()
        if self.can_splice(changed):
            self._splice_rows(changed)
            self._rows_unit_gen = self._model_unit_gen()
            return
        if hasattr(self, "unit_draws"):
            del self.unit_draws
        self._level_draws.clear()
        # In the same statement group as the draws, not a step later: icons are
        # keyed by ROW INDEX into the draws, so a unit mutation or filter change
        # that dropped one without the other would leave a stale icon painting
        # on a shifted row -- i.e. on the wrong unit.
        self._level_icon_layers.clear()
        # In the same group for the same reason: an in-flight warm's icons are
        # keyed to the row order this call just invalidated.
        self._unit_gen += 1
        self._refresh_source_caches()
        self._rows_unit_gen = self._model_unit_gen()

    def _in_unit_window(self) -> bool:
        """Whether the model is ahead of the row table: a mutate-now,
        repaint-later tool's window (the Eraser's, Convert's, the Unit Brush's)."""
        return self._model_unit_gen() != self._rows_unit_gen

    def _note_level_build(self, kind: str, mip: int) -> None:
        """Records a lazy "draws" or "icons" build made while the model is
        ahead of the row table, for invalidate_units() to drop."""
        if self._in_unit_window():
            self._window_built.add((kind, mip))

    def _drop_window_built(self) -> None:
        for kind, mip in self._window_built:
            (self._level_draws if kind == "draws" else self._level_icon_layers).pop(mip, None)
        self._window_built.clear()

    @staticmethod
    def _is_row_move(splice: UnitSplice) -> bool:
        """Same owner, present before and after: the row stays where it is."""
        return (
            splice.old_own_tile is not None
            and splice.new_own_tile is not None
            and splice.old_player_id in (None, splice.player_id)
        )

    def _row_of(self, unit) -> int | None:
        hits = np.flatnonzero(self._row_uid == id(unit))
        return int(hits[0]) if hits.size else None

    def _wants_row(self, player_id: int, unit) -> bool:
        """Whether a fresh _flat_unit_rows() walk would give `unit` a row."""
        mm = self.scenario.map_manager
        return self.unit_filter.matches(player_id, unit) and (
            render_units.unit_tile_bounds(unit, mm.map_width, mm.map_height) is not None
        )

    def _flat_splice_eligible(self, changed: list[UnitSplice]) -> bool:
        """Whether _splice_rows() reproduces a fresh walk for `changed`. No
        tile-sharing test, unlike _splice_eligible(): Flat paints overlapping
        rects in row order and the splice keeps that order exactly. What it
        can't do is a moved or added neighbour-dependent const (a pure
        reassign of one moves nothing, so no neighbour's icon changes), a unit
        edited twice in one
        batch (its row would be looked up against the wrong state), or a Move
        that enters or leaves the map (its row would have to appear or vanish
        mid-block, not at the block's end). Nor an insertion that isn't at its
        owner's list tail: undo's restore() puts a removed unit back mid-list,
        so each player's inserted units must be exactly the last ones in
        units[p], in `changed` order."""
        seen: set[int] = set()
        inserted: dict[int, list] = {}
        for s in changed:
            if (not s.is_reassign and not _const_splice_eligible(s.unit)) or id(s.unit) in seen:
                return False
            seen.add(id(s.unit))
            if self._is_row_move(s):
                if (self._row_of(s.unit) is not None) != self._wants_row(s.player_id, s.unit):
                    return False
            elif s.new_own_tile is not None:
                inserted.setdefault(s.player_id, []).append(s.unit)
        units = self.scenario.unit_manager.units
        for player_id, added in inserted.items():
            tail = units[player_id][-len(added):]
            if len(tail) != len(added) or any(a is not b for a, b in zip(added, tail, strict=True)):
                return False
        return True

    def _splice_rows(self, changed: list[UnitSplice]) -> None:
        """The row splice itself, applied to the row columns (tiles, offsets,
        colours, row_uid, row_player) and every resident level's icon layer;
        then unit_draws and every cached level's draws are re-derived from the
        columns (_derive_bboxes()), all sharing the one colours array.
        Overrides are _splice_overrides()': the real dict when a reassigned
        wall's icon reads it, else {} (_flat_splice_eligible() refuses a
        moved one)."""
        scenario = self.scenario
        mm = scenario.map_manager
        w, h = mm.map_width, mm.map_height
        overrides = _splice_overrides(scenario, changed)
        icon_layers = {mip: icons for mip, icons in self._level_icon_layers.items() if icons is not None}
        moves = [s for s in changed if self._is_row_move(s)]
        removals = [s for s in changed if not self._is_row_move(s) and s.old_own_tile is not None]
        insertions = [s for s in changed if not self._is_row_move(s) and s.new_own_tile is not None]

        # 1. Moves rewrite their own row in place; indices don't shift.
        for s in moves:
            row = self._row_of(s.unit)
            if row is None:
                continue
            bounds = render_units.unit_tile_bounds(s.unit, w, h)
            self._row_tiles[row] = bounds
            self._row_offsets[row] = render_units.unit_paint_offset(s.unit)
            self._row_colors[row] = render_units._unit_color(s.unit, scenario.player_colors[s.player_id])
            team_index = scenario.team_indices[s.player_id]
            for mip, icons in icon_layers.items():
                icon = render_flat._flat_unit_icon(
                    s.unit, s.player_id, s.index, overrides, team_index, bounds, self._mip_tile_px[mip],
                    render_units.player_art_for(scenario, s.player_id), self.layers.farm_overlay,
                )
                if icon is None:
                    icons.pop(row, None)
                else:
                    icons[row] = icon

        # 2. Removals, looked up before anything shifts.
        removed = np.array(sorted({r for s in removals if (r := self._row_of(s.unit)) is not None}), dtype=np.int64)
        if removed.size:
            keep = np.ones(len(self._row_uid), dtype=bool)
            keep[removed] = False
            self._row_tiles, self._row_offsets, self._row_colors = (
                self._row_tiles[keep], self._row_offsets[keep], self._row_colors[keep]
            )
            self._row_uid, self._row_player = self._row_uid[keep], self._row_player[keep]
            for mip, icons in icon_layers.items():
                keys = np.fromiter(icons.keys(), dtype=np.int64, count=len(icons))
                survivors = keep[keys]
                new_keys = keys[survivors] - np.searchsorted(removed, keys[survivors])
                values = [v for v, alive in zip(icons.values(), survivors.tolist(), strict=True) if alive]
                icon_layers[mip] = dict(zip(new_keys.tolist(), values, strict=True))

        # 3. Insertions at the end of the owner's block, in `changed` order.
        added = [
            (int(np.searchsorted(self._row_player, s.player_id, side="right")), s, render_units.unit_tile_bounds(s.unit, w, h))
            for s in insertions
            if self._wants_row(s.player_id, s.unit)
        ]
        if added:
            positions = np.array([pos for pos, _s, _b in added], dtype=np.int64)
            order = np.argsort(positions, kind="stable")
            final = np.empty(len(added), dtype=np.int64)
            final[order] = positions[order] + np.arange(len(added))
            grown = len(self._row_uid) + len(added)
            is_new = np.zeros(grown, dtype=bool)
            is_new[final] = True

            def grow(old: np.ndarray, new_values) -> np.ndarray:
                out = np.empty((grown, *old.shape[1:]), dtype=old.dtype)
                out[~is_new] = old
                out[final] = new_values
                return out

            self._row_uid = grow(self._row_uid, [id(s.unit) for _p, s, _b in added])
            self._row_player = grow(self._row_player, [s.player_id for _p, s, _b in added])
            self._row_tiles = grow(self._row_tiles, [b for _p, _s, b in added])
            self._row_offsets = grow(self._row_offsets, [render_units.unit_paint_offset(s.unit) for _p, s, _b in added])
            self._row_colors = grow(
                self._row_colors, [render_units._unit_color(s.unit, scenario.player_colors[s.player_id]) for _p, s, _b in added]
            )
            sorted_positions = positions[order]
            for mip, icons in icon_layers.items():
                keys = np.fromiter(icons.keys(), dtype=np.int64, count=len(icons))
                new_keys = keys + np.searchsorted(sorted_positions, keys, side="right")
                rekeyed = dict(zip(new_keys.tolist(), icons.values(), strict=True))
                for row, (_p, s, bounds) in zip(final.tolist(), added, strict=True):
                    icon = render_flat._flat_unit_icon(
                        s.unit, s.player_id, s.index, overrides, scenario.team_indices[s.player_id],
                        bounds, self._mip_tile_px[mip], render_units.player_art_for(scenario, s.player_id),
                        self.layers.farm_overlay,
                    )
                    if icon is not None:
                        rekeyed[row] = icon
                icon_layers[mip] = rekeyed

        self.unit_draws = (self._derive_bboxes(0), self._row_colors)
        for mip in self._level_draws:
            self._level_draws[mip] = (self._derive_bboxes(mip), self._row_colors)
        self._level_icon_layers.update(icon_layers)
        # A farm moved, added, removed, recoloured or replaced to or from a
        # draping const: the drape is rebuilt whole. A unit merely on a farm is not.
        if self.farms_draped() and any(
            id(s.unit) in self._farm_unit_ids or render_sprites._terrain_overlay_for(s.unit.unit_const) is not None
            for s in changed
        ):
            self._rebuild_farm_tiles()
        # An in-flight warm walked the pre-splice rows; this makes its install refuse.
        self._unit_gen += 1
        rows = len(self._row_uid)
        assert all(len(bboxes) == rows for bboxes, _colors in (self.unit_draws, *self._level_draws.values())), (
            "a Flat row splice left a level's draws out of step with the row table"
        )

    def _refresh_unit_sources(self) -> None:
        """See _ChunkCacheBase._refresh_unit_sources(): the base version
        calls _refresh_source_caches(), which this class defines as a no-op
        once unit_draws exists, so a filter change has to go through
        invalidate_units() to force the real rebuild (level 0's draws AND
        every other level's)."""
        self.invalidate_units()

    def _level_unit_draws(self, mip: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-level _flat_unit_draws() -- Flat's counterpart to Stepped's
        per-level building_bboxes. No generation counter needed here,
        unlike IsoChunkCache._level(): unit_draws depends only on
        unit.x/unit.y, unit_const, owning player index and map dimensions,
        none of which any terrain/elevation edit touches -- the same
        property that makes _refresh_source_caches() a no-op after
        construction. A unit edit is what makes them stale, and
        invalidate_units() either splices the edited rows into every entry
        here or clears this dict --
        which is also why set_unit_filter() routes through invalidate_units()
        rather than rebuilding self.unit_draws alone: these per-level draws
        must honour the new filter too, or hidden units would keep painting
        at every zoom level except the one that happened to be resident.

        A new level derives from the row columns (_derive_bboxes()), except
        inside a mutate-now window, where the columns lag the model: there it
        walks the live lists, as the icon layer built beside it does, so the
        two agree row for row, and invalidate_units() drops it. Level 0 keeps
        self.unit_draws (the row table itself) unless its icon layer is absent
        inside the window: that layer will walk the live lists, so level 0 does
        too, held in _level_draws[0] until invalidate_units() drops it (TASK-604)."""
        if not self.with_units:
            return None
        draws = self._level_draws.get(mip)
        live_mip0 = self.sprites_enabled and self._in_unit_window() and 0 not in self._level_icon_layers
        if mip == 0 and draws is None and not live_mip0:
            return self.unit_draws
        if draws is None:
            with perf_trace.level("flat-draws", mip):
                if self._in_unit_window():
                    draws = render_flat._flat_unit_draws(self.scenario, self._mip_tile_px[mip], self.unit_filter)
                else:
                    draws = (self._derive_bboxes(mip), self._row_colors)
            self._level_draws[mip] = draws
            self._note_level_build("draws", mip)
        return draws

    def _level_icons(self, mip: int) -> dict[int, unit_sprites.SpriteDraw] | None:
        """Per-level _flat_icon_layer() -- P3-g7's counterpart to
        _level_unit_draws() above, built lazily for the same reason and going
        stale through the same invalidate_units() funnel. None when this cache
        is not drawing sprites at all.

        **Level 0 is stored in the dict like every other level**, deliberately
        NOT mirroring _level_unit_draws()' `mip == 0` special case. That case
        exists only because self.unit_draws is a public attribute read
        externally (tests/test_flat_chunks.py); icons have no such reader, so a
        self.icons twin would buy nothing and add a second lifetime to keep in
        lockstep with unit_draws -- precisely the row-index desync this layer's
        whole keying scheme depends on avoiding. One dict, one lifetime.

        The row-count assert is the other half of that: _flat_icon_layer() and
        _flat_unit_draws() are two separate walks that MUST agree row for row,
        and this turns a future divergence into a build-time failure instead of
        icons silently painting on the wrong units. It compares WALK LENGTHS,
        not the maximum key -- a skipped unit that failed to advance `row`
        shifts every later icon onto its neighbour while staying in range, and
        a max-key check would sail past exactly that."""
        if not (self.with_units and self.sprites_enabled):
            return None
        if mip not in self._level_icon_layers:
            with perf_trace.level("flat-icons", mip):
                icons = render_flat._flat_icon_layer(
                    self.scenario, self._mip_tile_px[mip], self.unit_filter, self.layers.farm_overlay
                )
            self._install_icon_layer(mip, icons)
        return self._level_icon_layers[mip]

    def _install_icon_layer(self, mip: int, icons_and_rows: tuple[dict[int, unit_sprites.SpriteDraw], int]) -> None:
        """Installs an already-built (icons, row_count) pair, row-count assert
        included -- the shared tail of _level_icons() above and
        level_warm_job()'s install closure, so the warm path can't grow its
        own copy of the assert-then-store step (or quietly skip the assert)."""
        icons, rows = icons_and_rows
        draws = self._level_unit_draws(mip)
        assert rows == (0 if draws is None else len(draws[0])), (
            f"icon layer at mip {mip} walked {rows} units but _flat_unit_draws() "
            f"produced {0 if draws is None else len(draws[0])} rows -- the two walks have diverged"
        )
        self._level_icon_layers[mip] = icons
        self._note_level_build("icons", mip)

    def level_warm_job(self, mip: int) -> LevelWarmJob | None:
        """Flat's resumable warm for one level -- IsoChunkCache.
        level_warm_job()'s counterpart, same shape, different validity
        predicate (see self._unit_gen's own comment in __init__ for why
        "absent from _level_icon_layers" cannot be the whole test here).

        The level's draws are derived from the row columns inside the
        install (about 0.1 ms, TASK-031.72), so only the icon walk (~727ms
        cold) is worth resuming. With sprites off there is no job at all: such
        a level is resident already (is_level_resident()).

        scenario.unit_gen (_model_unit_gen()) is compared too, ahead of the
        row-count assert: an Eraser or Convert mutation moves only the model
        until its flush, so _unit_gen alone installs a walk built across it
        (TASK-326)."""
        if not (self.with_units and self.sprites_enabled):
            return None
        if mip in self._level_icon_layers:
            return None
        start_gen = self._unit_gen
        start_unit_gen = self._model_unit_gen()
        gen = render_flat._flat_icon_layer_sliced(
            self.scenario, self._mip_tile_px[mip], self.unit_filter, self.layers.farm_overlay
        )

        def install(icons_and_rows: tuple[dict[int, unit_sprites.SpriteDraw], int]) -> bool:
            if self._unit_gen != start_gen or self._model_unit_gen() != start_unit_gen:
                return False
            if mip in self._level_icon_layers:
                return False
            self._install_icon_layer(mip, icons_and_rows)
            return True

        return LevelWarmJob(gen=gen, install=install, kind="icons")

    def level_warmable(self, mip: int) -> bool:
        """level_warm_job(mip) is not None, without building the job."""
        return self.with_units and self.sprites_enabled and mip not in self._level_icon_layers

    def is_level_resident(self, mip: int) -> bool:
        """Whether a composite at mip builds nothing costly (TASK-031.72).
        Units off: always. Sprites on: the icon layer is built (its install
        builds the draws first, so icons imply draws). Sprites off: the
        draws derive from the row columns in about 0.1 ms, so every level is
        resident, except inside a mutate-now window, where a level not built
        before it would walk the live lists (and one built in it is dropped
        at invalidate_units()): only level 0 and levels built earlier count.

        "Absent" is a complete staleness test here (unlike level_warm_job(),
        which needs self._unit_gen to distinguish "never built" from
        "invalidated mid-warm", see __init__'s own comment on that counter):
        a chunk warm only ever calls this synchronously, with no warm in
        flight to have invalidated anything between the check and the
        get_chunk() call it gates."""
        if not self.with_units:
            return True
        if self.sprites_enabled:
            return mip in self._level_icon_layers
        if not self._in_unit_window():
            return True
        return mip == 0 or (mip in self._level_draws and ("draws", mip) not in self._window_built)

    def set_sprites_enabled(self, enabled: bool) -> None:
        """Turns real .sld icons on or off on a LIVE cache (P3-g7).

        IsoChunkCache.set_sprites_enabled()'s shape, mip loop included --
        Flat has a real ladder, so that class is the precedent here rather
        than g6's single-level SlopedChunkCache, despite it being the more
        recent one. See it for why each step is load-bearing: the resident
        set must be captured BEFORE invalidate_region() empties the cache,
        and the resident levels are warmed eagerly so the 0.5-4.1s cold .sld
        decode happens under the caller's wait cursor rather than inside the
        next Qt paint with a normal cursor."""
        if enabled == self.sprites_enabled:
            return
        self.sprites_enabled = enabled
        resident = sorted({key[0] for key in self._cache}) or [0]
        self._refresh_unit_sources()
        for mip in resident:
            self._level_icons(mip)
        self.invalidate_region((0, 0, *self.canvas_dims(0)), terrain_changed=False)

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """Unchanged by P3-g7, and worth saying so rather than leaving the
        next reader to check: an icon is contain-fitted into its own footprint
        rect and so can never leave it, while Flat's canvas is exact by
        construction. g6's canvas_dims()-widening fix for Sloped, and its
        MapView/MapCanvasItem.refresh_canvas_dims() follow-up, have no
        analogue here."""
        mm = self.scenario.map_manager
        tile_px = self._mip_tile_px[mip]
        return mm.map_width * tile_px, mm.map_height * tile_px

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        # Resolved first so their level builds stay level events, outside the phase.
        unit_draws = self._level_unit_draws(mip)
        icons = self._level_icons(mip)
        with perf_trace.repaint_phase("composite"):
            return render_flat.composite_rect_flat(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                self._mip_tile_px[mip],
                unit_draws=unit_draws,
                with_units=self.with_units,
                icons=icons,
                layers=self.layers,
                grid=self.grid,
                terrain_ids=self._terrain_ids,
                layer_ids=self._layer_ids,
                key_map=self._blend_key_map,
                farm_tiles=self._farm_tiles,
            )


MAX_PICK_PLANES = 4
"""How many Sloped chunk pick planes (SlopedChunkCache._pick_plane) stay
memoized at once. Sized for the access pattern, not by feel: a cursor is in
one chunk at a time and a drag along a chunk seam alternates between two, so
2 already removes the thrash case; 4 covers a diagonal drag through a
four-chunk corner without letting the memo grow into a second full-canvas
cache. At DEFAULT_CHUNK_PX=512 that is 512*512*4 = 1 MB per plane, 4 MB
capped -- negligible against the colour cache's whole-canvas budget."""

PICK_PLANE_PATCH_MAX_FRACTION = 0.35
"""How much of one resident pick plane an edit's bbox may cover before
SlopedChunkCache.patch() drops that plane outright instead of rewriting the
intersected sub-rect in place (Batch B step B8). At or below the fraction:
patch. Above it: drop, and let _pick_plane() rebuild on demand.

Patching costs its share of a full rebuild EAGERLY, always; dropping costs a
full rebuild LAZILY, and only if that plane is read again.

Measured, not chosen. tools/bench_pick_plane_patch.py section 1 on
F7_2_Dos Pilas (480x480, tile_px=32, chunk_px=512) times each sub-rect
against a full-plane rebuild taken right beside it (8.4-8.8ms):

    square sub-rect   10% area -> 20% cost   25% -> 32%   40% -> 54%
                      50% -> 62%   75% -> 83%   90% -> 98%   100% -> 98%
    full-width band   10% area -> 32% cost   25% -> 42%   50% -> 62%
                      75% -> 84%   90% -> 92%

There is no crossover below 100% to sit on: cost stays under the full
rebuild all the way up, because the only fixed term is the per-call
candidate walk plus the np.full, worth about 10% of a plane for a square
sub-rect and about 24% for a band (a band spans every candidate column, so
it pays more border per pixel). So the value comes from a budget instead:
one edit's eager pick-plane work should stay inside the single full rebuild
dropping already costs, since the cursor's own plane is read on the very
next mouseMoveEvent and for that one plane dropping is not really lazy. A
drag along a chunk seam alternates between TWO resident planes (see
MAX_PICK_PLANES above). Interpolating the rows either side of 0.35 puts one
patched plane at 47% (square) to 51% (band) of a full rebuild, so two of
them cost 0.94 to 1.01 of one and the seam case breaks even at worst, while the common mid-plane small-brush edit at ~0.3 of a
plane is a clear win. A wide brush spreads 0.6 to 0.8 over three or four
planes, lands above the fraction, and keeps exactly today's drop behaviour
rather than paying ~3x eagerly for planes that may never be read."""


class SlopedChunkCache(_LeveledChunkCache):
    """Qt-free LRU cache of composited Sloped-mode canvas chunks -- Phase 6's
    Track C3 counterpart to IsoChunkCache/
    FlatChunkCache. Same grid/LRU mechanics (_ChunkCacheBase), composited
    via composite_rect_sloped() instead of composite_rect_iso()/
    composite_rect_flat() -- a chunk's pixels never depend on which OTHER
    chunks happen to be cached or in what order they were requested, the
    same correctness bar tools/verify_iso_chunks.py/tests/
    test_flat_chunks.py hold their own compositors to (see tests/
    test_sloped_chunks.py for this class's own version of those checks).

    On the shared lazy per-level machinery (_LeveledChunkCache, TASK-031.59):
    each level's unit layers build in _level() on the first composite after
    a gen bump and are spliced in place by elevation and unit edits while
    current, as Stepped's are. The hooks below are where Sloped differs: a
    level's corner_rise feeds its sprite walk, re-anchor, unit pack and
    compositor, and its headroom widens every building bbox at that level.

    The mip ladder is Stepped's mip_projections_for() set, unfiltered: the
    coarser levels since TASK-031.60, the finer ones since TASK-146 (GH #131:
    baked grid ink scales with the shown level's residual magnification, so
    the two elevation views agree only on one ladder). Each level's corner_rise and headroom are
    recomputed at its own proj (_level_rise()), never scaled from level 0, so
    a level depends on elevations and its own proj only, never on level 0's
    array, at the same cost as scaling. (At exact power-of-two levels a
    recomputed corner_rise equals level 0's scaled down under every rule, so
    only a spy tells them apart; the native tier at 3/2, TASK-599, need not.) Level 0 shares
    self.corner_rise and _target_headroom, which stay eager and scene-space
    (pick planes, overlays, unit_pick); reading them never builds a level.

    proj must carry corner_headroom_steps=1 (see sloped_elevations_and_
    proj()) -- built by the caller and passed in, matching IsoChunkCache's
    own convention of taking proj as a constructor parameter rather than
    building it itself. The ladder derives the steps from proj itself, so a
    headroom-0 proj gets a headroom-0 ladder."""

    style = "sloped"
    _has_unit_pack = True
    _sloped = True
    _level_type = _SlopedLevel

    def __init__(
        self,
        scenario: LoadedScenario,
        elevations: np.ndarray,
        corner_rise: np.ndarray,
        proj: iso_geometry.IsoProjection,
        tile_px: int,
        chunk_px: int = DEFAULT_CHUNK_PX,
        max_chunks: int | None = None,
        with_units: bool = True,
        unit_filter: UnitFilter = UnitFilter(),
        sprites: bool = SPRITES_ENABLED,
        layers: LayerState = DEFAULT_LAYERS,
    ):
        self.scenario = scenario
        self.elevations = elevations
        # Superseded immediately by _refresh_source_caches()' own re-derivation
        # below, which owns this array from here on (see its docstring). Kept
        # as a parameter because every caller already holds the value and the
        # constructor stays parallel to elevations/proj, but a caller cannot
        # make it disagree with elevations by passing a mismatched pair.
        self.corner_rise = corner_rise
        self.proj = proj
        self.tile_px = tile_px
        self.with_units = with_units
        self.unit_filter = unit_filter
        # Must be set before _refresh_source_caches() below, since it reads
        # this to decide whether to build a sprite layer at all -- same
        # ordering trap IsoChunkCache.__init__ documents for itself.
        self.sprites_enabled = sprites
        # Same ordering constraint: _level() reads the build-time layer
        # fields (_BUILD_TIME_LAYER_FIELDS) when it builds the sprite layer.
        self.layers = render_key(layers)
        self._pick_planes: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        assert proj.tile_px == tile_px, f"proj.tile_px {proj.tile_px} != tile_px {tile_px}"
        mm = scenario.map_manager
        projs = iso_geometry.mip_projections_for(
            mm.map_width, mm.map_height, proj, settings.get_elev_step_pct(),
            corner_headroom_steps=proj.corner_headroom_px // proj.elev_step,
            max_tile_px=render_paint.finest_tile_pixels(), native_tier_max=render_paint.native_tier_max(),
        )
        self._init_levels(projs)
        # Both canvas_dims() answers (sprites off and on) must scale exactly, as IsoChunkCache asserts.
        for lvl in self._levels.values():
            for dims in (lambda p: (p.canvas_w, p.canvas_h), render_bbox._canvas_pixel_dims):
                (rw, rh), (lw, lh) = dims(proj), dims(lvl.proj)
                assert lw * proj.tile_px == rw * lvl.tile_px and lh * proj.tile_px == rh * lvl.tile_px, (
                    f"level tile_px={lvl.tile_px}'s canvas {(lw, lh)} isn't an exact mip of {(rw, rh)}"
                )
        self._init_terrain_ids()
        self._record_flag_hosts()
        self._record_wall_verdict()
        self._refresh_source_caches()
        self._init_max_chunks(chunk_px, max_chunks)

    def _level_rise(self, lvl: _SlopedLevel) -> _SlopedLevel:
        """lvl with its corner_rise and headroom current: level 0's are
        self.corner_rise and _target_headroom; another level's are recomputed
        at its own proj, once per rebinding of self.corner_rise. Lazy, so an
        edit pays only for the levels it splices or composites: a deferred
        level's sync check (_level_synced()) never computes it."""
        if lvl.rise_source is not self.corner_rise:
            if lvl is self._levels[0]:
                lvl.corner_rise, lvl.headroom = self.corner_rise, self._target_headroom
            else:
                # Rebound, never written in place: prepare_sloped holds the array without copying it.
                rise = iso_geometry.corner_rise_px(self.elevations, lvl.proj, rule=render_sloped.SLOPE_CORNER_RULE)
                lvl.corner_rise, lvl.headroom = rise, render_sloped._unit_rise_headroom_px(rise, self.elevations, lvl.proj)
            lvl.rise_source = self.corner_rise
        return lvl

    def _level_synced(self, lvl: _IsoLevel) -> bool:
        # A stale rise_source means _level_rise() would rebind the heights: unsynced, so skip computing them.
        return lvl.rise_source is self.corner_rise and super()._level_synced(lvl)

    def _level_corner_rise(self, lvl: _IsoLevel) -> np.ndarray | None:
        return self._level_rise(lvl).corner_rise

    def _level_fold(self) -> np.ndarray | None:
        return self.fold

    def _level_heights(self, lvl: _IsoLevel) -> np.ndarray:
        return self._level_rise(lvl).corner_rise

    def _level_extra_top(self, lvl: _IsoLevel) -> int:
        return self._level_rise(lvl).headroom

    def _refresh_corner_rise(self) -> None:
        """corner_rise rebuilt from self.elevations (a new array, never
        written in place) and the headroom recomputed against it. The
        rebinding is what makes every other level re-derive its own in _level_rise().
        The fold (iso_geometry.tile_folds(), TASK-008) is rebuilt with it: it
        reads elevations only, so one array serves every level, and rebinding
        it beside corner_rise keeps every `heights is` staleness check exact."""
        self.fold = iso_geometry.tile_folds(self.elevations)
        self.corner_rise = iso_geometry.corner_rise_px(self.elevations, self.proj, rule=render_sloped.SLOPE_CORNER_RULE)
        self._target_headroom = render_sloped._unit_rise_headroom_px(self.corner_rise, self.elevations, self.proj)

    # Level 0's layers, each read through _level(0) so it forces the lazy build first.
    @property
    def sprites(self) -> render_sprites.SpriteLayer | None:
        return self._level(0).sprites

    @property
    def bystander_grid(self) -> render_bbox.BystanderGrid | None:
        return self._level(0).bystander_grid

    @property
    def _headroom(self) -> int:
        return self._level(0).extra_top

    @property
    def _unit_pack(self) -> native_composite.UnitPack | None:
        return self._level(0).unit_pack

    @property
    def _sprite_memo(self) -> render_sprites.SpriteMemo | None:
        return self._level(0).memo

    def _refresh_source_caches(self, elevation_changed: set | None = None, splice_levels=None) -> None:
        """Refreshes corner_rise and the unit sources: units_by_tile eagerly,
        the per-level unit layers lazily or by splice (_LeveledChunkCache).

        building_bboxes reuses _building_bboxes_iso()/_unit_screen_bbox_
        iso() rather than a Sloped-specific reimplementation, but as of
        Track C5's Step 2 it must pass extra_top_px: Sloped's own proj is
        geometrically identical to Stepped's for the same map (see
        IsoProjection.corner_headroom_px's own comment), yet a unit there
        no longer sits at elevation * elev_step -- it sits at
        iso_geometry.unit_rise_px(), which under SLOPE_CORNER_RULE = "max"
        is never lower and can be higher. The "bystander" widening this
        feeds is an accepted coarse superset in the OVER direction only
        (composite_rect_iso()'s own docstring: "a union bbox flagging a
        tile as a bystander slightly more often than the tightest possible
        test would is a no-op extra paint, never a missed one"); under-
        covering is a missed paint, i.e. a stale pixel, so the headroom
        below is derived rather than assumed.

        `headroom` is an exact global bound, not the plan's "one
        elev_step": for every tile, how far its highest own corner sits
        above its own elevation, maxed over the map. One elev_step is what
        that evaluates to under this project's +-1-elevation-neighbour
        invariant, but a LOADED file is not required to satisfy that
        invariant, and this costs only four vectorized array maxes. Zero on
        a flat map, so flat-map byte-identity is untouched. It is
        _target_headroom, recomputed with corner_rise (_refresh_corner_rise());
        a level records the value its building_bboxes were built with as
        _IsoLevel.extra_top (level 0's is the _headroom alias).

        Sloped honours unit_filter exactly as Flat and Stepped do -- and,
        since C5, so does unit picking (unit_pick.pick_unit's sloped
        branch reads this same cache's corner_rise through
        MapView._sloped_cache).

        corner_rise is rebuilt here too (Track C4's Step 3), which is what
        makes Sloped editable at all: before this it was received once at
        construction and never refreshed, so an elevation edit repainted
        tiles against the pre-edit height field. It belongs in THIS method
        specifically, not in the patch() override above it, because this is
        the one hook whose contract is already "run whenever elevations
        could have changed underneath this cache" -- putting it in patch()
        would skip invalidate_region() and any future caller.

        Rebuilt whole rather than per-dirty-ring, and the cost is measured
        rather than assumed: 4.6ms at 480x480, this project's largest map,
        against a 25ms end-to-end sloped patch there (2026-08-21). So it is
        real but not dominant, and a ring-only update -- which the C4 plan
        offers as the alternative -- stays available as a later optimization
        rather than being needed now. It also makes the redundant
        rebuild at construction a non-issue, and buys a real property for it:
        corner_rise can no longer disagree with self.elevations, however the
        constructor was called.

        A level's sprite layer (Track P3-g6) is built against corner_rise
        (_level_corner_rise()), since a sprite's anchor depends on it via
        unit_rise_px(), and kept in step with it: every corner_rise rebuild
        here either splices the current levels or bumps the gen. with_farms
        defaults True (Track C6): a farm drapes over its own footprint tiles'
        warped terrain, matching Stepped, whenever sprites are on; the
        sprites-off case still falls back to the plain mark (see render_sloped.py's
        own farm handling for that residual). merge_sprite_bboxes() runs
        AFTER the headroom-widened building bboxes (_assemble_level()).

        elevation_changed (draw-perf plan Step 3): None means "unknown" --
        construction, or a unit-filter/sprite-toggle refresh via
        _refresh_unit_sources() -- and rebuilds units_by_tile and corner_rise
        and bumps the gen (_refresh_all_units()), so every level rebuilds on
        its next composite. No unit layer is built here.

        A patch() caller instead passes the elevation-changed subset of its
        own edit. 3a applies here too: units_by_tile never rebuilds on a
        patch(), since no terrain or elevation edit can move a unit. An
        empty elevation_changed (a terrain-paint-only edit) skips
        everything below, since none of it depends on anything terrain
        paint touches. A non-empty one always rebuilds corner_rise whole
        (it depends on EVERY changed tile, not just ones under units) and
        recomputes the headroom, then re-anchors only the units that read a
        changed corner rather than rebuilding every unit (the 2026-09-24
        anchor-local splice; the wholesale rebuild was 194-208ms per stroke
        step on June Event with sprites on).

        Sloped reads a unit's own tile's FOUR corners (unit_rise_px), and
        under SLOPE_CORNER_RULE each corner blends the up-to-4 tiles
        touching it, so a changed tile can move the corners of every tile in
        the 3x3 around it. The seeds are exact rather than that dilation:
        elevation_changed (a building bbox reads its own tile's elevation)
        plus every tile with a corner that actually moved
        (_corner_changed_tiles(), which drops a ring tile held up by a higher
        neighbour), plus every tile whose fold flipped (_fold_changed_tiles():
        a fold reads the second-highest elevation at each corner, which
        corner_rise does not carry, so it can flip with no corner moving),
        plus any unit sharing a tile with one (_elevation_splices()'
        component, re-derived unchanged).
        They are re-anchored on each current level against that level's NEW
        corner_rise (_level_rise(), recomputed at its own proj) and checked
        against its own headroom by _refresh_level_units(), so sprites and
        corner_rise cannot drift apart; a level outside splice_levels defers
        them to its pending set instead. The seeds and the splice component
        are computed once, from level 0: tile space, and exact across levels
        under `max`, the live SLOPE_CORNER_RULE (TASK-031.60). Under `average`
        they would be a superset only toward a coarser level: a finer level
        resolves a corner move level 0's floor rounds away, so a seed could be
        missed there (TASK-146; not live while the rule is `max`). A component
        past _SLOPED_ELEV_SPLICE_MAX_UNITS (elevation_splice_cap()) bumps the
        gen instead, with one elev_splice_refused token, and each level
        rebuilds once on its next composite (TASK-031.28: an elevation
        paste's or undo's unit step no longer builds the layers eagerly first).

        The headroom feeds every building's bbox, not just the re-anchored
        ones, and it moves more often than flat <-> non-flat: the copied
        propagation's gap-fill branch can leave a 2-level diagonal step
        (2026-10-08 probe, headroom 8 to 16 px). Sprites never read it and
        the unit pack never reads a bbox, so a headroom change after the
        splice rebuilds only the bbox layer (_splice_level()), an empty
        component included, and the pack is still refreshed."""
        if elevation_changed is None:
            self._refresh_corner_rise()
            self._refresh_all_units()
            return
        if not elevation_changed:
            return
        old_corner_rise, old_fold = self.corner_rise, self.fold
        self._refresh_corner_rise()
        if not self.with_units:
            return
        seeds = elevation_changed | _corner_changed_tiles(old_corner_rise, self.corner_rise)
        self._refresh_level_units(seeds | _fold_changed_tiles(old_fold, self.fold), splice_levels)

    def elevation_splice_cap(self) -> int:
        return _SLOPED_ELEV_SPLICE_MAX_UNITS

    def elevation_seed_tiles(self, elevation_changed: set, new_elev: dict) -> set:
        # _refresh_source_caches()' exact seeds, from a corner_rise of the post-edit elevations.
        elevations = self.elevations.copy()
        for (x, y), level in new_elev.items():
            elevations[y, x] = level
        new_cr = iso_geometry.corner_rise_px(elevations, self.proj, rule=render_sloped.SLOPE_CORNER_RULE)
        seeds = elevation_changed | _corner_changed_tiles(self.corner_rise, new_cr)
        return seeds | _fold_changed_tiles(self.fold, iso_geometry.tile_folds(elevations))

    def refresh_wholesale(self, dirty_indices, flatten: bool = False) -> bool | None:
        """IsoChunkCache.refresh_wholesale() for Sloped: the elevations written
        in place, then, when any moved, corner_rise and the headroom rebuilt
        and the gen bumped, so the next level build re-resolves from the
        sprite memo (not _refresh_source_caches(), so units_by_tile is not
        walked again). The whole-canvas eviction re-reads the terrain ids and
        clears the pick planes. `flatten` is Stepped-only and ignored here."""
        moved = _write_dirty_elevations(self.scenario, self.elevations, self.proj, dirty_indices)
        if moved is None:
            return None
        self._mutation_epoch += 1
        if moved:
            self._refresh_corner_rise()
            self._source_gen += 1
        self.invalidate_region((0, 0, *self.canvas_dims(0)))
        return moved

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        """(width, height) in canvas pixels -- proj.canvas_w/canvas_h alone,
        NOT render_terrain_sloped_with_proj()'s own padded allocation
        (that function adds Stepped's skirt_headroom formula on top,
        purely so a flat map's output is byte-identical to render_terrain_
        iso's -- see its own docstring). Sloped paints no skirts (see
        corner_rise_px's own docstring), so real content never needs that
        extra padding; get_chunk()'s existing high-edge clip already
        handles this cache's canvas being smaller than that function's.
        A pixel that rounds just past this tight boundary at the very top
        edge (corner_headroom_px's own float-rounding safety margin, see
        IsoProjection's comment) is silently dropped by _clipped_paint,
        the same accepted tradeoff that function's own docstring documents
        for its scratch-canvas call site -- not new here, not a correctness
        gap this class introduces.

        When sprites_enabled (Track P3-g6), returns render_bbox._canvas_pixel_
        dims(self.proj) instead -- the same skirt-padded height Stepped's
        own buffer uses -- rather than the tight bound above. Sloped's tight
        canvas fits terrain alone; a sprite reaches further than that (see
        _sprite_reach_px), and unlike Stepped, Sloped has no skirt margin to
        absorb the difference. This does not eliminate sprite-reach
        clipping -- Stepped doesn't either, see that function's own accepted
        upward gap -- it brings Sloped to the same tolerance Stepped
        already has, instead of a new, Sloped-only regression. set_sprites_
        enabled()'s invalidate_region() call is what makes the newly-
        visible strip actually get painted once this answer changes.

        LEVEL `mip` pixels, from the level's own proj, read directly (never
        through _level()); __init__ asserts both answers scale exactly."""
        proj = self._levels[mip].proj
        if self.sprites_enabled:
            return render_bbox._canvas_pixel_dims(proj)
        return proj.canvas_w, proj.canvas_h

    def _composite_rect(self, mip: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        lvl = self._level(mip)
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units(lvl, mip)
        corner_rise = self._level_corner_rise(lvl)
        with perf_trace.repaint_phase("composite"):
            return render_sloped.composite_rect_sloped(
                self.scenario,
                x0,
                y0,
                x1,
                y1,
                corner_rise,
                lvl.proj,
                lvl.tile_px,
                self.units_by_tile,
                lvl.building_bboxes,
                self.with_units,
                sprites=lvl.sprites,
                bystander_grid=lvl.bystander_grid,
                layers=self.layers,
                grid=self.grid,
                unit_pack=unit_pack,
                terrain_ids=self._terrain_ids,
                layer_ids=self._layer_ids,
                key_map=self._blend_key_map,
                fold=self.fold,
            )

    def _prepare_rect(
        self, mip: int, x0: int, y0: int, x1: int, y1: int, heights: np.ndarray | None = None
    ) -> tuple | None:
        # heights is always None here: corner_rise is replaced on refresh, never written in place.
        lvl = self._level(mip)
        # Resolved first so a pack build stays its own level event, outside the phase.
        unit_pack = self._native_units(lvl, mip)
        corner_rise = self._level_corner_rise(lvl)
        with perf_trace.repaint_phase("prepare"):
            return render_bbox.prepare_rect_native(
                True, self.scenario, x0, y0, x1, y1, corner_rise, lvl.proj, lvl.tile_px,
                lvl.building_bboxes,
                self.with_units, lvl.bystander_grid, self.layers, self.grid, unit_pack,
                terrain_ids=self._terrain_ids, layer_ids=self._layer_ids, key_map=self._blend_key_map,
                fold=self.fold,
            )

    def _pick_plane(self, cx: int, cy: int) -> np.ndarray:
        """Chunk (cx, cy)'s int32 tile-id plane, built on demand and
        memoized -- deliberately held OUTSIDE self._cache and outside
        self._cache_bytes, per the C plan's decision 7.

        The reason is memory, and it was decided against a measured cost:
        Track B-C already accepted a 743MB -> 1127MB RSS regression, and
        keeping an int32 plane resident alongside every cached RGB chunk
        would add ~33% on top of that (4 bytes/px against 3) for no
        correctness gain -- a pick plane is only ever needed for the ONE
        chunk under the cursor, where an RGB chunk is needed for every
        visible one. So this is a tiny separate LRU, not a second channel
        on the main one. Moving the mouse within a chunk is free; crossing
        a chunk boundary costs one ID-only recomposite, which the Gate 2
        benchmark (tools/bench_sloped_patch.py) measures separately from
        the colour path precisely because it is much cheaper.

        Same high-edge clip as get_chunk(), so a plane straddling the
        canvas edge is the same ragged shape its colour chunk is."""
        key = (cx, cy)
        cached = self._pick_planes.get(key)
        if cached is not None:
            self._pick_planes.move_to_end(key)
            return cached

        canvas_w, canvas_h = self.canvas_dims()
        x0, y0 = cx * self.chunk_px, cy * self.chunk_px
        x1, y1 = min(x0 + self.chunk_px, canvas_w), min(y0 + self.chunk_px, canvas_h)
        plane = render_sloped.composite_ids_rect_sloped(
            self.scenario, x0, y0, x1, y1, self.corner_rise, self.proj, self.tile_px
        )
        self._pick_planes[key] = plane
        self._pick_planes.move_to_end(key)
        while len(self._pick_planes) > MAX_PICK_PLANES:
            self._pick_planes.popitem(last=False)
        return plane

    def pick_tile(self, sx: int, sy: int) -> tuple[int, int] | None:
        """The (tile_x, tile_y) whose sloped surface covers reference-canvas
        pixel (sx, sy), or None for a background pixel or one off-canvas --
        Track C4's answer to "what did the user just click on", and the
        Sloped counterpart to iso_geometry.screen_to_tile().

        Reference-canvas pixels, which is also scene space: mip decision D2
        pins scene space to the reference level permanently, so viewer.py
        can pass a scene position through unconverted. The pick planes are
        level 0's whatever level is on screen (TASK-031.60, TASK-146): at a
        coarser level the visible pixels and the plane can disagree by about
        one scene pixel at tile edges, as Stepped's analytic pick does; at a
        finer level by under one scene pixel, since the finer level resolves
        detail level 0 rounds away.

        Worth stating because it is the opposite of what the missing-feature
        history suggests: this is STRICTLY BETTER than Stepped's
        screen_to_tile(), which returns None on any skirt-face pixel (a
        documented accepted residual -- there is no exact analytic inverse
        for a skirt). Sloped paints no skirts at all, so it has no such
        hole: every pixel inside the ground outline resolves to exactly one
        tile, which tests/test_sloped_pick.py asserts exhaustively."""
        canvas_w, canvas_h = self.canvas_dims()
        if not (0 <= sx < canvas_w and 0 <= sy < canvas_h):
            return None
        cx, cy = sx // self.chunk_px, sy // self.chunk_px
        plane = self._pick_plane(cx, cy)
        tile_id = int(plane[sy - cy * self.chunk_px, sx - cx * self.chunk_px])
        if tile_id == render_sloped.PICK_ID_NONE:
            return None
        map_w = self.scenario.map_manager.map_width
        return tile_id % map_w, tile_id // map_w

    def patch(
        self, bbox: tuple[int, int, int, int], elevation_changed: set | None = None, levels=None, rebuild_levels=None
    ) -> None:
        """"Patch, don't drop" for the pick planes too (Batch B step B8),
        the same contract base patch() already holds for colour chunks: a
        resident plane the bbox only partly covers gets that sub-rect
        recomposited and slice-assigned back in place, so the drag's very
        next mouseMoveEvent reads a valid plane instead of paying a full
        512-squared rebuild. Only the planes the bbox covers more than
        PICK_PLANE_PATCH_MAX_FRACTION of are still dropped; see that
        constant for the measurement behind the split. Non-resident planes
        need no action at all, for base patch()'s own reason: _pick_plane()
        always builds fresh against current state.

        Three things about the ordering below are load-bearing:

        An empty elevation_changed returns without touching any plane. A
        plane's ids depend only on corner_rise, proj, tile_px and the map
        dims, never on terrain ids or units, and _refresh_source_caches()
        provably leaves corner_rise alone on that value, so a
        terrain-paint-only edit cannot move a single id.
        elevation_changed=None still means "unknown" and takes the full
        path.

        The rewrite happens AFTER super().patch(), not before it like the
        old drop did, because corner_rise, the plane's one mutable input, is
        rebuilt inside it via _refresh_source_caches(). Rasterizing first
        would paint pre-edit geometry into a plane that then looks current.
        If super().patch() raises, every plane is dropped and the exception
        re-raised: a half-updated corner_rise with warm planes is
        the one state that must not survive. Each survivor is re-get()'d
        afterwards too, since the LRU may have evicted it in between, and
        deliberately WITHOUT move_to_end(): a patch is not a use, and
        promoting it would evict a plane the cursor is about to read.

        Under-coverage would be a silent wrong click target rather than a
        crash (pick_tile feeds unit_pick._pick_unit_sloped, where a 1x1
        unit's membership test is the terrain tile), so it is worth stating
        why it cannot happen. _render_tile_sloped_ids inherits its extent
        from _render_tile_sloped by construction (same corner_rise lookups,
        same d_min, same tile_screen_origin, same sloped_quad_indices entry,
        same _clipped_paint), so a tile's id footprint matches its colour
        footprint exactly; and the bbox is dirty_screen_bbox_sloped's,
        which the colour path already trusts and which is computed
        with_units plus the sprite term, i.e. a strict superset of the
        terrain-only extent. Over-covering is free:
        composite_ids_rect_sloped is a rect-keyed core that starts from a
        fresh PICK_ID_NONE fill, so a rewritten sub-rect is self-clearing
        and needs no ring dilation.

        invalidate_region()/set_unit_filter() keep the wholesale clear:
        neither fires per patched tile, so there's nothing to save by
        narrowing them, and unlike patch() their bbox/call is not on the
        hot per-move path."""
        px0, py0, px1, py1 = bbox
        if px1 <= px0 or py1 <= py0:
            return
        if elevation_changed is not None and not elevation_changed:
            super().patch(bbox, elevation_changed, levels, rebuild_levels=rebuild_levels)
            return

        targets = []
        for (cx, cy), plane in list(self._pick_planes.items()):
            plane_x0, plane_y0 = cx * self.chunk_px, cy * self.chunk_px
            # plane.shape, never chunk_px: _pick_plane() clips an edge plane
            # to canvas_dims(), exactly as base patch() uses chunk.shape.
            plane_x1, plane_y1 = plane_x0 + plane.shape[1], plane_y0 + plane.shape[0]
            ix0, iy0 = max(px0, plane_x0), max(py0, plane_y0)
            ix1, iy1 = min(px1, plane_x1), min(py1, plane_y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            covered = (ix1 - ix0) * (iy1 - iy0) / (plane.shape[1] * plane.shape[0])
            if covered > PICK_PLANE_PATCH_MAX_FRACTION:
                self._pick_planes.pop((cx, cy), None)
            else:
                targets.append((cx, cy, ix0, iy0, ix1, iy1))

        try:
            super().patch(bbox, elevation_changed, levels, rebuild_levels=rebuild_levels)
        except BaseException:
            self._pick_planes.clear()
            raise

        for cx, cy, ix0, iy0, ix1, iy1 in targets:
            plane = self._pick_planes.get((cx, cy))
            if plane is None:
                continue
            plane_x0, plane_y0 = cx * self.chunk_px, cy * self.chunk_px
            sub = render_sloped.composite_ids_rect_sloped(
                self.scenario, ix0, iy0, ix1, iy1, self.corner_rise, self.proj, self.tile_px
            )
            plane[iy0 - plane_y0 : iy1 - plane_y0, ix0 - plane_x0 : ix1 - plane_x0] = sub

    def invalidate_region(self, bbox: tuple[int, int, int, int], levels=None, terrain_changed: bool = True) -> None:
        self._pick_planes.clear()
        super().invalidate_region(bbox, levels, terrain_changed)

    def set_unit_filter(self, unit_filter: UnitFilter) -> None:
        # Defensive, not load-bearing: the pick plane is terrain-only (see
        # _render_tile_sloped_ids), so no filter change can alter an id, and
        # Track C5 left it that way rather than folding units in. Kept so the
        # memo can never outlive a source-state change if that ever varies.
        self._pick_planes.clear()
        super().set_unit_filter(unit_filter)
