"""Tracks all 45 verify_*.py checks through the pytest migration. Keyed
on nodeids once migrated, not on original_module/
original_check_name -- see Phase 4's coverage guard for why: those two
fields point at files/functions that stop existing once Phase 3 deletes
the verify scripts, so only the nodeid can still be checked for
collectability at that point.

The original migration inventory was 44, not the 42 a plain "^def check"
scan finds: generate_reference_pngs (verify_iso_render.py's 6th checks-list
entry, no check_ prefix, but returns (ok, detail) and counts toward that
script's own pass tally) and verify_roundtrip.py's own main()-level "every
file loads without raising" assertion (a separate try/except from
check_tail_completeness) are two more pass/fail entries hiding outside the
naming convention -- 42 + 1 + 1 = 44. generate_reference_pngs's own entry
records its Ordering-step-4 destination as a tool path, not a nodeid, once
extracted. The two purely-informational latency/chunk-size benches
(bench_incremental_latency, bench_chunk_px) are a separate thing again:
main()'s own "checks" list never counted either as pass/fail, so neither
one ever had a manifest entry to begin with -- they're just extracted into
tools/bench_*.py alongside generate_reference_pngs in that same step.

44 -> 45: the Units-section write path split verify_batch_api.py's
check_unit_edit_not_persisted into check_raw_unit_mutation_does_not_persist
(renamed, same check) and check_unit_edit_persists_through_the_model (new)
once units gained a real write path -- the old check's docstring ("save()
does not persist unit edits") stopped being true unconditionally.

`family` is this file's own bookkeeping for tests/conftest.py's Phase 1
adapter dispatch (which fixtures a generated test needs) -- not part of
the plan's documented 5-field schema, kept here anyway rather than in a
second file so the two can't drift apart:
  none      no scenario file needed at all
  path      one real .aoe2scenario file
  path_tmp  one real file + a scratch directory to write into
  files     the whole corpus file list at once (order/state coupling
            across files, or a random draw shared across the whole run)
  special   doesn't come from a check_* function; wired up by hand in
            tests/test_legacy_adapter.py instead of the generic dispatch

`tier` is the plan's own Tiers vocabulary: a tuple of pytest marker names
(currently only "corpus" and "gui" appear; "slow" is unused until Phase 3
migration data says a specific test needs it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestEntry:
    original_module: str
    original_check_name: str
    family: str
    tier: tuple[str, ...]
    migrated_test_nodeid: str
    prose: str


MANIFEST: list[ManifestEntry] = [
    ManifestEntry(
        "verify_roundtrip", "check_tail_completeness", "path",
        ("corpus",), "tests/test_scenario_io.py::test_tail_completeness_corpus",
        "Re-derives the decompressed body independently and confirms the loader's "
        "trigger_tail, combined with everything parsed before it, reconstructs it exactly. "
        "Migrated as two tests, not one: test_tail_completeness_corpus (recorded here, "
        "parametrized per real corpus file, same behavior as the original script) and "
        "test_tail_completeness (default tier, against the shipped blank template "
        "descape/templates/blank_120x120.aoe2scenario -- new coverage a fresh clone "
        "didn't have before, per the plan's new-test #5).",
    ),
    ManifestEntry(
        "verify_roundtrip", "<main: load_map_and_units per file>", "special",
        ("corpus",), "tests/test_scenario_io.py::test_scenario_loads_without_raising_corpus",
        "main()'s own try/except around load_map_and_units(path) -- 'every file in the "
        "directory loads without raising' is a separate assertion from check_tail_completeness, "
        "not folded into it. Same two-test split as check_tail_completeness above: "
        "test_scenario_loads_without_raising_corpus (recorded here) plus "
        "test_scenario_loads_without_raising (default tier, against the fixture).",
    ),
    ManifestEntry(
        "verify_iso_rect_candidates", "check_real_files", "files", ("corpus",), "",
        "Exact-order equivalence: iso_geometry.tiles_in_screen_rect() must be np.array_equal, "
        "in the same depth_order, to a brute-force full-grid scan, for >=200 random + explicit "
        "edge-case screen rects per real example file.",
    ),
    ManifestEntry(
        "verify_iso_rect_candidates", "check_out_of_range_returns_empty", "none", (), "",
        "A rect entirely outside the canvas (all four directions, on a synthetic 20x20 map) "
        "returns an empty (0, 2) array, not a crash or a spurious candidate.",
    ),
    ManifestEntry(
        "verify_write_path", "check_zero_edit_identity", "path_tmp",
        ("corpus",), "tests/test_write_path.py::test_zero_edit_identity_corpus",
        "Load, write with no changes, and confirm the written file's decompressed body is "
        "byte-identical to the original. Migrated as two tests: test_zero_edit_identity_corpus "
        "(recorded here, parametrized per real corpus file, same behavior as the original "
        "script) and test_zero_edit_identity (default tier, against the shipped blank "
        "template descape/templates/blank_120x120.aoe2scenario -- new coverage per the "
        "plan's new-test #5).",
    ),
    ManifestEntry(
        "verify_write_path", "check_edit_reload", "path_tmp",
        ("corpus",), "tests/test_write_path.py::test_edit_reload_corpus",
        "Paint one tile's terrain, raise one tile's elevation, set another to an absolute "
        "elevation level; write; reload through the real load path; confirm the edits are "
        "exactly what's now on disk. Same two-test split as check_zero_edit_identity above.",
    ),
    ManifestEntry(
        "verify_write_path", "check_history_invariants", "path",
        ("corpus",), "tests/test_write_path.py::test_history_invariants_corpus",
        "EditHistory undo-to-start matches a fresh independent load; redo-to-end matches the "
        "post-edit snapshot; a genuine no-op edit pushes no record; applying a new edit after "
        "an undo() truncates the discarded redo tail. Same two-test split as the other two "
        "checks in this script; complements tests/test_edit_history.py's fake-tile unit tests "
        "with the same invariants against a real, loaded scenario's real TerrainTiles.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_raise_elevation_under_buildings", "path_tmp", ("corpus",), "",
        "Raise the tile under a real building by 1, save, reload, confirm the reloaded "
        "elevation is the pre-edit value + 1 (clamped). Skips (returns None) if the map isn't "
        "square or no player has an on-map building.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_raise_elevation_clamp", "path_tmp", ("corpus",), "",
        "Same as check_raise_elevation_under_buildings but starting the tile at MAX_ELEVATION -- "
        "must stay clamped, not go out of range. Skips under the same conditions.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_raise_elevation_off_map", "path", ("corpus",), "",
        "A unit moved off-map must make raise_elevation_under return False, not raise. Skips "
        "if no player has an on-map building to move off-map.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_neighbors", "path", ("corpus",), "",
        "neighbors() edge/corner behavior: interior tile has 4 (8 diagonal) neighbors; every "
        "corner -- not just (0,0), the opposite corner rules out an xy_to_i wraparound bug too "
        "-- has 2 (3 diagonal); a non-corner edge tile has 3.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_recolor_dirt_near_water", "path_tmp", ("corpus",), "",
        "Recolor every DIRT_2 tile bordering water to BEACH, save, reload: every recolored tile "
        "really did border water and is now BEACH with layer reset, and re-running detection "
        "against the reloaded file finds nothing left to recolor.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_is_water_classification", "none", (), "",
        "Direct is_water() assertions against known TerrainId members, including the SHALLOWS "
        "family (needs its own keyword match separate from 'WATER') and confirming ICE is "
        "deliberately excluded.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_raise_elevation_under_many", "path_tmp", ("corpus",), "",
        "Apply raise_elevation_under_many across every on-map building of whichever player has "
        "the most; every affected tile lands at exactly its pre-edit elevation + amount "
        "(clamped), once per unique tile -- the guarantee a naive per-unit loop lacks.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_raw_unit_mutation_does_not_persist", "path_tmp", ("corpus",), "",
        "Renamed from check_unit_edit_not_persisted (phase 3.5a, stage 5.1): mutate a real "
        "unit's player directly (the banned `unit.player =` setter, not a UnitEditModel), save "
        "with no model, reload: the reloaded unit's player is unchanged -- the containment "
        "guarantee phase 3.5a's write path depends on.",
    ),
    ManifestEntry(
        "verify_batch_api", "check_unit_edit_persists_through_the_model", "path_tmp", ("corpus",), "",
        "New in phase 3.5a (stage 5.1), the other half of the pair above: the same kind of edit "
        "(a player reassignment), made through a UnitEditModel and passed to save(units=...), "
        "does persist -- proving persistence is opt-in per edit, not a per-script flag.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_full_pixel_roundtrip", "none", (), "",
        "Every pixel of every tile's diamond, forward via tile_screen_origin/diamond_indices "
        "and back via screen_to_tile, on synthetic maps with mixed elevation: returns either "
        "its own tile or a genuinely occluding tile with strictly greater depth -- never None "
        "or a tile with lesser-or-equal depth.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_diamond_partition", "none", (), "",
        "At a single flat elevation, every tile's diamond_indices pixels cover the map's iso "
        "silhouette with no gap and no double-cover.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_real_files", "files", ("corpus",), "",
        "Tile-center round-trip for every tile at its real elevation, on every real example "
        "file. Also prints a per-file canvas-megabytes line with no threshold -- a live number "
        "a later doc/Performance table depends on; dropped by pytest's stdout capture in this "
        "Phase 1 form, to be carried forward via record_property when this migrates for real "
        "in Phase 3.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_bounds_containment", "files", ("corpus",), "",
        "Computed canvas dimensions actually contain every tile's full diamond bbox, for every "
        "real example file plus a synthetic 480x480 worst case.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_skirt_geometry", "none", (), "",
        "skirt_quad_indices() output stays within its documented extent and every source index "
        "in range; its two sides' column union exactly equals diamond_indices' own column set "
        "-- the regression guard for the Step 0 comb defect.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_shadow_geometry", "none", (), "",
        "shadow_quad_indices()'s wedge band exactly equals the back neighbour's own exposed "
        "sliver (2-D set equality against an independently-derived oracle -- soundness and "
        "completeness at once), stays within its documented extent including the apex-column "
        "exclusion, is internally unique and disjoint between its two sides and from "
        "diamond/skirt pixels, empties exactly at rise_px >= 2*half_h - 2, and raises "
        "ValueError on a bad side or non-positive rise_px.",
    ),
    ManifestEntry(
        "verify_iso_geometry", "check_ground_outline_corners", "none", (), "",
        "ground_outline_corners() returns the true screen-space extremes of the whole grid's "
        "flat diamond tiling, checked by brute-force scanning every tile's own four corners "
        "independently rather than restating the function's own extremal-tile assumption.",
    ),
    ManifestEntry(
        "verify_iso_render", "check_determinism", "files", ("corpus",), "",
        "Two render_terrain_iso() calls on the same (smallest real) scenario are byte-identical. "
        "Single-file despite the list[Path] signature -- only ever uses files[0] by size.",
    ),
    ManifestEntry(
        "verify_iso_render", "check_full_coverage", "none", (), "",
        "Composited render on synthetic maps has zero gaps and zero leaks past the expected "
        "footprint.",
    ),
    ManifestEntry(
        "verify_iso_render", "check_occlusion_scripted", "none", (), "",
        "Scripted elevation deltas produce the correct occlusion in the compositor.",
    ),
    ManifestEntry(
        "verify_iso_render", "check_pick_oracle", "none", (), "",
        "Pick-oracle correctness for the compositor.",
    ),
    ManifestEntry(
        "verify_iso_render", "check_shadow_clipping", "none", (), "",
        "Every contact-shadow band pixel lands in bounds on the full canvas (the positive "
        "invariant that replaced the old negative-absolute-row premise, unreachable since the "
        "band became a wedge), and the scratch-canvas offset call site still clips without "
        "crash or index wraparound.",
    ),
    ManifestEntry(
        "verify_iso_render", "generate_reference_pngs", "special", ("corpus",),
        "tool:tools/gen_iso_reference_pngs.py",
        "Not a check -- a generator payload that rode along as the 6th entry in main()'s "
        "checks list (no check_ prefix, but returned (ok, detail) and counted toward the pass "
        "tally). Extracted into tools/gen_iso_reference_pngs.py in Ordering step 4; the "
        "Phase 1 scaffolding test (test_iso_render__generate_reference_pngs) and the function "
        "itself were removed from verify_iso_render.py at the same time.",
    ),
    ManifestEntry(
        "verify_iso_units", "check_position", "none", (), "",
        "A unit renders at its own tile's elevation.",
    ),
    ManifestEntry(
        "verify_iso_units", "check_occlusion", "none", (), "",
        "Unit occlusion interleaves correctly with terrain, not a blanket overlay.",
    ),
    ManifestEntry(
        "verify_iso_units", "check_incremental_lone_unit", "none", (), "",
        "Incremental vs. full render match for a lone unit's own tile.",
    ),
    ManifestEntry(
        "verify_iso_units", "check_incremental_building_center", "none", (), "",
        "Incremental vs. full render match for a building's CENTER tile.",
    ),
    ManifestEntry(
        "verify_iso_units", "check_incremental_building_corner", "none", (), "",
        "Incremental vs. full render match for a building's FOOTPRINT CORNER tile.",
    ),
    ManifestEntry(
        "verify_iso_incremental", "check_incremental_and_undo_redo", "path", ("corpus",), "",
        "A scripted sequence of edit ops applied incrementally matches a full re-render at "
        "every step, and undo/redo stay consistent with it.",
    ),
    ManifestEntry(
        "verify_iso_incremental", "check_out_of_range_fallback", "path", ("corpus",), "",
        "An out-of-range incremental edit falls back correctly instead of producing a wrong or "
        "crashing partial redraw.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_stitched_matches_full", "files", ("corpus",), "",
        "IsoChunkCache.render_rect() over the whole canvas is np.array_equal to an independent "
        "render_terrain_iso() call, per real example file.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_arbitrary_rects_match_full", "files", ("corpus",), "",
        "~30 random non-chunk-aligned rects per file (non-zero origin, size not a multiple of "
        "chunk_px, straddling seams) match the corresponding crop of an independent full "
        "render. Seeds one shared np.random.default_rng(RNG_SEED) across the whole file loop -- "
        "per-file parametrization on migration will need a derived per-file seed to keep a "
        "failure reproducible from its node id alone (see Phase 3's determinism note).",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_order_independence", "files", ("corpus",), "",
        "Two fresh caches over the same scenario, chunks fetched in different orders (row-major "
        "vs. shuffled-and-reversed), assemble to byte-identical pixels. Same shared-RNG "
        "determinism note as check_arbitrary_rects_match_full.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_patch_after_scripted_ops", "files", ("corpus",), "",
        "patch() after each of a scripted op sequence (paint/raise/lower/big-jump, then their "
        "undos) re-establishes byte-identity against a fresh full render, per editable file.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_eviction_round_trips", "files", ("corpus",), "",
        "Forcing a chunk out of a small-max_chunks cache and re-fetching it reproduces the "
        "exact same pixels it had before eviction. Single-file (files[0]) despite the "
        "list[Path] signature -- does not loop over the corpus.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_invalidate_region_round_trips", "files", ("corpus",), "",
        "invalidate_region() on a fully-warmed cache evicts the targeted region, and "
        "re-fetching it reproduces a fresh full render. Single-file (files[0]) despite the "
        "list[Path] signature.",
    ),
    ManifestEntry(
        "verify_iso_chunks", "check_composite_rect_iso_elevations_untouched", "files", ("corpus",), "",
        "composite_rect_iso() never mutates its elevations argument -- bit-identical before and "
        "after. Single-file (files[0]) despite the list[Path] signature.",
    ),
    ManifestEntry(
        "verify_iso_viewer_pick", "check_file", "path", ("corpus", "gui"), "",
        "Drives a real ViewerWindow headlessly (QT_QPA_PLATFORM=offscreen): elevation-snapshot "
        "freshness after a live Flat-mode edit, pick round-trip on edited tiles, edit tools "
        "enabled in both Terrain Styles, and a live edit made while already in Stepped mode "
        "(pixels change, snapshot stays fresh, undo restores exactly).",
    ),
    ManifestEntry(
        "verify_copy_paste", "check_file", "path", ("corpus", "gui"), "",
        "Drives a real ViewerWindow headlessly: Copy/Paste disabled on Pan; terrain copy/paste "
        "round trip with layer reset; elevation copy/paste with FORCED neighbor propagation "
        "(source set far enough from the destination's neighbor to trigger "
        "MapManager._elevation_tile_recursion); both undoable; paste disabled/no-op on a "
        "clipboard-kind/tool mismatch. The one script whose CLI deliberately has no examples/ "
        "default -- carried forward by this suite's --scenario-dir option.",
    ),
]

assert len(MANIFEST) == 45, f"expected 45 manifest entries, got {len(MANIFEST)}"
