#!/usr/bin/env python3
"""Why an arrow-nudge of a unit sharing a tile with another unit costs ~4x a
lone unit's. Run from a DEscape checkout root:

    .venv/bin/python3 tools/probe_shared_tile_nudge.py [--nudges 20] [--mode alt|right]
        [--only 837|solo] [--analyse]

Drives a real offscreen ViewerWindow (install pinned, settings isolated to a
throwaway dir), no trigger held, on Dos Pilas (32k units), and arrow-nudges
(a) the first player unit, a Map Revealer (837) sharing its tile with a GAIA
Reeds tree (1350), and (b) a lone Longship. Logs every _batch_splice_eligible()
answer with the rejecting tile and occupant, which invalidate_units() branch
ran, and a cProfile of the timed nudges. `--mode right` walks the unit off
the shared tiles, so the same unit flips to the splice path part-way.
`--analyse` runs render_cache._reanchor_units() on copies of the visible
level for (a)'s nudge and diffs the result against the full rebuild's: what
the splice would get wrong if the shared-tile guard were dropped.
Informational only; absolute times drift under load, shares don't.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from testkit import qt_window, settings_isolation

SCENARIO = Path("examples/F7_2_Dos Pilas (648).aoe2scenario")
LOG: list[dict] = []
COUNTS: dict[str, int] = {}


def _install_probes() -> None:
    from descape import render_cache

    orig_batch = render_cache._batch_splice_eligible

    def batch_probe(units_by_tile, changed):
        result = orig_batch(units_by_tile, changed)
        entry = {"result": result, "n": len(changed), "const_ok": [], "blockers": []}
        for s in changed:
            entry["const_ok"].append((s.unit.unit_const, render_cache._const_splice_eligible(s.unit)))
            for tile in sorted(s.changed_tiles):
                others = [u for u, _ in units_by_tile.get(tile, ()) if u is not s.unit]
                if others:
                    side = ("old" if tile in s.old_tiles else "") + ("new" if tile in s.new_tiles else "")
                    entry["blockers"].append((tile, side, [(u.unit_const, u.x, u.y) for u in others]))
            entry["tiles"] = (s.old_tiles, s.new_tiles)
        LOG.append(entry)
        return result

    render_cache._batch_splice_eligible = batch_probe

    def count(cls, name):
        original = getattr(cls, name)

        def wrapped(self, *a, **k):
            key = f"{cls.__name__}.{name}"
            COUNTS[key] = COUNTS.get(key, 0) + 1
            return original(self, *a, **k)

        setattr(cls, name, wrapped)

    for cls in (render_cache.IsoChunkCache, render_cache.SlopedChunkCache):
        count(cls, "_refresh_source_caches")
        count(cls, "invalidate_units")
    count(render_cache.IsoChunkCache, "_splice_levels")


def _solo_entry(loaded, entries):
    """A player unit with no shared tile now or one tile to either side, not a
    wall/variant const, with a graphic (same picker as probe_w4_ref_index)."""
    from descape import render, unit_sprites

    mm = loaded.map_manager
    by_tile = render._units_by_tile(loaded)
    walls = unit_sprites.wall_connector_consts()
    for e in entries:
        const = e.unit.unit_const
        if e.player_id == 0 or const in walls or unit_sprites.rotation_variant_eligible(const):
            continue
        if unit_sprites.graphic_map().get(const) is None:
            continue
        tiles = set()
        for dx in (-1, 0, 1, 2):
            got = render.occupied_tiles_for(const, e.unit.x + dx, e.unit.y, mm.map_width, mm.map_height)
            if got is None:
                break
            tiles.update(got)
        else:
            if all(all(u is e.unit for u, _ in by_tile.get(t, ())) for t in tiles):
                return e
    return None


def _nudges(window, n: int, mode: str) -> list[float]:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    out = []
    for i in range(n):
        key = Qt.Key_Right if (mode == "right" or i % 2 == 0) else Qt.Key_Left
        t0 = time.perf_counter()
        QTest.keyClick(window.map_view, key)
        QApplication.processEvents()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _norm_draws(draws):
    return [(getattr(o, "name", None) or getattr(o, "file_name", None) or type(o).__name__, x, y) for o, x, y in draws]


def _visible_level(cache, window):
    target = window.map_view.viewport_chunk_target()
    mip = target[0] if target else 0
    if hasattr(cache, "_levels"):
        lvl = cache._level(mip)  # rebuilds if stale, so it reflects live scenario state
        return mip, lvl.building_bboxes, lvl.sprites, lvl.proj, None, 0
    return 0, cache.building_bboxes, cache.sprites, cache.proj, cache.corner_rise, cache._headroom


def _analyse_splice_on_copies(window, entry) -> None:
    """What the splice would have produced for this unit's nudge, against what
    the full rebuild produced, on copies. No descape state is mutated except
    by the real nudges themselves."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from descape import render, render_cache

    cache = window._cache
    unit, pid = entry.unit, entry.player_id
    index = window._unit_list_index(pid, unit)
    mm = window.scenario.map_manager
    for key in (Qt.Key_Right, Qt.Key_Left):
        old_own, old_tiles = window._unit_footprint(unit)
        mip, bb, sprites, proj, corner_rise, extra = _visible_level(cache, window)
        bb_copy = dict(bb)
        ubt_copy = {t: list(v) for t, v in cache.units_by_tile.items()}
        pre_sprites = sprites
        QTest.keyClick(window.map_view, key)
        QApplication.processEvents()
        new_own, new_tiles = window._unit_footprint(unit)
        splice = render_cache.UnitSplice(pid, index, unit, old_own, new_own, tuple(old_tiles), tuple(new_tiles))
        touched = set(old_tiles) | set(new_tiles) | {old_own, new_own}
        print(f"  -- splice-on-copies, {'Right' if key == Qt.Key_Right else 'Left'}: {old_own}->{new_own} level mip {mip}")
        print(f"     _splice_eligible on pre-edit units_by_tile: {render_cache._splice_eligible(ubt_copy, splice)}")
        for t in sorted(touched):
            print(f"     pre  units_by_tile{t}: {[(u.unit_const, u.x, u.y) for u, _ in ubt_copy.get(t, ())]}")
        # Individual contributions at the live level
        for u2, _c in {id(u): (u, c) for t in touched for u, c in ubt_copy.get(t, ())}.values():
            p2 = next(p for p, lst in enumerate(window.scenario.unit_manager.units) if any(x is u2 for x in lst))
            i2 = window._unit_list_index(p2, u2)
            contrib = render._resolve_unit_sprite(
                window.scenario,
                proj,
                cache.elevations,
                cache.unit_filter,
                corner_rise,
                {},
                cache.layers.farm_overlay,
                p2,
                i2,
                u2,
                cache.layers.tree_scale,
                cache.layers.hero_glow,
            )
            bbx = render._building_bbox_for(u2, mm.map_width, mm.map_height, proj, cache.elevations, extra)
            print(
                f"     contribution const {u2.unit_const} p{p2} @({u2.x},{u2.y}): sprite="
                f"{None if contrib is None else {k: _norm_draws(v) for k, v in contrib.by_anchor.items()}} "
                f"building_bbox={bbx}"
            )
        # The splice on copies.
        ubt_spliced = {t: list(v) for t, v in ubt_copy.items()}
        render_cache._splice_units_by_tile(ubt_spliced, window.scenario, cache.unit_filter, splice)
        spliced_sprites = render_cache._reanchor_units(
            bb_copy,
            pre_sprites,
            window.scenario,
            proj,
            cache.elevations,
            cache.unit_filter,
            corner_rise,
            extra,
            [splice],
            {},
            cache.layers.farm_overlay,
            cache.layers.tree_scale,
            cache.layers.hero_glow,
        )
        _mip, bb_real, sprites_real, *_ = _visible_level(cache, window)
        diffs = [
            ("building_bboxes", k, bb_copy.get(k), bb_real.get(k))
            for k in sorted(set(bb_copy) | set(bb_real))
            if bb_copy.get(k) != bb_real.get(k)
        ]
        if spliced_sprites is not None:
            for name in ("by_anchor", "bboxes", "farm_by_tile"):
                a, b = getattr(spliced_sprites, name), getattr(sprites_real, name)
                for k in sorted(set(a) | set(b)):
                    va, vb = a.get(k), b.get(k)
                    if name == "by_anchor":
                        va = va and _norm_draws(va)
                        vb = vb and _norm_draws(vb)
                    if va != vb:
                        diffs.append((name, k, va, vb))
            real_skip, splice_skip = set(sprites_real.skip_ids), set(spliced_sprites.skip_ids)
            if real_skip != splice_skip:
                diffs.append(("skip_ids", None, len(splice_skip - real_skip), len(real_skip - splice_skip)))
        for t in sorted(set(ubt_spliced) | set(cache.units_by_tile)):
            a = [id(u) for u, _ in ubt_spliced.get(t, ())]
            b = [id(u) for u, _ in cache.units_by_tile.get(t, ())]
            if a != b:
                diffs.append(
                    (
                        "units_by_tile",
                        t,
                        [(u.unit_const) for u, _ in ubt_spliced.get(t, ())],
                        [(u.unit_const) for u, _ in cache.units_by_tile.get(t, ())],
                    )
                )
        print(f"     splice-vs-rebuild diffs ({len(diffs)}): (field, key, splice, rebuild)")
        for d in diffs[:12]:
            print(f"       {d}")


def _session(label: str, pick: str, nudges: int, mode: str, analyse: bool = False) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import unit_pick
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    window.resize(1600, 1000)
    window.show()
    QApplication.processEvents()
    try:
        window.load_scenario(SCENARIO)
        QApplication.processEvents()
        window.mode_combo.setCurrentText("Units")
        QApplication.processEvents()
        if window.map_view._unit_index is None:
            window._rebuild_unit_index()
        entries = window.map_view._unit_index.entries
        entry = next(e for e in entries if e.player_id != 0)
        if pick == "solo":
            entry = _solo_entry(window.scenario, entries) or entry
        window._selection = [unit_pick.unit_key(entry.player_id, entry.unit)]
        window._refresh_selection_view()
        # Centre the view on the unit so the patched bbox is on screen.
        window.map_view.center_on_tile(int(entry.unit.x), int(entry.unit.y))
        window.map_view.setFocus()
        QApplication.processEvents()
        print(f"  viewport_chunk_target {window.map_view.viewport_chunk_target()} | centre tile {window.map_view.viewport_centre_tile()}")
        if pick == "first" and analyse:
            _analyse_splice_on_copies(window, entry)
        print(
            f"\n=== {label}: const {entry.unit.unit_const} player {entry.player_id} at "
            f"({entry.unit.x}, {entry.unit.y}) | style {window._render_style} | "
            f"sprites {window._cache.sprites_enabled} | cache {type(window._cache).__name__} | "
            f"trigger overlay ref {window._trigger_overlay_ref} | mode {mode}"
        )
        _nudges(window, 2, "alt")  # warm-up, returns home
        LOG.clear()
        COUNTS.clear()
        prof = cProfile.Profile()
        prof.enable()
        totals = _nudges(window, nudges, mode)
        prof.disable()
        print(
            f"  per nudge: median {statistics.median(totals):.1f}ms mean {statistics.fmean(totals):.1f}ms "
            f"min {min(totals):.1f} max {max(totals):.1f} (n={nudges})"
        )
        print(f"  counts over {nudges} nudges: {COUNTS}")
        results = [e["result"] for e in LOG]
        print(f"  _batch_splice_eligible calls {len(LOG)}: True {results.count(True)} False {results.count(False)}")
        print(
            "   per call: "
            + " ".join(
                f"{e['tiles'][0][0] if e['tiles'][0] else None}->{e['tiles'][1][0] if e['tiles'][1] else None}:"
                f"{'T' if e['result'] else 'F'}{[b[0] for b in e['blockers']] if e['blockers'] else ''}"
                for e in LOG
            )
        )
        for i, e in enumerate(LOG[:2]):
            print(f"   [{i}] result={e['result']} const_ok={e['const_ok']} tiles={e['tiles']} blockers={e['blockers']}")
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s)
        st.sort_stats("cumulative").print_stats(30)
        text = s.getvalue()
        body = text[text.find("   ncalls") :]
        print("  cProfile (cumulative, top 30):")
        print("\n".join("    " + line for line in body.splitlines() if line.strip()))
        total = st.total_tt
        print(f"  profile total {total * 1000 / nudges:.1f}ms/nudge")
        for needle in (
            "_patch_unit_edit_cache",
            "invalidate_units",
            "_refresh_source_caches",
            "_units_by_tile",
            "sprite_draws_by_anchor",
            "_composite_rect",
            "patch",
            "_on_unit_references_moved",
            "_splice_levels",
            "_reanchor_units",
            "dirty_screen_bbox_iso",
            "paintEvent",
        ):
            rows = [(k, v) for k, v in st.stats.items() if k[2] == needle]
            for (fn, line, name), (_cc, nc, _tt, ct, _callers) in rows:
                print(f"    {name:28s} {Path(fn).name}:{line} calls {nc} cum {ct * 1000 / nudges:.1f}ms/nudge ({ct / total * 100:.0f}%)")
    finally:
        window.edit_history.mark_saved()
        window.close()
        QApplication.processEvents()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nudges", type=int, default=20)
    parser.add_argument("--mode", choices=("alt", "right"), default="alt")
    parser.add_argument("--only", choices=("837", "solo"), default=None)
    parser.add_argument("--analyse", action="store_true", help="splice-on-copies diff for the 837 unit")
    args = parser.parse_args()
    qt_window.ensure_qapp()
    settings_isolation.pin_install_path()
    with tempfile.TemporaryDirectory(prefix="probe_shared_tile_") as tmp:
        settings_isolation.isolate_settings(Path(tmp))
        import descape.settings as settings_module

        settings_module._autosave_enabled = False
        _install_probes()
        if args.only in (None, "837"):
            _session("shared-tile unit", "first", args.nudges, args.mode, args.analyse)
        if args.only in (None, "solo"):
            _session("splice-eligible unit", "solo", args.nudges, args.mode)


if __name__ == "__main__":
    main()
