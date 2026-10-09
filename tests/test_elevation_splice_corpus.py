"""Corpus-tier probe of the elevation splice over shared-tile components
(2026-09-28 plan): random brush-9 raises on real files with real sprites, each
centred on a unit that shares a tile, spliced, then compared for exact
equality with a fresh cache. Modelled on test_convert_splice_corpus.py, whose
state helpers (content-compared draws, unit pack rows) this reuses.

Checked per round: each resident level's building_bboxes and sprite layer and
the native unit pack's rows where the native backend is live. A Stepped round
must not bump the source gen and a Sloped one must not rebuild its unit
layers, except past the component cap or, on Sloped, when the raise moves the
map-wide headroom (both fallbacks by design).

Needs AOE2DE_INSTALL_PATH, since conftest hides the configured install.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from test_convert_splice_corpus import VIEW_PX, _assert_matches_fresh, _footprint, _make_cache, _mips

from descape import asset_source, render, render_cache
from descape.elevation_tools import set_tiles_elevation
from descape.render import dirty_screen_bbox_iso, dirty_screen_bbox_sloped
from descape.render_cache import IsoChunkCache, SlopedChunkCache
from descape.scenario_io import load_map_and_units
from descape.unit_filter import UnitFilter

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
FILES = (
    "0_June_Event_Scenario.aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
)
ROUNDS = 8
BRUSH = 9


def _shared_tile_units(scenario, cache) -> list:
    """Every filter-visible, on-map unit sharing a footprint or own tile with another."""
    own_index = render.unit_own_tile_index(scenario)
    out = []
    for units in scenario.unit_manager.units:
        for u in units:
            own, tiles = _footprint(scenario, u)
            if tiles is None or not any(e[0] is u for e in cache.units_by_tile.get(tiles[0], ())):
                continue
            if len(own_index.get(own, ())) > 1 or any(len(cache.units_by_tile.get(t, ())) > 1 for t in tiles):
                out.append(u)
    return out


def _over_cap(scenario, cache, tiles) -> bool:
    """Whether a refused raise was refused for the cache's component cap alone."""
    out = render_cache._elevation_splices(scenario, cache.units_by_tile, cache.unit_filter, tiles, 10**9)
    return out is not None and len(out) > cache.elevation_splice_cap()


@pytest.mark.corpus
@pytest.mark.parametrize("style", ["stepped", "sloped"])
@pytest.mark.parametrize("name", FILES)
def test_random_shared_tile_raises_splice_exactly_like_a_fresh_cache(name, style, monkeypatch):
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible -- set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    path = EXAMPLES / name
    if not path.exists():
        pytest.skip(f"{name} is not in examples/")
    scenario = load_map_and_units(path)
    mm = scenario.map_manager
    rng = random.Random(f"{name}-{style}")
    unit_filter = UnitFilter()
    cache = _make_cache(style, scenario, unit_filter)
    candidates = _shared_tile_units(scenario, cache)
    assert candidates, f"{name} has no unit on a shared tile -- vacuous"

    calls = []
    real_splices = render_cache._elevation_splices

    def recorded(*args, **kwargs):
        out = real_splices(*args, **kwargs)
        calls.append((args[3], out))
        return out

    monkeypatch.setattr(render_cache, "_elevation_splices", recorded)
    rebuilds = []
    real_rebuild = SlopedChunkCache._rebuild_unit_layers
    monkeypatch.setattr(
        SlopedChunkCache, "_rebuild_unit_layers", lambda self, h: (rebuilds.append(h), real_rebuild(self, h))[1]
    )
    stats = {"spliced": 0, "units": 0, "seeds": 0, "over_cap": 0, "headroom": 0}

    for r in range(ROUNDS):
        seed = rng.choice(candidates)
        cx, cy = int(seed.x), int(seed.y)
        window = [
            (x, y)
            for x in range(cx - BRUSH // 2, cx + BRUSH // 2 + 1)
            for y in range(cy - BRUSH // 2, cy + BRUSH // 2 + 1)
            if 0 <= x < mm.map_width and 0 <= y < mm.map_height
        ]
        before = [t.elevation for t in mm.terrain]
        set_tiles_elevation(mm, [(x, y, mm.get_tile(x, y).elevation + 1) for x, y in window])
        dirty = [i for i, t in enumerate(mm.terrain) if t.elevation != before[i]]
        changed: set = set()
        bbox_fn = dirty_screen_bbox_iso if isinstance(cache, IsoChunkCache) else dirty_screen_bbox_sloped
        bbox = bbox_fn(
            scenario, dirty, cache.elevations, cache.proj, with_units=True, with_sprites=True,
            elevation_changed=changed,
        )
        assert changed, f"round {r}: the raise changed nothing"
        gen = getattr(cache, "_source_gen", None)
        headroom = getattr(cache, "_headroom", None)
        calls.clear()
        rebuilds.clear()
        cache.patch(bbox, elevation_changed=changed)
        cache.invalidate_region((0, 0, *cache.canvas_dims(0)))
        tag = f"round {r}: brush {BRUSH} at ({cx}, {cy})"

        # A Sloped headroom change still splices; only the bbox layer rebuilds whole.
        if style == "sloped" and cache._headroom != headroom:
            stats["headroom"] += 1
        assert len(calls) == 1, f"{tag}: the elevation splice ran {len(calls)} times"
        tiles, out = calls[0]
        if out is None:
            assert _over_cap(scenario, cache, tiles), f"{tag}: fell back below the component cap"
            stats["over_cap"] += 1
            if isinstance(cache, IsoChunkCache):
                for mip in _mips(cache):
                    cache.render_rect(0, 0, VIEW_PX, VIEW_PX, mip=mip)
        else:
            # The window's centre unit shares a tile, so the pre-component guard would have refused this.
            assert id(seed) in {id(s.unit) for s in out}, f"{tag}: the shared-tile seed was not spliced"
            stats["spliced"] += 1
            stats["units"] += len(out)
            stats["seeds"] += sum(1 for s in out if s.new_own_tile in tiles)
            if isinstance(cache, IsoChunkCache):
                assert cache._source_gen == gen, f"{tag}: the source gen bumped"
            else:
                assert not rebuilds, f"{tag}: the unit layers were rebuilt wholesale"
        _assert_matches_fresh(cache, style, scenario, unit_filter, tag)

    print(f"{name} {style}: {stats}")
    assert stats["spliced"], f"no round took the splice path: {stats}"
