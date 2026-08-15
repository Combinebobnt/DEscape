"""Verifies MapView._iso_elevations stays in sync with tile.elevation across
a brush stroke whose elevation propagation re-changes tiles mid-drag.

Root cause being guarded: ViewerWindow.on_edit_stroke_tile used to compute
its incremental repaint set as `stroke_dirty_indices(...) - seen_indices`.
stroke_dirty_indices is CUMULATIVE (everything differing from the
stroke-start snapshot), so once a tile appeared it stayed for the rest of
the drag -- while set_tiles_elevation's propagation routinely changes one
tile several times as the brush moves over it. Subtracting a set of
INDICES therefore handed each tile to _apply_dirty exactly once, at its
first value. dirty_screen_bbox_iso is the only thing that writes the
_iso_elevations snapshot, so that snapshot froze at the first value and
drifted permanently.

Symptoms that made it visible, all from the one cause: top faces looked
correct (they render from tile.elevation), but the hover highlight and
screen_to_tile hit-testing read the snapshot array, and _render_tile_iso
computes skirt/contact-shadow deltas as
`tile.elevation - elevations[neighbour]` -- mixing the fresh value with
the stale one, so shadows got wrong deltas too.

Deliberately does NOT import viewer.py: that needs Qt, and the defect
lives in the dirty-set bookkeeping, not in any widget. These tests
replicate that bookkeeping directly against the real EditHistory,
set_tiles_elevation and dirty_screen_bbox_iso, so they pin the contract
those three have with each other rather than a GUI detail.
"""

from __future__ import annotations

import pytest

from descape import brush, render
from descape.edit_history import EditHistory, tile_state
from descape.elevation_tools import set_tiles_elevation
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


def _drag(scenario, elevations, proj, level, steps, *, state_keyed):
    """Replays ViewerWindow.on_edit_stroke_tile's dirty-set bookkeeping.

    state_keyed=True is the fixed behaviour (compare last-applied STATE);
    False is the old index-membership version, kept so the tests can show
    the guard actually discriminates rather than passing vacuously.
    """
    mm = scenario.map_manager
    history = EditHistory()
    history.begin_stroke(mm.terrain)
    seen_state: dict[int, tuple[int, int, int]] = {}
    seen_indices: set[int] = set()
    for i in range(steps):
        footprint = brush.brush_tiles(40 + i, 40, 5, "circle", mm.map_width, mm.map_height)
        set_tiles_elevation(mm, [(tx, ty, level) for tx, ty in footprint])
        all_dirty = history.stroke_dirty_indices(mm.terrain)
        if state_keyed:
            new_dirty = {i2 for i2 in all_dirty if tile_state(mm.terrain[i2]) != seen_state.get(i2)}
            for i2 in all_dirty:
                seen_state[i2] = tile_state(mm.terrain[i2])
        else:
            new_dirty = set(all_dirty) - seen_indices
            seen_indices = set(all_dirty)
        render.dirty_screen_bbox_iso(scenario, new_dirty, elevations, proj, with_units=True)
    return history


def _out_of_sync(scenario, elevations):
    return [
        (t.x, t.y, int(t.elevation), int(elevations[t.y, t.x]))
        for t in scenario.map_manager.terrain
        if int(elevations[t.y, t.x]) != int(t.elevation)
    ]


def _fresh():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    elevations, proj = render.elevations_and_proj(scenario)
    return scenario, elevations, proj


def test_snapshot_tracks_every_tile_after_a_propagating_drag():
    """THE regression assertion: after a drag that re-changes tiles via
    propagation, the snapshot must equal tile.elevation everywhere.
    """
    scenario, elevations, proj = _fresh()
    _drag(scenario, elevations, proj, level=6, steps=8, state_keyed=True)
    bad = _out_of_sync(scenario, elevations)
    assert not bad, f"{len(bad)} tiles out of sync, e.g. {bad[:5]}"


def test_old_index_keyed_bookkeeping_would_have_drifted():
    """Pins that the fix is load-bearing. If this ever stops finding
    drift, the fixture no longer reproduces the propagation-re-change the
    real defect needed, and the test above has gone vacuous.
    """
    scenario, elevations, proj = _fresh()
    _drag(scenario, elevations, proj, level=6, steps=8, state_keyed=False)
    bad = _out_of_sync(scenario, elevations)
    assert bad, "fixture no longer reproduces the drift -- the guard above is now vacuous"
    at_level = [b for b in bad if b[2] == 6]
    assert at_level, "expected tiles sitting at the target level whose snapshot reads lower"


def test_shadow_deltas_use_a_consistent_elevation_source():
    """The rendering consequence, stated directly: _render_tile_iso computes
    its skirt/contact-shadow deltas as tile.elevation - elevations[neighbour].
    With a drifted snapshot those two disagree, so a tile can be handed a
    delta it never actually has against that neighbour.
    """
    scenario, elevations, proj = _fresh()
    _drag(scenario, elevations, proj, level=6, steps=8, state_keyed=True)
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    for tile in mm.terrain:
        for nx, ny in ((tile.x, tile.y - 1), (tile.x + 1, tile.y)):
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            from_array = int(tile.elevation) - int(elevations[ny, nx])
            from_tiles = int(tile.elevation) - int(mm.get_tile(nx, ny).elevation)
            assert from_array == from_tiles, (
                f"tile ({tile.x},{tile.y}) vs neighbour ({nx},{ny}): "
                f"delta {from_array} from the snapshot, {from_tiles} from the real tiles"
            )


@pytest.mark.parametrize("level", [1, 6])
def test_snapshot_in_sync_for_a_single_step_stroke_too(level):
    """A one-step stroke still propagates (the brush is wider than one
    tile), so this is not just the trivial case.
    """
    scenario, elevations, proj = _fresh()
    _drag(scenario, elevations, proj, level=level, steps=1, state_keyed=True)
    assert not _out_of_sync(scenario, elevations)
