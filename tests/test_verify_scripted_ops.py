"""Coverage for tools/verify_iso_incremental.py's and
tools/verify_iso_chunks.py's duplicated _scripted_ops() helper: its
"elevation raise" op must produce a real change even when the target tile
is already at MAX_ELEVATION.

Before this fix, op_raise did min(MAX_ELEVATION, tile.elevation + 1), a
silent no-op at the ceiling -- on old-allies-final-v2.aoe2scenario (the
corpus file whose center tile sits at exactly 7, the old MAX_ELEVATION)
this made the op produce zero dirty tiles and get skipped entirely, one of
the two independent defects behind the "old-allies-final-v2" corpus
failures (see descape-plan-fix-for-parsed-pebble.md, Defect A). Widening
MAX_ELEVATION to 15 incidentally hides the bug for every real corpus file
today, so this test exercises the fallback branch directly against a
synthetic tile forced to the ceiling, rather than relying on a corpus file
happening to have one.
"""

from __future__ import annotations

import pytest

import conftest
from descape import iso_geometry
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units


@pytest.mark.parametrize("module_name", ["verify_iso_incremental", "verify_iso_chunks"])
def test_op_raise_changes_a_tile_already_at_the_ceiling(module_name: str) -> None:
    module = conftest.load_verify_module(module_name)
    scenario = load_map_and_units(BLANK_TEMPLATE_PATH)
    mm = scenario.map_manager
    cx, cy = mm.map_width // 2, mm.map_height // 2
    mm.get_tile(cx, cy).elevation = iso_geometry.MAX_ELEVATION

    ops = dict(module._scripted_ops(mm))
    ops["elevation raise"]()

    assert mm.get_tile(cx, cy).elevation != iso_geometry.MAX_ELEVATION
