"""Property-level coverage for descape/mirror_tools.py -- no Qt, no
AoE2ScenarioParser. Uses the same fake-tile/fake-map-manager protocol as
tests/test_fill_tools.py/test_edit_history.py (plain objects exposing
map_width/map_height/terrain, and terrain_id/elevation/layer per tile).
FakeTile/FakeMapManager are duplicated here rather than imported, matching
those modules' own deliberately self-contained style.
"""

from __future__ import annotations

import random

from descape.edit_history import tile_state
from descape.mirror_tools import MODE_BY_ID, MODES, TRANSFORMS, doubled, plan_mirror


class FakeTile:
    def __init__(self, terrain_id: int = 0, elevation: int = 0, layer: int = -1):
        self.terrain_id = terrain_id
        self.elevation = elevation
        self.layer = layer


class FakeMapManager:
    def __init__(self, terrain: list[FakeTile], map_width: int, map_height: int):
        self.terrain = terrain
        self.map_width = map_width
        self.map_height = map_height


def _square(n: int) -> FakeMapManager:
    return FakeMapManager([FakeTile() for _ in range(n * n)], n, n)


def _apply(mm: FakeMapManager, plan) -> None:
    """Test-local stand-in for the bare assignment loop on_mirror() hands to
    edit_history.apply() -- mutates mm.terrain in place from plan.changes."""
    for idx, (terrain_id, elevation, layer) in plan.changes:
        tile = mm.terrain[idx]
        tile.terrain_id, tile.elevation, tile.layer = terrain_id, elevation, layer


def _is_elevation_legal(mm: FakeMapManager) -> bool:
    """8-connected ±1 sweep, independent of plan_mirror's own -- used to pin
    checker-vs-ground-truth rather than checker-vs-mode-table."""
    w, h = mm.map_width, mm.map_height
    for y in range(h):
        for x in range(w):
            e = mm.terrain[y * w + x].elevation
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if abs(e - mm.terrain[ny * w + nx].elevation) > 1:
                        return False
    return True


def _rep(mode, x: int, y: int, n: int) -> tuple[int, int]:
    """Re-derives plan_mirror's own `rep` computation from the module's
    public TRANSFORMS/doubled/preferred_domain -- the exact formula the
    module's docstring publishes, not a duplicate of a hidden internal."""
    orbit = {TRANSFORMS[name](x, y, n) for name in mode.group}
    return min(
        orbit,
        key=lambda t: (0 if mode.preferred_domain(*doubled(t[0], t[1], n)) else 1, t[1], t[0]),
    )


def _source(mode, slice_index: int, x: int, y: int, n: int) -> tuple[int, int]:
    rep = _rep(mode, x, y, n)
    return TRANSFORMS[mode.group[slice_index]](*rep, n)


# ---------------------------------------------------------------------
# Group closure
# ---------------------------------------------------------------------


def test_group_is_closed_under_composition() -> None:
    for n in (4, 5, 8, 9, 37):
        points = [(x, y) for x in (0, 1, n // 2, n - 2, n - 1) for y in (0, 1, n // 2, n - 2, n - 1)]
        for name_a, fn_a in TRANSFORMS.items():
            for name_b, fn_b in TRANSFORMS.items():
                for x, y in points:
                    composed = fn_a(*fn_b(x, y, n), n)
                    # Must land on some single transform's own image of (x, y) --
                    # i.e. the composition is itself one of the eight, applied once.
                    assert any(fn_c(x, y, n) == composed for fn_c in TRANSFORMS.values()), (
                        name_a,
                        name_b,
                        n,
                        (x, y),
                    )


def test_identity_is_a_no_op() -> None:
    for n in (4, 5, 120):
        for x, y in [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (n // 2, n // 3)]:
            assert TRANSFORMS["id"](x, y, n) == (x, y)


def test_every_transform_has_an_inverse_in_the_set() -> None:
    for n in (4, 5, 37):
        for name, fn in TRANSFORMS.items():
            for x, y in [(0, 0), (n - 1, 0), (1, n - 2), (n // 2, 3)]:
                image = fn(x, y, n)
                assert any(TRANSFORMS[inv](*image, n) == (x, y) for inv in TRANSFORMS), (name, n, (x, y))


# ---------------------------------------------------------------------
# rep is orbit-constant -- the whole symmetry guarantee rests on this.
# ---------------------------------------------------------------------


def test_rep_is_orbit_constant() -> None:
    for mode in MODES:
        for n in (4, 5, 8, 9, 37):
            for x in range(n):
                for y in range(n):
                    base_rep = _rep(mode, x, y, n)
                    for g_name in mode.group:
                        gx, gy = TRANSFORMS[g_name](x, y, n)
                        assert _rep(mode, gx, gy, n) == base_rep, (mode.mode_id, n, (x, y), g_name)


# ---------------------------------------------------------------------
# Screen orientation (Decision 3) -- pins the labelling so it can't drift.
# ---------------------------------------------------------------------


def test_a_is_the_true_west_east_mirror() -> None:
    for n in (4, 5, 120):
        assert TRANSFORMS["a"](0, 0, n) == (n - 1, n - 1)
        assert TRANSFORMS["a"](n - 1, 0, n) == (n - 1, 0)  # fixed


def test_d_is_the_true_north_south_mirror() -> None:
    for n in (4, 5, 120):
        assert TRANSFORMS["d"](0, 0, n) == (0, 0)  # fixed
        assert TRANSFORMS["d"](n - 1, 0, n) == (0, n - 1)
        assert TRANSFORMS["d"](0, n - 1, n) == (n - 1, 0)


# ---------------------------------------------------------------------
# Exact coverage: destination indices in `changes` are never duplicated.
# n=120 is scoped to slice_index=0 and do_elevation=False -- the full
# 32-slice-combo x 5-n matrix from the plan's own Tests section is too slow
# for the default tier at n=120 (per-mode timing measured directly); n in
# {4, 5, 8, 9} still gets every mode x every slice.
# ---------------------------------------------------------------------


def test_no_destination_index_is_written_twice() -> None:
    for n in (4, 5, 8, 9):
        mm = _square(n)
        for mode in MODES:
            for slice_index in range(len(mode.group)):
                plan = plan_mirror(mm, mode.mode_id, slice_index, do_terrain=True, do_elevation=True)
                indices = [idx for idx, _ in plan.changes]
                assert len(indices) == len(set(indices)), (mode.mode_id, slice_index, n)

    n = 120
    mm = _square(n)
    for mode in MODES:
        plan = plan_mirror(mm, mode.mode_id, 0, do_terrain=True, do_elevation=False)
        indices = [idx for idx, _ in plan.changes]
        assert len(indices) == len(set(indices)), mode.mode_id


# ---------------------------------------------------------------------
# Degenerate orbits: source_indices is really {d : source(d) == d}, not a
# naive preferred_domain(d) check -- odd-n C4's centre cross is exactly
# where those two definitions would disagree if source_indices were
# computed the wrong way.
# ---------------------------------------------------------------------


def test_degenerate_orbits_under_odd_n_c4() -> None:
    n = 5
    mm = _square(n)
    mode = MODE_BY_ID[6]  # 4-way rotational (90 degrees)
    for slice_index in range(len(mode.group)):
        plan = plan_mirror(mm, 6, slice_index, do_terrain=False, do_elevation=False)
        expected = frozenset(
            y * n + x for x in range(n) for y in range(n) if _source(mode, slice_index, x, y, n) == (x, y)
        )
        assert plan.source_indices == expected, slice_index

        # The centre tile is fixed by every transform in the group, so it is
        # always its own source regardless of slice choice.
        centre = (n // 2) * n + (n // 2)
        assert centre in plan.source_indices


# ---------------------------------------------------------------------
# Symmetry -- the headline property.
# ---------------------------------------------------------------------


def test_result_is_symmetric_under_the_whole_group() -> None:
    rng = random.Random(0)
    for mode in MODES:
        n = 9  # odd, so on-axis/degenerate tiles are exercised too
        mm = _square(n)
        for tile in mm.terrain:
            tile.terrain_id = rng.randrange(0, 50)
            tile.elevation = rng.randrange(0, 3)
            tile.layer = rng.choice([-1, rng.randrange(0, 50)])

        plan = plan_mirror(mm, mode.mode_id, 0, do_terrain=True, do_elevation=True)
        _apply(mm, plan)

        for x in range(n):
            for y in range(n):
                base = tile_state(mm.terrain[y * n + x])
                for g_name in mode.group:
                    gx, gy = TRANSFORMS[g_name](x, y, n)
                    assert tile_state(mm.terrain[gy * n + gx]) == base, (mode.mode_id, g_name, (x, y))


# ---------------------------------------------------------------------
# Involution / idempotence.
# ---------------------------------------------------------------------


def test_applying_the_same_mode_and_slice_twice_is_a_no_op_the_second_time() -> None:
    rng = random.Random(1)
    n = 9
    for mode in MODES:
        mm = _square(n)
        for tile in mm.terrain:
            tile.terrain_id = rng.randrange(0, 50)
            tile.elevation = rng.randrange(0, 3)

        first = plan_mirror(mm, mode.mode_id, 0, do_terrain=True, do_elevation=True)
        _apply(mm, first)

        second = plan_mirror(mm, mode.mode_id, 0, do_terrain=True, do_elevation=True)
        assert second.changes == [], mode.mode_id


# ---------------------------------------------------------------------
# Source is untouched.
# ---------------------------------------------------------------------


def test_source_indices_never_appear_in_changes() -> None:
    rng = random.Random(2)
    n = 9
    for mode in MODES:
        mm = _square(n)
        for tile in mm.terrain:
            tile.terrain_id = rng.randrange(0, 50)
            tile.elevation = rng.randrange(0, 3)
        for slice_index in range(len(mode.group)):
            plan = plan_mirror(mm, mode.mode_id, slice_index, do_terrain=True, do_elevation=True)
            changed = {idx for idx, _ in plan.changes}
            assert not (plan.source_indices & changed), (mode.mode_id, slice_index)


# ---------------------------------------------------------------------
# Elevation seam check. See the module docstring / final report for the
# measured deviation from the plan's "modes 1-4/7-9 always safe" claim:
# mode 7's two diagonal boundaries contain on-axis lattice tiles that
# self-source under the specified tie-break, so it CAN violate from
# otherwise-legal content -- these tests pin the checker against ground
# truth (an independent legality sweep), not against a fixed mode list.
# ---------------------------------------------------------------------


def _ramp_grid(n: int, seam_x: int) -> FakeMapManager:
    """Flat 0 for x < seam_x, ramping 0->3 over x in [seam_x, seam_x+3) and
    flat 3 beyond -- legal everywhere (max delta 1 per step)."""
    mm = _square(n)
    for y in range(n):
        for x in range(n):
            mm.terrain[y * n + x].elevation = 0 if x < seam_x else min(x - seam_x, 3)
    return mm


def test_elevation_seam_check_detects_a_real_rotation_seam() -> None:
    n = 8
    mm = _ramp_grid(n, seam_x=n // 2)
    assert _is_elevation_legal(mm)  # fixture precondition: legal before mirroring

    plan5 = plan_mirror(mm, 5, 0, do_terrain=False, do_elevation=True)
    assert plan5.elevation_violations != []

    plan6 = plan_mirror(mm, 6, 0, do_terrain=False, do_elevation=True)
    assert plan6.elevation_violations != []


def test_elevation_seam_check_matches_ground_truth_when_legal() -> None:
    """A flat, uniform grid mirrors to another flat, uniform grid under
    every mode -- checker and an independent legality sweep must agree
    it's clean."""
    n = 8
    for mode in MODES:
        mm = _square(n)  # elevation 0 everywhere -- trivially legal and already symmetric
        for slice_index in range(len(mode.group)):
            plan = plan_mirror(mm, mode.mode_id, slice_index, do_terrain=False, do_elevation=True)
            assert plan.elevation_violations == [], (mode.mode_id, slice_index)
            _apply(mm, plan)
            assert _is_elevation_legal(mm)


def test_elevation_violations_empty_when_do_elevation_false() -> None:
    n = 8
    mm = _ramp_grid(n, seam_x=n // 2)
    for mode in MODES:
        plan = plan_mirror(mm, mode.mode_id, 0, do_terrain=True, do_elevation=False)
        assert plan.elevation_violations == [], mode.mode_id


# ---------------------------------------------------------------------
# Odd-n degeneracies.
# ---------------------------------------------------------------------


def test_odd_n_centre_tile_is_fixed_under_every_mode() -> None:
    n = 7
    centre = n // 2
    for mode in MODES:
        for g_name in mode.group:
            assert TRANSFORMS[g_name](centre, centre, n) == (centre, centre), (mode.mode_id, g_name)


def test_odd_n_centre_row_and_column_mirror_without_exception() -> None:
    n = 7
    for mode in MODES:
        mm = _square(n)
        for i, tile in enumerate(mm.terrain):
            tile.terrain_id = i % 17
        for slice_index in range(len(mode.group)):
            plan = plan_mirror(mm, mode.mode_id, slice_index, do_terrain=True, do_elevation=False)
            assert isinstance(plan.changes, list)  # must not raise


# ---------------------------------------------------------------------
# Contents flags.
# ---------------------------------------------------------------------


def test_do_terrain_false_leaves_terrain_and_layer_untouched() -> None:
    n = 6
    mm = _square(n)
    for i, tile in enumerate(mm.terrain):
        tile.terrain_id = i
        tile.layer = i % 5
        tile.elevation = 0
    before = [(t.terrain_id, t.layer) for t in mm.terrain]

    plan = plan_mirror(mm, 9, 0, do_terrain=False, do_elevation=True)
    _apply(mm, plan)

    assert [(t.terrain_id, t.layer) for t in mm.terrain] == before


def test_do_elevation_false_leaves_elevation_untouched() -> None:
    n = 6
    mm = _square(n)
    for i, tile in enumerate(mm.terrain):
        tile.terrain_id = i % 3
        tile.elevation = i % 4
    before = [t.elevation for t in mm.terrain]

    plan = plan_mirror(mm, 9, 0, do_terrain=True, do_elevation=False)
    _apply(mm, plan)

    assert [t.elevation for t in mm.terrain] == before
    assert plan.elevation_violations == []


# ---------------------------------------------------------------------
# Slice labels -- cosmetic, but must be well-formed: distinct dropdown text
# per mode, and no mode crashes at module load (already proven by the
# import itself, since MODES is built eagerly).
# ---------------------------------------------------------------------


def test_slice_labels_are_distinct_dropdown_text_per_mode() -> None:
    for mode in MODES:
        assert len(mode.slice_labels) == len(mode.group)
        assert len(set(mode.slice_labels)) == len(mode.group)


def test_only_mode_9_needs_disambiguation_suffixes() -> None:
    # Modes 1-8 should derive genuinely distinct compass labels without a
    # " (N)" suffix; mode 9 (8-way) cannot, per D4's group structure -- see
    # mirror_tools._derive_slice_labels' docstring. Pinned so a change to
    # the sector-classification scheme that silently starts suffixing every
    # mode gets caught here.
    for mode in MODES:
        needs_suffix = any(label.endswith(")") and " (" in label for label in mode.slice_labels)
        if mode.mode_id == 9:
            assert needs_suffix
        else:
            assert not needs_suffix, mode.mode_id
