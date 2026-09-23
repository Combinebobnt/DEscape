"""Property-level coverage for descape/mirror_tools.py -- no Qt, no
AoE2ScenarioParser. Uses the same fake-tile/fake-map-manager protocol as
tests/test_fill_tools.py/test_edit_history.py (plain objects exposing
map_width/map_height/terrain, and terrain_id/elevation/layer per tile).
FakeTile/FakeMapManager are duplicated here rather than imported, matching
those modules' own deliberately self-contained style.
"""

from __future__ import annotations

import math
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


# --- Stage 2b: gate reorientation ---------------------------------------------
#
# The whole design rests on one measured claim: a gate's occupied tiles,
# reflected by a D4 element, are exactly its target sibling's tiles modulo
# translation -- including the two DIAGONAL orientations, whose footprints are
# sparse 6-tile sets inside a 4x4 box rather than solid runs. This runs it as
# a default-tier test rather than as a throwaway probe, and it applies the real
# TRANSFORMS functions rather than a table re-typed here, so it pins the
# transforms against drift too.


def _gate_shape(unit_const: int, n: int = 40):
    from descape import render

    tiles = render.occupied_tiles_for(unit_const, *render.span_anchor(10, 10, *_gate_span(unit_const)), n, n)
    return _normalized(tiles)


def _gate_span(unit_const: int):
    from descape import render
    from descape.terrain_palette import tile_span

    return tile_span(unit_const, render.NON_BUILDING_SPAN)


def _normalized(tiles):
    min_x = min(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)
    return sorted((x - min_x, y - min_y) for x, y in tiles)


def test_every_gate_orientation_reflects_onto_the_sibling_reorient_picks():
    from descape import gate_orientation
    from descape.mirror_tools import TRANSFORMS, reorient_gate_const

    n = 40
    checked = 0
    for siblings in gate_orientation.groups().values():
        if all(_gate_span(const) == (1, 1) for const in siblings):
            continue  # the 1x1 corner groups: nothing to reflect, see below
        for const in siblings:
            tiles = _gate_shape(const, n)
            for name, fn in TRANSFORMS.items():
                reflected = _normalized([fn(x, y, n) for x, y in tiles])
                target = reorient_gate_const(const, name)
                assert reflected == _gate_shape(target, n), (
                    f"const {const} under {name}: reflected tiles are not {target}'s shape"
                )
                checked += 1
    assert checked == 576, f"expected 24 groups x 4 orientations x 8 elements, checked {checked}"


def test_the_1x1_corner_gates_pass_through_verbatim():
    """Measured decision, not a gap: all four siblings of a corner group share
    one graphic, and a 1x1 footprint is invariant under every element, so a
    remap would churn the stored const for no visual effect."""
    from descape import gate_orientation
    from descape.mirror_tools import TRANSFORMS, reorient_gate_const

    corners = [
        const
        for siblings in gate_orientation.groups().values()
        if all(_gate_span(c) == (1, 1) for c in siblings)
        for const in siblings
    ]
    assert len(corners) == 24, f"expected six 1x1 groups, found {len(corners) // 4}"
    for const in corners:
        for name in TRANSFORMS:
            assert reorient_gate_const(const, name) == const


def test_reorient_refuses_a_non_gate():
    import pytest

    from descape.mirror_tools import reorient_gate_const

    with pytest.raises(ValueError):
        reorient_gate_const(117, "d")  # a wall


# --- Angular modes (6-way / 3-fold) -------------------------------------------
#
# Not exact, so these assert exactness only where it is real (the sub-elements
# that ARE D4 elements, the source wedge, idempotence) and bound it elsewhere.
# The D4 block above stays untouched and never iterates ANGULAR_MODES.


def _angular():
    from descape.mirror_tools import ANGULAR_MODES

    return ANGULAR_MODES


def _random_map(n: int, seed: int) -> FakeMapManager:
    rng = random.Random(seed)
    mm = _square(n)
    for tile in mm.terrain:
        tile.terrain_id = rng.randrange(0, 50)
        tile.elevation = rng.randrange(0, 3)
        tile.layer = rng.choice([-1, rng.randrange(0, 50)])
    return mm


def test_the_lattice_modes_are_unchanged_and_the_angular_ones_live_apart() -> None:
    assert [mode.mode_id for mode in MODES] == list(range(1, 10))
    assert all(mode.kind == "lattice" and mode.angular is None for mode in MODES)
    assert [mode.mode_id for mode in _angular()] == [10, 11, 12]
    for mode in _angular():
        assert mode.kind == "angular" and mode.group == ()
        assert MODE_BY_ID[mode.mode_id] is mode


def test_the_angular_element_algebra_is_closed_and_matches_its_own_action() -> None:
    from descape.mirror_tools import AngularElement, angular_compose, angular_invert

    rng = random.Random(3)
    points = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(20)]
    everything = [AngularElement(s, f) for s in range(6) for f in (False, True)]
    for g in everything:
        for h in everything:
            gh = angular_compose(g, h)
            for u, v in points:
                expected = g.apply(*h.apply(u, v))
                got = gh.apply(u, v)
                assert abs(got[0] - expected[0]) < 1e-9 and abs(got[1] - expected[1]) < 1e-9
        assert angular_compose(g, angular_invert(g)) == AngularElement(0, False)
    sizes = {10: 3, 11: 6, 12: 6}
    for mode in _angular():
        elements = set(mode.angular.elements)
        assert len(elements) == sizes[mode.mode_id] == mode.angular.wedges
        assert AngularElement(0, False) in elements
        for g in elements:
            assert angular_invert(g) in elements
            for h in elements:
                assert angular_compose(g, h) in elements, (mode.mode_id, g, h)
    d3 = set(MODE_BY_ID[12].angular.elements)
    assert sum(element.reflect for element in d3) == 3
    assert AngularElement(0, True) in d3  # `a` itself: the exact mirror axis


def test_the_angular_elements_that_are_d4_elements_act_exactly_like_them() -> None:
    """Cheapest possible centre-convention bug detector: exact equality, no
    tolerance, on tiles and in doubled space."""
    from descape.mirror_tools import _UV_TRANSFORMS, AngularElement

    pairs = {AngularElement(0, False): "id", AngularElement(3, False): "r2", AngularElement(0, True): "a"}
    for n in (8, 9):
        for x in range(n):
            for y in range(n):
                u, v = doubled(x, y, n)
                for element, name in pairs.items():
                    got = element.apply(u, v)
                    assert got == _UV_TRANSFORMS[name](u, v), (element, name, u, v)
                    gx, gy = TRANSFORMS[name](x, y, n)
                    assert got == doubled(gx, gy, n)


def test_the_continuous_reduction_is_exactly_equivariant() -> None:
    """reduce(g(p)) == reduce(p) for every group element, BEFORE rounding --
    the analytic counterpart of the D4 path's `rep` orbit-constancy."""
    from descape.mirror_tools import angular_reduce

    rng = random.Random(4)
    for mode in _angular():
        spec = mode.angular
        width = 360.0 / spec.wedges
        for j in range(spec.wedges):
            checked = 0
            while checked < 200:
                u, v = rng.uniform(-100, 100), rng.uniform(-100, 100)
                angle = (math.degrees(math.atan2(v, u)) - spec.start_degrees) % width
                if min(angle, width - angle) < 1e-6:
                    continue  # on a wedge boundary: sector() picks a side, fine either way
                base = angular_reduce(spec, j, u, v)
                for g in spec.elements:
                    image = angular_reduce(spec, j, *g.apply(u, v))
                    assert abs(image[0] - base[0]) < 1e-9 and abs(image[1] - base[1]) < 1e-9
                checked += 1


def test_the_source_wedge_is_preserved_byte_for_byte() -> None:
    """No source index is ever changed, every tile of wedge j is a source, and
    the surplus is exactly the tiles on `a`'s axis that `a` fixes -- empty for
    the pure rotations, and non-empty in mode 12 only beside that axis."""
    for n in (40, 41):
        mm = _random_map(n, 5)
        for mode in _angular():
            spec = mode.angular
            with_surplus = set()
            for j in range(spec.wedges):
                plan = plan_mirror(mm, mode.mode_id, j, True, True)
                changed = {idx for idx, _ in plan.changes}
                assert not (changed & plan.source_indices)
                wedge = {
                    y * n + x for y in range(n) for x in range(n) if spec.sector(*doubled(x, y, n)) == j
                }
                assert plan.source_indices >= wedge
                surplus = plan.source_indices - wedge
                if mode.mode_id != 12:
                    # The odd-n centre tile is fixed by every rotation, so it is
                    # a legitimate surplus when wedge j does not own it.
                    assert surplus <= {(n // 2) * n + n // 2} if n % 2 else not surplus
                    continue
                for idx in surplus:
                    u, v = doubled(idx % n, idx // n, n)
                    assert u + v == 0, (n, j, idx)
                if surplus and n % 2 == 0:
                    with_surplus.add(j)
            if mode.mode_id == 12 and n % 2 == 0:
                # `a`'s axis is the 135/315 degree boundary pair: wedges 1|2 and 4|5.
                # The half-open sector() gives its tiles to one side of each.
                assert with_surplus <= {1, 2, 4, 5}
                assert len(with_surplus & {1, 2}) == 1 and len(with_surplus & {4, 5}) == 1


def test_an_angular_mirror_applied_twice_is_a_no_op_the_second_time() -> None:
    """What the wedge-constrained snap buys: fails the moment the snap may
    read a tile this same operation rewrites."""
    for n in (40, 41):
        for mode in _angular():
            for j in range(mode.angular.wedges):
                mm = _random_map(n, 6 + j)
                _apply(mm, plan_mirror(mm, mode.mode_id, j, True, True))
                assert plan_mirror(mm, mode.mode_id, j, True, True).changes == [], (n, mode.mode_id, j)


def test_six_way_rotation_reproduces_the_exact_180_degree_mirror() -> None:
    """Mode 11 from wedge j must agree with the exact r2 on wedge j+3, and
    with mode 5 itself where mode 5's source half holds wedge j."""
    n = 40
    for j in range(6):
        mm = _random_map(n, 7)
        before = [tile_state(tile) for tile in mm.terrain]
        _apply(mm, plan_mirror(mm, 11, j, True, True))
        spec = MODE_BY_ID[11].angular
        opposite = [
            (x, y) for y in range(n) for x in range(n) if spec.sector(*doubled(x, y, n)) == (j + 3) % 6
        ]
        assert opposite
        for x, y in opposite:
            sx, sy = TRANSFORMS["r2"](x, y, n)
            assert tile_state(mm.terrain[y * n + x]) == before[sy * n + sx]
        if j in (3, 4):  # wedges wholly inside mode 5's slice-0 half u + v < 0
            exact = _random_map(n, 7)
            _apply(exact, plan_mirror(exact, 5, 0, True, True))
            for x, y in opposite:
                assert tile_state(mm.terrain[y * n + x]) == tile_state(exact.terrain[y * n + x])


def test_six_way_reflective_reproduces_the_exact_screen_mirror_on_as_image() -> None:
    from descape.mirror_tools import AngularElement, angular_compose

    n = 40
    spec = MODE_BY_ID[12].angular
    for j in range(6):
        mm = _random_map(n, 8)
        before = [tile_state(tile) for tile in mm.terrain]
        _apply(mm, plan_mirror(mm, 12, j, True, True))
        mirrored_wedge = spec.elements.index(angular_compose(AngularElement(0, True), spec.elements[j]))
        tiles = [(x, y) for y in range(n) for x in range(n) if spec.sector(*doubled(x, y, n)) == mirrored_wedge]
        assert tiles
        for x, y in tiles:
            sx, sy = TRANSFORMS["a"](x, y, n)
            assert tile_state(mm.terrain[y * n + x]) == before[sy * n + sx], (j, x, y)


def test_every_snapped_source_is_within_one_tile_of_its_preimage() -> None:
    """The 2x2-block path is bounded by 1 tile (Chebyshev). The widened path,
    measured at zero occurrences on these sizes, gets its own looser bound so
    this assertion keeps covering the 2x2 path on its own."""
    from descape.mirror_tools import GATHER_SNAP, GATHER_WIDEN, angular_gather, angular_reduce

    for n in (40, 41, 120):
        for mode in _angular():
            spec = mode.angular
            for j in range(spec.wedges):
                gather = angular_gather(mode.mode_id, j, n)
                snapped = 0
                for d_idx, (src, kind) in enumerate(zip(gather.sources, gather.kinds, strict=True)):
                    if kind not in (GATHER_SNAP, GATHER_WIDEN):
                        continue
                    pu, pv = angular_reduce(spec, j, *doubled(d_idx % n, d_idx // n, n))
                    su, sv = doubled(src % n, src // n, n)
                    error = max(abs(su - pu), abs(sv - pv)) / 2  # doubled -> tiles
                    assert error <= (1.0 if kind == GATHER_SNAP else 3.0), (n, mode.mode_id, j, d_idx)
                    snapped += kind == GATHER_SNAP
                assert snapped > 0


def test_unreachable_is_exactly_the_off_map_preimage_set() -> None:
    from descape.mirror_tools import angular_reduce

    n = 40
    mm = _random_map(n, 9)
    for mode in _angular():
        spec = mode.angular
        for j in range(spec.wedges):
            plan = plan_mirror(mm, mode.mode_id, j, True, True)
            expected = set()
            for y in range(n):
                for x in range(n):
                    pu, pv = angular_reduce(spec, j, *doubled(x, y, n))
                    if not (-n <= pu < n and -n <= pv < n):
                        expected.add(y * n + x)
            assert plan.unreachable == expected
            assert not plan.unreachable & {idx for idx, _ in plan.changes}
            assert not plan.unreachable & plan.source_indices


def test_the_unreachable_fraction_is_a_corner_sized_slice_of_the_map() -> None:
    """Measured at 120 and 480: 5.6-15.5% depending on mode and wedge, from
    the 21.5% of the square outside its inscribed circle. A loose band, so a
    geometry regression shows without the test becoming a magic number."""
    for n in (120,):
        for mode in _angular():
            for j in range(mode.angular.wedges):
                plan = plan_mirror(_square(n), mode.mode_id, j, False, False)
                fraction = len(plan.unreachable) / (n * n)
                assert 0.04 < fraction < 0.18, (mode.mode_id, j, fraction)


def test_the_elevation_sweep_catches_a_staircase_row_the_snap_drops() -> None:
    """Decision 5's failure mode, pinned directly: a legal one-level-per-tile
    ramp comes out with a +-2 step because nearest-neighbour skipped a row."""
    n = 40
    mm = _square(n)
    for y in range(n):
        for x in range(n):
            mm.terrain[y * n + x].elevation = x  # legal: exactly +-1 between x-neighbours
    assert _is_elevation_legal(mm)
    from descape.mirror_tools import angular_gather

    for mode_id in (10, 11, 12):
        plan = plan_mirror(mm, mode_id, 0, False, True)
        gather = angular_gather(mode_id, 0, n)
        dropped = []
        for a, b in plan.elevation_violations:
            sa, sb = gather.sources[a], gather.sources[b]
            if sa < 0 or sb < 0:
                continue  # an untouched corner tile, not a snap
            # Adjacent destinations whose sources are two columns apart: a skipped row.
            if abs(sa % n - sb % n) == 2 and abs(sa // n - sb // n) <= 2:
                dropped.append((a, b))
        assert dropped, mode_id


def test_angular_slice_labels_are_distinct_per_wedge() -> None:
    for mode in _angular():
        assert len(mode.slice_labels) == mode.angular.wedges
        assert len(set(mode.slice_labels)) == mode.angular.wedges


def test_the_angular_gather_is_fast_enough_for_a_live_dialog() -> None:
    """Decision 3's cost check at the largest standard size, uncached."""
    import time

    from descape.mirror_tools import angular_gather

    angular_gather.cache_clear()
    start = time.perf_counter()
    angular_gather(11, 1, 480)
    elapsed = time.perf_counter() - start
    angular_gather.cache_clear()
    assert elapsed < 10.0, elapsed
