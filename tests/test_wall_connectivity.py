"""render.wall_variant_rotation_overrides() -- a wall/gate's shape derived
from its neighbours when the stored `rotation` carries no shape signal at all
(some scenario files encode every wall rotation in radians, which turns out
to carry no shape information at all; see that function's own docstring).

**Must be synthetic, not corpus-backed**: examples/ is gitignored
(.gitignore:6), so a default-tier test cannot depend on it. Copies
tests/test_sprite_chunks.py's module-local harness (Unit/_MapManager/
_UnitManager/_Scenario/_scenario) rather than importing testkit/fakes.py --
its SyntheticUnit has no `rotation` field at all, and that module scopes
itself to four named modules on purpose.

**Cross-path consistency** (sprite_draws_by_anchor_sliced,
_flat_icon_layer_sliced, overlay_units -> _flat_icon_for_unit) is checked by
spying on the ROTATION VALUE each site passes into
unit_sprites.sprite_pieces_for()/icon_for() -- the exact quantity frame
dispatch is a deterministic function of -- rather than by diffing rendered
pixels. This project has already shipped a shared bug that passed a shared
pixel assertion for the wrong reason (P3-g3's sprite-anchor half-tile float),
so asserting on the argument the production code is caught reading is the
more direct check, not a weaker one.

**Mutation arm for the real-gate case, run by hand and confirmed 2026-09-08,
not asserted**: widening `unit_sprites.WALL_CONNECTOR_CONSTS` to also include
789 (i.e. "fixing" the documented asymmetry) turns the third stage of
`test_a_real_gate_orientation_swap_changes_a_neighbouring_walls_connector_membership`
red, since the probe wall would then keep its derived override instead of
falling back to its own stored rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from descape import render, unit_sprites
from descape.terrain_palette import PLAYER_COLORS

MAP_W = MAP_H = 12
WALL_CONST = 9001


@dataclass
class Tile:
    x: int
    y: int
    elevation: int
    terrain_id: int = 0
    layer: int = -1


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0


class _MapManager:
    def __init__(self, tiles):
        self.map_width, self.map_height = MAP_W, MAP_H
        self.terrain = tiles
        self._by_xy = {(t.x, t.y): t for t in tiles}

    def get_tile(self, x, y):
        return self._by_xy[(x, y)]


class _UnitManager:
    def __init__(self, units_by_player):
        self.units = units_by_player


class _Scenario:
    def __init__(self, tiles, units_by_player):
        self.map_manager = _MapManager(tiles)
        self.unit_manager = _UnitManager(units_by_player)
        self.player_colors = tuple(PLAYER_COLORS)
        self.team_indices = tuple(range(len(PLAYER_COLORS)))


def _scenario(units_by_player):
    tiles = [Tile(x, y, 0) for y in range(MAP_H) for x in range(MAP_W)]
    return _Scenario(tiles, units_by_player)


# Radian encodings of variant indices 1..4 at angle_count 5 -- see
# unit_sprites.variant_index()'s own docstring for the formula. Used to plant
# a stored rotation that is deliberately the WRONG shape, so a test that
# passes because the override never ran is distinguishable from one where it
# ran and picked the right answer.
_RADIAN = {i: i * 2 * 3.141592653589793 / 5 for i in range(5)}


@pytest.fixture
def wall_install(monkeypatch):
    """Registers WALL_CONST as a variant-index, angle_count=5 wall/gate
    connector -- no real .sld install needed, since every test here checks
    the rotation VALUE handed to the frame-dispatch functions, not decoded
    pixels."""
    monkeypatch.setattr(unit_sprites, "_ROTATION_VARIANT_CONSTS", frozenset({WALL_CONST}))
    monkeypatch.setattr(unit_sprites, "WALL_CONNECTOR_CONSTS", frozenset({WALL_CONST}))
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {WALL_CONST: {"graphic_id": 1, "file_name": "wall_synth_x1",
                              "angle_count": 5, "mirroring_mode": 6, "frame_count": 1}},
    )
    unit_sprites.clear_caches()
    yield
    unit_sprites.clear_caches()


# --- the pure table, exercised through a real scenario -----------------


def test_a_run_along_x_resolves_to_index_0(wall_install):
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, _RADIAN[3]),
        Unit(6.0, 5.0, WALL_CONST, _RADIAN[3]),  # deliberately the WRONG shape
        Unit(7.0, 5.0, WALL_CONST, _RADIAN[3]),
    ]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert overrides[(1, 1)] == 0.0  # the middle piece has neighbours on +-x only


def test_a_run_along_y_resolves_to_index_1(wall_install):
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, _RADIAN[3]),
        Unit(5.0, 6.0, WALL_CONST, _RADIAN[3]),
        Unit(5.0, 7.0, WALL_CONST, _RADIAN[3]),
    ]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert overrides[(1, 1)] == 1.0


def test_an_l_junction_resolves_to_index_2(wall_install):
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, _RADIAN[3]),
        Unit(6.0, 5.0, WALL_CONST, _RADIAN[3]),  # east neighbour of the corner
        Unit(5.0, 4.0, WALL_CONST, _RADIAN[3]),  # north neighbour of the corner
    ]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert overrides[(1, 0)] == 2.0


def test_an_isolated_piece_falls_through_to_its_stored_value(wall_install):
    """No neighbours -> no entry in the table at all, so callers keep
    resolving through today's variant_index() on the stored rotation
    unchanged -- an isolated piece's shape is author-chosen, not derivable."""
    scn = _scenario([[], [Unit(2.0, 2.0, WALL_CONST, _RADIAN[3])]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert (1, 0) not in overrides


def test_a_literal_integer_rotation_is_untouched_when_the_file_has_no_radian_evidence(wall_install):
    """The whole-corpus regression guard, at unit-test scale: a file where
    every wall/gate rotation is already a literal index (today's confirmed-
    in-game path) must see ZERO overrides, even when the connectivity mask
    would derive a different shape -- precondition 3 only relaxes inside a
    file that also carries a real radian-encoded wall
    (unit_sprites.rotation_variant_eligible()'s docstring)."""
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, 2.0),  # stored as a literal index, not radian
        Unit(6.0, 5.0, WALL_CONST, 2.0),
        Unit(7.0, 5.0, WALL_CONST, 2.0),
    ]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert overrides == {}


def test_a_literal_zero_rotation_is_overridden_inside_a_radian_file(wall_install):
    """The counterexample-plan finding this whole mechanism exists for: inside
    a file that DOES carry a real radian-encoded wall, a literal-looking 0.0
    on another wall is not trustworthy either (measured 45.6% disagreement on
    examples/) and must go through the override too."""
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, 0.0),  # literal-looking, but this file is radian
        Unit(6.0, 5.0, WALL_CONST, 0.0),
        Unit(7.0, 5.0, WALL_CONST, _RADIAN[3]),  # real radian evidence elsewhere in the file
    ]])
    overrides = render.wall_variant_rotation_overrides(scn)
    assert overrides[(1, 1)] == 0.0


def test_gaia_never_overrides_a_variant_rotation(wall_install):
    """AGENTS.md's hard rule: GAIA's rotation is forced to 0.0 upstream of
    this table for every caller, and this table must not fight that -- a GAIA
    wall run still resolves through the (already-forced) value like any other
    literal 0.0 would inside a radian file, never through its raw stored
    field. The radian evidence has to come from a NON-GAIA unit here: GAIA's
    rotation is forced to 0.0 before the literal-vs-radian check ever runs,
    so a GAIA-only file can never look "radian" no matter what junk its
    units' raw rotation fields carry."""
    scn = _scenario([
        [
            Unit(5.0, 5.0, WALL_CONST, 999.0),  # GAIA: a real doodad-variant value, never an angle
            Unit(6.0, 5.0, WALL_CONST, 999.0),  # its one connector neighbour
        ],
        [Unit(0.0, 0.0, WALL_CONST, _RADIAN[3])],  # player 1: establishes file_is_radian
    ])
    overrides = render.wall_variant_rotation_overrides(scn)
    # A single neighbour is a non-{WEST,EAST}/{NORTH,SOUTH} mask, so the
    # derived shape is 2 -- computed off the FORCED 0.0, never off 999.0.
    assert overrides[(0, 0)] == 2.0


def test_the_table_ignores_unit_filter(wall_install):
    """A wall's real shape does not depend on which players are shown --
    hiding player 2 must not reshape player 1's wall. Built by construction
    over ALL units; there is no unit_filter parameter to even pass."""
    import inspect
    assert "unit_filter" not in inspect.signature(render.wall_variant_rotation_overrides).parameters


# --- cross-path consistency, and the row-vs-position key hazard ---------


@pytest.fixture
def rotation_spy(monkeypatch):
    """Records the rotation value each of the three frame-dispatch call
    sites hands to unit_sprites.sprite_pieces_for()/icon_for(), without
    needing a real install or decoded sprite -- both fakes return
    "unresolved" so every render path falls through to its normal
    no-sprite behaviour, and only the call log matters."""
    calls: dict[str, list[tuple[int, float]]] = {"sprite_pieces_for": [], "icon_for": []}

    def fake_sprite_pieces_for(unit_const, rotation, team_index, half_w):
        calls["sprite_pieces_for"].append((unit_const, rotation))
        return []

    def fake_icon_for(unit_const, rotation, team_index, footprint_w, footprint_h):
        calls["icon_for"].append((unit_const, rotation))
        return None

    monkeypatch.setattr(unit_sprites, "sprite_pieces_for", fake_sprite_pieces_for)
    monkeypatch.setattr(unit_sprites, "icon_for", fake_icon_for)
    return calls


def test_all_three_paths_resolve_the_same_unit_to_the_same_derived_rotation(wall_install, rotation_spy):
    scn = _scenario([[], [
        Unit(5.0, 5.0, WALL_CONST, _RADIAN[3]),
        Unit(6.0, 5.0, WALL_CONST, _RADIAN[3]),  # the unit under test: derives to index 0
        Unit(7.0, 5.0, WALL_CONST, _RADIAN[3]),
    ]])
    _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)

    render.sprite_draws_by_anchor(scn, proj, elevations)
    render._flat_icon_layer(scn, tile_px=16)
    img = render.render_terrain(scn)
    render.overlay_units(img, scn, with_sprites=True)

    for site, calls in rotation_spy.items():
        matching = [rotation for const, rotation in calls if const == WALL_CONST]
        assert 0.0 in matching, f"{site} never resolved the wall to the derived index: {matching}"
        # Every OTHER call for this const on this map must also be 2.0 -- the
        # two end units of the run have exactly one connector neighbour each
        # (not a through-run pair), which is also a derivable, overridden
        # shape (any single-neighbour mask -> index 2), just a different one
        # than the middle unit's.
        assert set(matching) <= {0.0, 2.0}, matching


@pytest.fixture
def real_gate_wall_install(monkeypatch):
    """Registers WALL_CONST as a variant-index, angle_count=5 connector
    ALONGSIDE the real gate consts already in WALL_CONNECTOR_CONSTS --
    unlike wall_install above, this does NOT replace that set: the whole
    point of the test below is exercising real gate membership (797/793 in,
    789 out), which a wholesale replacement would erase."""
    monkeypatch.setattr(
        unit_sprites, "_ROTATION_VARIANT_CONSTS",
        unit_sprites._ROTATION_VARIANT_CONSTS | frozenset({WALL_CONST}),
    )
    monkeypatch.setattr(
        unit_sprites, "graphic_map",
        lambda: {WALL_CONST: {"graphic_id": 1, "file_name": "wall_synth_x1",
                              "angle_count": 5, "mirroring_mode": 6, "frame_count": 1}},
    )
    unit_sprites.clear_caches()
    yield
    unit_sprites.clear_caches()


def test_a_real_gate_orientation_swap_changes_a_neighbouring_walls_connector_membership(
    real_gate_wall_install, rotation_spy
):
    """Retires the gate-orientation-cycling checklist's manual step 6: "cycle
    a gate that sits in a wall run and check the neighbouring wall pieces.
    This is the known WALL_CONNECTOR_CONSTS asymmetry; a palisade gate is the
    case to try."

    The first case in this file to use REAL gate consts rather than the
    synthetic WALL_CONST=9001 -- deliberately does NOT reuse the wall_install
    fixture above, which REPLACES WALL_CONNECTOR_CONSTS wholesale; this needs
    the real set's own asymmetry intact (AGENTS.md's hard rule: palisade
    closed's `("P", "A")` group is (789 ne, 797 e, 793 se, 801 n) in cycle
    order -- 797 and 793 are members of WALL_CONNECTOR_CONSTS, 789 and 801
    are not).

    span_low_corner()'s own invariant (descape/unit_model.py) is what makes
    this test possible without going through UnitEditModel.set_unit_const()
    at all: a gate's low-corner tile stays fixed across every orientation
    swap, and (measured directly) tile (4, 4) happens to be occupied by all
    three of 797/793/789's real footprints -- so the probe wall's own
    WEST/EAST neighbour tile never moves while only the gate's connector
    membership changes. That is the confounder any gate-cycling connector
    test has to watch for -- a cycle normally moves the gate's occupied tile
    set and its footprint together -- solved here by picking a tile all
    three orientations happen to share, not by re-deriving the geometry."""
    gate_e, gate_se, gate_ne = 797, 793, 789
    probe = Unit(3.0, 4.0, WALL_CONST, _RADIAN[3])
    gate = Unit(6.0, 6.0, gate_e)
    scn = _scenario([[], [probe, gate]])

    def _probe_rotations() -> set[float]:
        _, elevations, proj = render.render_terrain_iso_with_proj(scn, with_sprites=True)
        render.sprite_draws_by_anchor(scn, proj, elevations)
        render._flat_icon_layer(scn, tile_px=16)
        img = render.render_terrain(scn)
        render.overlay_units(img, scn, with_sprites=True)
        return {
            rotation
            for calls in rotation_spy.values()
            for const, rotation in calls
            if const == WALL_CONST
        }

    connector_rotations = _probe_rotations()
    assert connector_rotations == {2.0}, connector_rotations  # single EAST-only neighbour -> tower/corner

    for calls in rotation_spy.values():
        calls.clear()
    gate.unit_const, gate.x, gate.y = gate_se, 4.5, 6.0  # span_anchor() at the same low corner (4, 4)
    assert _probe_rotations() == connector_rotations, "797 -> 793 must not change connector membership"

    for calls in rotation_spy.values():
        calls.clear()
    gate.unit_const, gate.x, gate.y = gate_ne, 6.0, 4.5  # span_anchor() at the same low corner (4, 4)
    assert _RADIAN[3] in _probe_rotations(), (
        "793 -> 789 must drop the derived override (789 is outside WALL_CONNECTOR_CONSTS), "
        "falling back to the probe wall's own stored (radian) rotation -- the accepted asymmetry"
    )


def test_the_override_key_is_position_not_the_filtered_row_count(wall_install, rotation_spy):
    """The hazard step 4 of the execute plan calls out by name:
    _flat_icon_layer_sliced()'s `row` only advances past units unit_filter
    keeps, so keying the override table on `row` instead of this loop's own
    per-player `i` would silently misassign it as soon player 1's unit list
    has anything ahead of the wall that a filter removes."""
    decoy_const = 12345
    scn = _scenario([[], [
        Unit(0.0, 0.0, decoy_const, 0.0),  # index 0 in the list; filtered OUT below
        Unit(5.0, 5.0, WALL_CONST, _RADIAN[3]),  # index 1; the middle-of-run construction
        Unit(6.0, 5.0, WALL_CONST, _RADIAN[3]),  # is skipped here -- this one just needs
        Unit(5.0, 6.0, WALL_CONST, _RADIAN[1]),  # SOME non-zero mask so it has a derived
    ]])                                          # value distinct from its stored one.
    overrides = render.wall_variant_rotation_overrides(scn)
    assert (1, 1) in overrides, "sanity: the wall at list position 1 must have a derived override"

    # UnitFilter itself has no per-unit-const knob, so a small local predicate
    # stands in for "a filter that hides exactly the decoy" -- all that
    # matters is that it shifts `row` for every unit after it without
    # shifting `i`.
    class _HideDecoy:
        def matches(self, player_id, unit):
            return unit.unit_const != decoy_const

    render._flat_icon_layer(scn, tile_px=16, unit_filter=_HideDecoy())
    wall_calls = [rotation for const, rotation in rotation_spy["icon_for"] if const == WALL_CONST]
    assert overrides[(1, 1)] in wall_calls, (
        f"expected the position-1 override {overrides[(1, 1)]} to reach icon_for "
        f"even with the decoy filtered out; got {wall_calls} -- if this fails, the "
        f"override table is keyed on the filtered row count, not list position"
    )
