"""GH #151: top-down Flat draws walls and gates along the square grid.

unit_sprites.flat_art() turns a wall's shape index and a gate's orientation
one 45-degree step at icon_for()'s door. The synthetic tests pin the turn and
its wiring; whether the frame it picks really points the Flat way can only be
measured on game art, which is this file. Each check runs through icon_for(),
the only entry point top-down Flat has, and reads the icon's own alpha with
tools/verify_flat_facing.py's helpers.

Measured on the install 2026-10-19 (icon at 128 px, a gate's tiles at 64 px):
every wall's stored-0 icon is the widest of its five and stored-1 the
narrowest; stored-3's foot line falls and stored-4's rises (smallest gap 0.105
px/px, const 119). Fence 1062 reads 1.1 / 89.1 / 34.7 / 145.9 degrees at
stored 0 / 1 / 3 / 4. Gates read 179.2-179.9 (A), 90.4-91.3 (B), 30.2-54.5 (C)
and 126.8-148.4 (D) over states A, B and X. The diagonals' slack is the iso
camera's 2:1 squash, which no frame choice removes.

Needs AOE2DE_INSTALL_PATH rather than the configured install, for the reason
tests/test_sprite_variant14_real_assets.py's docstring spells out. Not
`corpus`-marked, per tests/test_wall_variants_real_assets.py: it decodes a few
dozen frames already on disk, so it earns the default tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import asset_source, gate_orientation, terrain_palette, unit_sprites

import conftest

WALL_PX = 128
GATE_TILE_PX = 64
FENCE = 1062
# The cardinals' tolerance is verify_flat_facing's; the diagonals carry the squash.
DIAGONAL_TOLERANCE_DEG = 20.0
# Each orientation's Flat screen axis, in gate_orientation's A, C, B, D cycle order.
_GATE_AXES = (0.0, 45.0, 90.0, 135.0)


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


@pytest.fixture
def tool():
    _require_install()
    return conftest.load_verify_module("verify_flat_facing")


def _wall_alpha(const: int, shape: int) -> np.ndarray:
    icon = unit_sprites.icon_for(const, float(shape), 0, WALL_PX, WALL_PX)
    assert icon is not None, f"wall {const} shape {shape} drew no icon"
    return icon.rgba[..., 3]


def _foot_slope(alpha: np.ndarray) -> float:
    """The least-squares slope of each column's lowest ink row, y down."""
    cols = np.flatnonzero((alpha > 0).any(axis=0))
    feet = [np.flatnonzero(alpha[:, c] > 0)[-1] for c in cols]
    return float(np.polyfit(cols.astype(float), np.array(feet, dtype=float), 1)[0])


@pytest.mark.parametrize("const", unit_sprites.wall_family_consts())
def test_every_wall_shape_draws_a_flat_icon(tool, const):
    for shape in range(5):
        _wall_alpha(const, shape)


@pytest.mark.parametrize("const", unit_sprites.wall_family_consts())
def test_a_flat_wall_runs_wide_along_x_narrow_along_y_and_its_diagonals_lean_apart(tool, const):
    """x-runs draw the widest icon, y-runs the narrowest, and a +x+y run's
    foot line falls left to right where a +x-y run's rises."""
    alphas = [_wall_alpha(const, shape) for shape in range(5)]
    aspects = [a.shape[1] / a.shape[0] for a in alphas]
    assert max(range(5), key=aspects.__getitem__) == 0, f"aspects w/h {aspects}"
    assert min(range(5), key=aspects.__getitem__) == 1, f"aspects w/h {aspects}"
    falls, rises = _foot_slope(alphas[3]), _foot_slope(alphas[4])
    assert falls > rises, f"stored 3 foot slope {falls:.3f}, stored 4 {rises:.3f}"


def test_the_fence_runs_along_its_flat_axes(tool):
    """Fence is the thinnest wall, so its principal axis is its run."""
    want = {0: (0.0, tool.TOLERANCE_DEG), 1: (90.0, tool.TOLERANCE_DEG),
            3: (45.0, DIAGONAL_TOLERANCE_DEG), 4: (135.0, DIAGONAL_TOLERANCE_DEG)}
    for shape, (axis, tolerance) in want.items():
        got = tool.principal_axis_deg(_wall_alpha(FENCE, shape))
        assert tool.axis_gap_deg(got, axis) <= tolerance, f"shape {shape}: axis {got:.2f}, want {axis}"


def _gate_icon(const: int):
    span_x, span_y = terrain_palette.tile_span(const, (1, 1))
    icon = unit_sprites.icon_for(const, 0.0, 0, span_x * GATE_TILE_PX, span_y * GATE_TILE_PX)
    assert icon is not None, f"gate {const} drew no icon"
    assert icon.rgba.shape[1] <= span_x * GATE_TILE_PX and icon.rgba.shape[0] <= span_y * GATE_TILE_PX
    return icon


def _groups(states: str):
    groups = gate_orientation.groups()
    assert len(groups) == 24
    return [(key, consts) for key, consts in sorted(groups.items()) if key[1] in states]


@pytest.mark.parametrize(("key", "consts"), _groups("ABX"), ids=str)
def test_a_flat_gate_lies_along_its_own_footprint(tool, key, consts):
    for const, axis in zip(consts, _GATE_AXES, strict=True):
        got = tool.principal_axis_deg(_gate_icon(const).rgba[..., 3])
        tolerance = tool.TOLERANCE_DEG if axis in (0.0, 90.0) else DIAGONAL_TOLERANCE_DEG
        assert tool.axis_gap_deg(got, axis) <= tolerance, f"gate {const} {key}: axis {got:.2f}, want {axis}"


@pytest.mark.parametrize(("key", "consts"), _groups("C"), ids=str)
def test_a_flat_corner_gate_is_one_icon_across_its_four_siblings(tool, key, consts):
    """State C's four siblings share one graphic and one 1x1 span, so the turn
    is a visual no-op there; it must still draw."""
    icons = [_gate_icon(const).rgba for const in consts]
    for const, rgba in zip(consts[1:], icons[1:], strict=True):
        assert np.array_equal(rgba, icons[0]), f"gate {const} differs from {consts[0]}"


def _parent_file(entry: dict) -> str:
    """The file whose absence drops the whole icon; a missing non-parent
    piece (a flag) is skipped, see _native_pieces_for()."""
    parents = [p["file_name"] for p in entry.get("pieces", ()) if p.get("parent")]
    return str(parents[0] if parents else entry["file_name"])


def test_no_gate_loses_its_icon_to_the_turn(tool):
    """For every gate const and owner art pair, wherever the unturned art
    resolves to a parent file this install has, the turned sibling's entry
    resolves too and icon_for() draws it.

    Measured 2026-10-19: 48 of the pairs (the X-state C orientation, owned
    by a Chronicles civ 46-56) turn to a sibling whose civ flag piece is
    missing on this install, so that icon draws without its flag. The same
    sibling placed directly draws without it in every view already."""
    art = unit_sprites.building_art()
    civs = [None, *sorted(art.civ_art)]
    checked, lost = 0, []
    for consts in gate_orientation.groups().values():
        for const in consts:
            turned = gate_orientation.cycle_const(const, 1)
            span_x, span_y = terrain_palette.tile_span(const, (1, 1))
            for civ in civs:
                for age in range(2, 6):
                    before = unit_sprites.resolve_entry(const, civ, age)
                    if before is None or unit_sprites.sld_frame_count(_parent_file(before)) is None:
                        continue
                    checked += 1
                    after = unit_sprites.resolve_entry(turned, civ, age)
                    icon = unit_sprites.icon_for(
                        const, 0.0, 0, span_x * GATE_TILE_PX, span_y * GATE_TILE_PX, art=(civ, age)
                    )
                    if after is None or icon is None:
                        lost.append((const, civ, age))
    assert checked, "no gate art resolved at all, so this proves nothing"
    assert not lost, f"{len(lost)} of {checked} lose their icon, first {lost[:5]}"
