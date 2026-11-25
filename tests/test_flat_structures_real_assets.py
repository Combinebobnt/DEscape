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

from descape import (
    asset_source,
    cliff_catalog,
    flat_deproject,
    game_dataset,
    games,
    gate_orientation,
    terrain_palette,
    unit_sprites,
)

import conftest

WALL_PX = 128
GATE_TILE_PX = 64
FENCE = 1062
AQUEDUCT = 231
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


def test_the_aqueduct_runs_along_its_flat_axes(tool):
    """Aqueduct 231, a wall since GH #110, is turned by flat_art() too.
    Measured 2026-10-20: 178.2 / 90.2 / 68.8 / 111.2 degrees at stored
    0 / 1 / 3 / 4. Its diagonal icons are 74x128 to the fence's 123x128, so
    they sit past the diagonal slack; they must still lean apart the right way."""
    assert AQUEDUCT in unit_sprites.wall_family_consts()
    axes = {shape: tool.principal_axis_deg(_wall_alpha(AQUEDUCT, shape)) for shape in (0, 1, 3, 4)}
    assert tool.axis_gap_deg(axes[0], 0.0) <= tool.TOLERANCE_DEG, axes
    assert tool.axis_gap_deg(axes[1], 90.0) <= tool.TOLERANCE_DEG, axes
    assert 0.0 < axes[3] < 90.0 < axes[4] < 180.0, axes


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


def _drawn_pieces(const, civ, age) -> int:
    # A missing non-parent piece (a flag) is skipped, a missing parent drops them all (_native_pieces_for()).
    return len(unit_sprites.sprite_pieces_for(const, 0.0, 1, 48, art=(civ, age, civ)) or ())


def _entry_pieces(entry: dict) -> int:
    return len(entry.get("pieces") or ()) or 1  # a single-graphic entry draws one piece


def test_no_gate_loses_a_piece_to_the_turn(tool):
    """For every gate const and owner art pair whose unturned art draws every
    piece of its entry on this install, the turned sibling draws every piece
    of its own too (TASK-433: a missing flag file dropped the pennants), and
    icon_for() still draws the turned icon."""
    art = unit_sprites.building_art()
    civs = [None, *sorted(art.civ_art)]
    checked, lost = 0, []
    for consts in gate_orientation.groups().values():
        for const in consts:
            turned = gate_orientation.cycle_const(const, 1)
            span_x, span_y = terrain_palette.tile_span(const, (1, 1))
            for civ in civs:
                for age in range(2, 7):
                    before = unit_sprites.resolve_entry(const, civ, age, civ)
                    if before is None or _drawn_pieces(const, civ, age) != _entry_pieces(before):
                        continue
                    checked += 1
                    after = unit_sprites.resolve_entry(turned, civ, age, civ)
                    icon = unit_sprites.icon_for(
                        const, 0.0, 0, span_x * GATE_TILE_PX, span_y * GATE_TILE_PX, art=(civ, age, civ)
                    )
                    if (
                        after is None or icon is None
                        or _drawn_pieces(turned, civ, age) != _entry_pieces(after)
                    ):
                        lost.append((const, civ, age))
    assert checked, "no gate art resolved at all, so this proves nothing"
    assert not lost, f"{len(lost)} of {checked} lose a piece or their icon, first {lost[:5]}"


# --- thin cliff straights (TASK-350) ------------------------------------
# No cliff frame runs along a tile diagonal, so icon_for() de-projects the
# thin straights (unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES) onto their
# footprint. Measured on the install 2026-11-25 at 32 px tiles, both games (22
# thin consts in AoE2, 14 in Return of Rome): 3x1 frames 10/11 read 178.9-179.9
# or 0.1 degrees, 1x3 frames 7/8 read 87.7-91.0, and 0.85-0.95 of the ink lands
# inside except Limestone's 3x1 frame 11 at 0.75. Dropping the de-project branch
# turns the first test red in both games; de-projecting every frame, the second.

CLIFF_TILE_PX = 32
# A frame keeps this much of its unclipped de-projected ink on its own footprint.
CLIFF_INSIDE_MIN = 0.7
# The frames that run along each thin span's long side: 3x1 along x, 1x3 along y.
_CLIFF_RUN = {(3, 1): ((10, 11), 0.0), (1, 3): ((7, 8), 90.0)}


@pytest.fixture(params=[game.id for game in games.GAMES])
def cliff_game(request, tool):
    """Each registered game's dataset active, skipped where its root has no resources."""
    game = games.get(request.param)
    if not (game.root(asset_source.get_install_path()) / "resources").is_dir():
        pytest.skip(f"the install has no {game.display_name} resources")
    with game_dataset.activated(game.id):
        unit_sprites.clear_caches()
        yield tool
    unit_sprites.clear_caches()


def _thin_cliffs() -> list[tuple[str, int, tuple[int, int]]]:
    """(family, const, span) for every cliff piece with a 1 in its span (suffixes 2 and 3)."""
    out = []
    for family, pieces in cliff_catalog.families().items():
        for piece in pieces:
            span = terrain_palette.tile_span(piece.unit_const, (1, 1))
            if span in _CLIFF_RUN:
                out.append((family, piece.unit_const, span))
    assert out, "no thin cliff piece in this game, so this proves nothing"
    return out


def _inside_share(const: int, frame: int, span: tuple[int, int]) -> float:
    """The share of the frame's de-projected ink, unclipped, that lands on its footprint (native scale)."""
    entry = unit_sprites.graphic_map()[const]
    (piece,) = unit_sprites._native_pieces_for(const, entry, float(frame), unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG)
    tile, pad = unit_sprites.NATIVE_TILE_W // 2, 6
    sx, sy = span
    out = flat_deproject.deprojected(
        piece.main, piece.hotspot_x, piece.hotspot_y, tile, tile, (sx + 2 * pad) * tile, (sy + 2 * pad) * tile,
    )
    ink = out[..., 3] > 0
    return float(ink[pad * tile : (pad + sy) * tile, pad * tile : (pad + sx) * tile].sum() / max(1, ink.sum()))


def test_every_thin_cliff_straight_lies_along_its_flat_footprint(cliff_game):
    """Frames 7, 8, 10 and 11 draw on every thin piece of every family; the
    two that run along the piece's long side read that axis and keep at
    least CLIFF_INSIDE_MIN of their ink on the footprint."""
    tool, failures, checked = cliff_game, [], 0
    for family, const, span in _thin_cliffs():
        runs, axis = _CLIFF_RUN[span]
        fw, fh = span[0] * CLIFF_TILE_PX, span[1] * CLIFF_TILE_PX
        for frame in sorted(unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES):
            icon = unit_sprites.icon_for(const, float(frame), 0, fw, fh)
            if icon is None:
                failures.append(f"{family} {const} frame {frame}: no icon")
                continue
            if frame not in runs:
                continue
            checked += 1
            got = tool.principal_axis_deg(icon.rgba[..., 3])
            if tool.axis_gap_deg(got, axis) > tool.TOLERANCE_DEG:
                failures.append(f"{family} {const} frame {frame}: axis {got:.1f}, want {axis}")
            share = _inside_share(const, frame, span)
            if share < CLIFF_INSIDE_MIN:
                failures.append(f"{family} {const} frame {frame}: {share:.2f} of its ink inside")
    assert checked, "no thin straight was measured"
    assert not failures, f"{len(failures)} of {checked}: {failures}"


def test_every_other_frame_of_a_thin_cliff_keeps_its_contain_fit(cliff_game):
    """Every frame outside FLAT_DEPROJECT_CLIFF_FRAMES draws exactly as _build_icon() would."""
    differ, checked = [], 0
    for family, const, span in _thin_cliffs():
        entry = unit_sprites.graphic_map()[const]
        fw, fh = span[0] * CLIFF_TILE_PX, span[1] * CLIFF_TILE_PX
        for frame in range(cliff_catalog.frame_count(const)):
            if frame in unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES:
                continue
            icon = unit_sprites.icon_for(const, float(frame), 0, fw, fh)
            want = unit_sprites._build_icon(
                const, entry, float(frame), unit_sprites.TEAM_COLORS[0], fw, fh,
                unit_sprites.FLAT_ANGLE_ZERO_OFFSET_DEG,
            )
            checked += want is not None
            if (icon is None) != (want is None) or (icon is not None and not np.array_equal(icon.rgba, want.rgba)):
                differ.append(f"{family} {const} frame {frame}")
    assert checked, "no contain-fitted frame drew, so this proves nothing"
    assert not differ, f"{len(differ)} of {checked} differ from the contain-fit: {differ[:5]}"
