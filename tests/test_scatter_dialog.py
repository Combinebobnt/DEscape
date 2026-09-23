"""descape/scatter_dialog.py: region eligibility and the Scatter dialog.

Eligibility runs against synthetic grids (testkit.fakes), the dialog against
an offscreen QDialog. Nothing here loads a real scenario: the write path and
the undo step are test_unit_edit_viewer.py's job.
"""

from __future__ import annotations

import pytest

from descape import scatter, scatter_dialog
from testkit import fakes

import conftest

# TerrainId.WATER_SHALLOW / GRASS / DIRT_3, by value: is_water() is a
# keyword match on the enum name, so the ids themselves are what matter.
_WATER = 1
_GRASS = 0
_DIRT = 6

# A GAIA fish, the motivating case, and a house-sized building whose
# footprint covers more than its own tile.
_FISH = 456
_HOUSE = 70


def _scenario(width=20, height=20, water=(), dirt=(), units=None):
    tiles = []
    for y in range(height):
        for x in range(width):
            if (x, y) in water:
                terrain_id = _WATER
            elif (x, y) in dirt:
                terrain_id = _DIRT
            else:
                terrain_id = _GRASS
            tiles.append(fakes.SyntheticTile(x, y, 0, terrain_id))
    return fakes.FakeScenario(width, height, tiles, units or [[] for _ in range(9)])


def _blob(x0, y0, x1, y1):
    return {(x, y) for y in range(y0, y1) for x in range(x0, x1)}


# -- eligibility --------------------------------------------------------------


def test_any_returns_the_whole_region() -> None:
    scenario = _scenario()
    tiles = scatter_dialog.eligible_tiles(scenario, (2, 3, 6, 9), scatter_dialog.RESTRICT_ANY)
    assert set(tiles) == _blob(2, 3, 6, 9)
    assert len(tiles) == 4 * 6


def test_water_returns_exactly_the_water_tiles_inside_the_region() -> None:
    pond = _blob(4, 4, 8, 8)
    scenario = _scenario(water=pond)
    tiles = scatter_dialog.eligible_tiles(scenario, (0, 0, 6, 6), scatter_dialog.RESTRICT_WATER)
    assert set(tiles) == pond & _blob(0, 0, 6, 6)
    assert set(tiles) == _blob(4, 4, 6, 6)


def test_terrain_returns_exactly_that_id_inside_the_region() -> None:
    patch = _blob(3, 3, 5, 5)
    scenario = _scenario(dirt=patch)
    tiles = scatter_dialog.eligible_tiles(scenario, (0, 0, 10, 10), scatter_dialog.RESTRICT_TERRAIN, _DIRT)
    assert set(tiles) == patch
    grass = scatter_dialog.eligible_tiles(scenario, (0, 0, 10, 10), scatter_dialog.RESTRICT_TERRAIN, _GRASS)
    assert set(grass) == _blob(0, 0, 10, 10) - patch


def test_terrain_with_no_id_chosen_is_empty() -> None:
    scenario = _scenario()
    assert scatter_dialog.eligible_tiles(scenario, (0, 0, 4, 4), scatter_dialog.RESTRICT_TERRAIN, None) == []


def test_an_unknown_restrict_mode_raises() -> None:
    with pytest.raises(ValueError):
        scatter_dialog.eligible_tiles(_scenario(), (0, 0, 4, 4), "everything")


def test_a_region_hanging_off_the_map_edge_is_clamped() -> None:
    scenario = _scenario(width=10, height=10)
    tiles = scatter_dialog.region_tiles((-4, -3, 12, 5), 10, 10)
    assert set(tiles) == _blob(0, 0, 10, 5)
    assert scatter_dialog.eligible_tiles(scenario, (-4, -3, 12, 5)) == tiles


def test_a_reversed_region_is_normalized() -> None:
    assert set(scatter_dialog.region_tiles((6, 8, 2, 4), 20, 20)) == _blob(2, 4, 6, 8)


def test_a_non_square_map_still_finds_eligible_tiles() -> None:
    """The flat-index read, pinned against the get_tile() trap: get_tile()
    raises on every coordinate of a non-square map, so a dialog built on it
    would silently report zero eligible tiles here."""
    pond = _blob(100, 60, 110, 70)
    scenario = _scenario(width=120, height=80, water=pond)
    tiles = scatter_dialog.eligible_tiles(scenario, (95, 55, 120, 80), scatter_dialog.RESTRICT_WATER)
    assert set(tiles) == pond
    assert len(scatter_dialog.eligible_tiles(scenario, (0, 0, 120, 80))) == 120 * 80


def test_region_terrain_counts_reports_the_regions_own_mix() -> None:
    scenario = _scenario(water=_blob(0, 0, 4, 4))
    counts = scatter_dialog.region_terrain_counts(scenario, (0, 0, 6, 6))
    assert counts[_WATER] == 16
    assert counts[_GRASS] == 36 - 16


# -- the dialog ---------------------------------------------------------------

pytestmark_gui = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


def _dialog(scenario, region=(0, 0, 10, 10), unit_const=_FISH, owner_id=0, last=None):
    conftest.ensure_qapp()
    return scatter_dialog.ScatterDialog(scenario, region, unit_const, owner_id, last=last)


def _ok_enabled(dialog) -> bool:
    from PyQt5.QtWidgets import QDialogButtonBox

    return dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()


@pytest.mark.gui
@pytestmark_gui
def test_zero_eligible_tiles_disables_ok() -> None:
    dialog = _dialog(_scenario())
    try:
        assert _ok_enabled(dialog)
        dialog.water_radio.setChecked(True)  # an all-grass map has no water
        assert dialog.eligible() == []
        assert not _ok_enabled(dialog)
        dialog.any_radio.setChecked(True)
        assert _ok_enabled(dialog)
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_count_and_density_are_mutually_exclusive_in_params() -> None:
    """scatter_units raises on both-or-neither, so one radio group is what
    keeps that ValueError unreachable from the GUI."""
    dialog = _dialog(_scenario())
    try:
        params = dialog.params()
        assert params.count == 40 and params.density is None
        dialog.density_radio.setChecked(True)
        params = dialog.params()
        assert params.count is None and params.density == pytest.approx(0.10)
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_an_accepted_dialog_can_never_ask_for_zero_units() -> None:
    """An empty edit would record a phantom undo step, so the spins float
    off zero and density rounds up."""
    dialog = _dialog(_scenario())
    try:
        dialog.count_spin.setValue(0)
        assert dialog.params().count == 1
        dialog.density_radio.setChecked(True)
        dialog.density_spin.setValue(0)
        params = dialog.params()
        assert params.density == pytest.approx(0.01)
        assert params.requested(1) == 1
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_a_reversed_count_range_comes_out_ordered() -> None:
    dialog = _dialog(_scenario())
    try:
        dialog.range_check.setChecked(True)
        dialog.count_spin.setValue(30)
        dialog.count_hi_spin.setValue(10)
        params = dialog.params()
        assert params.count == (10, 30)
        assert params.requested(500) is None  # only the RNG knows the draw
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_random_each_time_gives_seed_none() -> None:
    dialog = _dialog(_scenario())
    try:
        assert dialog.params().seed == 7
        dialog.random_check.setChecked(True)
        assert dialog.params().seed is None
        assert not dialog.seed_spin.isEnabled()
        assert not dialog.randomize_button.isEnabled()
        dialog.random_check.setChecked(False)
        assert dialog.params().seed == 7
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_randomize_changes_the_seed_within_range() -> None:
    dialog = _dialog(_scenario())
    try:
        seeds = set()
        for _ in range(8):
            dialog._on_randomize()
            seeds.add(dialog.params().seed)
        assert len(seeds) > 1
        assert all(0 <= s <= scatter_dialog.MAX_SEED for s in seeds)
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_the_jitter_spin_cannot_exceed_scatters_own_cap() -> None:
    dialog = _dialog(_scenario())
    try:
        dialog.jitter_spin.setValue(10.0)
        assert dialog.params().jitter <= scatter.MAX_JITTER
        assert dialog.jitter_spin.maximum() == pytest.approx(scatter_dialog.MAX_JITTER_UI)
        assert scatter_dialog.MAX_JITTER_UI < scatter.MAX_JITTER
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_avoid_subtracts_exactly_the_occupied_tiles() -> None:
    units = [[] for _ in range(9)]
    units[1] = [fakes.SyntheticUnit(x=3.5, y=3.5, unit_const=_HOUSE, reference_id=11)]
    scenario = _scenario(units=units)
    covered = scatter.occupied_tiles(scenario)
    assert covered, "the synthetic building must cover at least its own tile"
    dialog = _dialog(scenario, region=(0, 0, 8, 8))
    try:
        assert dialog.avoid_check.isChecked()  # default on
        with_avoid = set(dialog.eligible())
        dialog.avoid_check.setChecked(False)
        without_avoid = set(dialog.eligible())
        assert without_avoid == _blob(0, 0, 8, 8)
        assert with_avoid == without_avoid - covered
        assert with_avoid != without_avoid
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_a_fully_covered_region_disables_ok() -> None:
    units = [[] for _ in range(9)]
    units[1] = [
        fakes.SyntheticUnit(x=x + 0.5, y=y + 0.5, unit_const=_FISH, reference_id=100 + x * 4 + y)
        for x in range(2)
        for y in range(2)
    ]
    dialog = _dialog(_scenario(units=units), region=(0, 0, 2, 2))
    try:
        assert dialog.eligible() == []
        assert not _ok_enabled(dialog)
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_the_terrain_combo_defaults_to_the_regions_commonest_terrain() -> None:
    scenario = _scenario(dirt=_blob(0, 0, 5, 5))
    dialog = _dialog(scenario, region=(0, 0, 4, 4))
    try:
        assert dialog.terrain_combo.currentData() == _DIRT
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_the_terrain_default_is_not_sticky() -> None:
    """Sticky values must not defeat the per-region default: a dirt id kept
    from an earlier scatter would be wrong for the next region."""
    scenario = _scenario(dirt=_blob(0, 0, 5, 5))
    first = _dialog(scenario, region=(0, 0, 4, 4))
    try:
        first.terrain_radio.setChecked(True)
        assert first.terrain_combo.currentData() == _DIRT
        state = first.state()
    finally:
        first.deleteLater()
    second = _dialog(scenario, region=(10, 10, 14, 14), last=state)
    try:
        assert second.restrict_mode() == scatter_dialog.RESTRICT_TERRAIN
        assert second.terrain_combo.currentData() == _GRASS
    finally:
        second.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_the_live_count_label_reports_eligible_and_region_size() -> None:
    scenario = _scenario(water=_blob(0, 0, 3, 4))
    dialog = _dialog(scenario, region=(0, 0, 5, 4))
    try:
        dialog.water_radio.setChecked(True)
        assert "12 eligible tiles in a 5x4 region" in dialog.count_label.text()
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_the_owner_combo_starts_on_the_units_panels_owner() -> None:
    dialog = _dialog(_scenario(), owner_id=3)
    try:
        assert dialog.params().player == 3
        dialog.owner_combo.setCurrentIndex(dialog.owner_combo.findData(0))
        assert dialog.params().player == 0
    finally:
        dialog.deleteLater()


@pytest.mark.gui
@pytestmark_gui
def test_state_round_trips_into_the_next_dialog() -> None:
    scenario = _scenario(water=_blob(0, 0, 6, 6))
    first = _dialog(scenario)
    try:
        first.water_radio.setChecked(True)
        first.count_spin.setValue(12)
        first.jitter_spin.setValue(0.25)
        first.spacing_spin.setValue(3)
        first.avoid_check.setChecked(False)
        state = first.state()
    finally:
        first.deleteLater()
    second = _dialog(scenario, last=state)
    try:
        assert second.restrict_mode() == scatter_dialog.RESTRICT_WATER
        params = second.params()
        assert params.count == 12
        assert params.jitter == pytest.approx(0.25)
        assert params.min_spacing == 3
        assert not second.avoid_check.isChecked()
    finally:
        second.deleteLater()
