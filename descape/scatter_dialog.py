"""Edit > Scatter Units in Region…: which tiles of the committed Select
region are eligible, and the dialog that turns them into arguments for
scatter.scatter_units().

Eligibility lives here, not in scatter.py, which takes an explicit tile set
and never builds one (its own module docstring). The dialog's live "N
eligible tiles" count has to be exactly what gets placed, so the map clamp,
the terrain filter and the avoid-existing-units subtraction all happen here
and scatter receives a final list with avoid_occupied=False.

Terrain is read as mm.terrain[y * map_width + x], never through
MapManager.get_tile()/get_tile_safe(): get_tile() raises on every coordinate
of a non-square map and get_tile_safe() turns that into None, which would
silently report 0 eligible tiles there.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from AoE2ScenarioParser.datasets.terrains import TerrainId
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from descape import batch_api, object_catalog, scatter
from descape.terrain_palette import name_for_terrain_id
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID

RESTRICT_ANY = "any"
RESTRICT_WATER = "water"
RESTRICT_TERRAIN = "terrain"

# Strictly under scatter.MAX_JITTER (0.499), so the spin can never ask for
# more displacement than keeps a jittered unit inside its own tile.
MAX_JITTER_UI = 0.49
MAX_SEED = 2**31 - 1
# Density is a percentage in the UI. 1% of a single eligible tile still
# rounds up to one unit, so the Ok-implies-at-least-one rule holds.
MIN_DENSITY_PERCENT = 1


def region_tiles(region: tuple[int, int, int, int], width: int, height: int) -> list[tuple[int, int]]:
    """Every on-map tile of a half-open Select region (tx0, ty0, tx1, ty1).

    Clamped to the map here rather than left to scatter._normalize_tiles, so
    the live count and the placement see the same tiles.
    """
    tx0, ty0, tx1, ty1 = region
    x0, x1 = max(0, min(tx0, tx1)), min(width, max(tx0, tx1))
    y0, y1 = max(0, min(ty0, ty1)), min(height, max(ty0, ty1))
    return [(x, y) for y in range(y0, y1) for x in range(x0, x1)]


def eligible_tiles(
    scenario,
    region: tuple[int, int, int, int],
    restrict: str = RESTRICT_ANY,
    terrain_id: int | None = None,
) -> list[tuple[int, int]]:
    """The region's tiles that pass the chosen terrain filter."""
    mm = scenario.map_manager
    width, height = mm.map_width, mm.map_height
    tiles = region_tiles(region, width, height)
    if restrict == RESTRICT_ANY:
        return tiles
    if restrict == RESTRICT_WATER:
        return [(x, y) for x, y in tiles if batch_api.is_water(mm.terrain[y * width + x].terrain_id)]
    if restrict == RESTRICT_TERRAIN:
        if terrain_id is None:
            return []
        return [(x, y) for x, y in tiles if mm.terrain[y * width + x].terrain_id == terrain_id]
    raise ValueError(f"unknown restrict mode {restrict!r}")


def region_terrain_counts(scenario, region: tuple[int, int, int, int]) -> Counter:
    """How many tiles of each terrain_id the region holds."""
    mm = scenario.map_manager
    width = mm.map_width
    return Counter(mm.terrain[y * width + x].terrain_id for x, y in region_tiles(region, width, mm.map_height))


@dataclass(frozen=True)
class ScatterParams:
    """Exactly what ViewerWindow hands scatter.scatter_units()."""

    tiles: list[tuple[int, int]]
    unit_const: int
    player: int
    count: int | tuple[int, int] | None
    density: float | None
    seed: int | None
    jitter: float
    min_spacing: int

    def requested(self, eligible: int) -> int | None:
        """How many units were asked for, or None for a (lo, hi) range whose
        actual draw only the RNG knows."""
        if self.density is not None:
            return math.ceil(self.density * eligible)
        if isinstance(self.count, tuple):
            return None
        return self.count


class ScatterDialog(QDialog):
    """Modal setup for one scatter into an already-committed region.

    It never touches the map. `params()` reads the live widget state, so a
    test can drive the widgets and call it directly.
    """

    def __init__(self, scenario, region, unit_const, owner_id, parent=None, last=None):
        super().__init__(parent)
        self.setWindowTitle("Scatter Units in Region")
        self._scenario = scenario
        self._region = region
        self._unit_const = unit_const
        # Computed once, the first time Avoid is on, then kept for this
        # dialog's lifetime: occupied_tiles() walks all nine player lists.
        self._occupied: set[tuple[int, int]] | None = None
        self._eligible: list[tuple[int, int]] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Object: {object_catalog.display_name(unit_const)} ({unit_const})"))

        self.owner_combo = QComboBox()
        self.owner_combo.addItem("GAIA", GAIA_PLAYER_ID)
        for player_id in range(1, MAX_PLAYER_ID + 1):
            self.owner_combo.addItem(f"Player {player_id}", player_id)
        owner_index = self.owner_combo.findData(owner_id)
        self.owner_combo.setCurrentIndex(max(0, owner_index))
        owner_row = QHBoxLayout()
        owner_row.addWidget(QLabel("Owner:"))
        owner_row.addWidget(self.owner_combo, stretch=1)
        layout.addLayout(owner_row)

        layout.addWidget(self._build_restrict_group())
        layout.addWidget(self._build_amount_group())
        layout.addWidget(self._build_placement_group())

        self.count_label = QLabel("")
        layout.addWidget(self.count_label)
        note = QLabel(
            "Every unit is placed with rotation 0 and animation frame 0: for many "
            "GAIA objects rotation is a graphic-variant index, not an angle."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._restore(last or {})
        self._refresh_eligible()

    # -- construction ------------------------------------------------------

    def _build_restrict_group(self) -> QGroupBox:
        group = QGroupBox("Restrict to")
        box = QVBoxLayout(group)
        self.any_radio = QRadioButton("Any tile")
        self.water_radio = QRadioButton("Water only")
        self.terrain_radio = QRadioButton("Terrain:")
        self.any_radio.setChecked(True)
        box.addWidget(self.any_radio)
        box.addWidget(self.water_radio)

        self.terrain_combo = QComboBox()
        for terrain in sorted(TerrainId, key=lambda t: t.name):
            self.terrain_combo.addItem(name_for_terrain_id(terrain.value), terrain.value)
        self._select_default_terrain()
        self.terrain_combo.setEnabled(False)
        terrain_row = QHBoxLayout()
        terrain_row.addWidget(self.terrain_radio)
        terrain_row.addWidget(self.terrain_combo, stretch=1)
        box.addLayout(terrain_row)

        for radio in (self.any_radio, self.water_radio, self.terrain_radio):
            radio.toggled.connect(self._on_restrict_changed)
        self.terrain_combo.currentIndexChanged.connect(self._refresh_eligible)
        return group

    def _select_default_terrain(self) -> None:
        """Defaults to whichever terrain the region holds most of. A raw id
        no TerrainId member covers leaves the combo on its first entry."""
        counts = region_terrain_counts(self._scenario, self._region)
        if not counts:
            return
        index = self.terrain_combo.findData(counts.most_common(1)[0][0])
        if index >= 0:
            self.terrain_combo.setCurrentIndex(index)

    def _build_amount_group(self) -> QGroupBox:
        group = QGroupBox("Amount")
        form = QFormLayout(group)
        self.count_radio = QRadioButton("Count:")
        self.count_radio.setChecked(True)
        self.count_spin = QSpinBox()
        # Minimum 1, so an accepted dialog can never ask for zero units: an
        # empty _unit_edit would record a phantom undo step.
        self.count_spin.setRange(1, 100000)
        self.count_spin.setValue(40)
        self.range_check = QCheckBox("up to")
        self.count_hi_spin = QSpinBox()
        self.count_hi_spin.setRange(1, 100000)
        self.count_hi_spin.setValue(40)
        self.count_hi_spin.setEnabled(False)
        count_row = QHBoxLayout()
        count_row.addWidget(self.count_spin, stretch=1)
        count_row.addWidget(self.range_check)
        count_row.addWidget(self.count_hi_spin, stretch=1)
        form.addRow(self.count_radio, count_row)

        self.density_radio = QRadioButton("Density:")
        self.density_spin = QSpinBox()
        self.density_spin.setRange(MIN_DENSITY_PERCENT, 100)
        self.density_spin.setValue(10)
        self.density_spin.setSuffix(" % of eligible tiles")
        self.density_spin.setEnabled(False)
        form.addRow(self.density_radio, self.density_spin)

        self.count_radio.toggled.connect(self._on_amount_changed)
        self.range_check.toggled.connect(self._on_amount_changed)
        return group

    def _build_placement_group(self) -> QGroupBox:
        group = QGroupBox("Placement")
        form = QFormLayout(group)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, MAX_SEED)
        self.seed_spin.setValue(7)
        self.randomize_button = QPushButton("Randomize")
        self.randomize_button.clicked.connect(self._on_randomize)
        self.random_check = QCheckBox("Random each time")
        self.random_check.toggled.connect(self._on_random_toggled)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_spin, stretch=1)
        seed_row.addWidget(self.randomize_button)
        seed_row.addWidget(self.random_check)
        form.addRow(QLabel("Seed:"), seed_row)

        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, MAX_JITTER_UI)
        self.jitter_spin.setSingleStep(0.05)
        self.jitter_spin.setDecimals(2)
        form.addRow(QLabel("Jitter:"), self.jitter_spin)

        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(0, 64)
        self.spacing_spin.setSpecialValueText("off")
        form.addRow(QLabel("Min spacing:"), self.spacing_spin)

        self.avoid_check = QCheckBox("Avoid tiles under existing units")
        self.avoid_check.setChecked(True)
        self.avoid_check.toggled.connect(self._refresh_eligible)
        form.addRow(self.avoid_check)
        return group

    # -- live state --------------------------------------------------------

    def restrict_mode(self) -> str:
        if self.water_radio.isChecked():
            return RESTRICT_WATER
        if self.terrain_radio.isChecked():
            return RESTRICT_TERRAIN
        return RESTRICT_ANY

    def eligible(self) -> list[tuple[int, int]]:
        return self._eligible

    def _on_restrict_changed(self) -> None:
        self.terrain_combo.setEnabled(self.terrain_radio.isChecked())
        self._refresh_eligible()

    def _on_amount_changed(self) -> None:
        self.count_spin.setEnabled(self.count_radio.isChecked())
        self.range_check.setEnabled(self.count_radio.isChecked())
        self.count_hi_spin.setEnabled(self.count_radio.isChecked() and self.range_check.isChecked())
        self.density_spin.setEnabled(self.density_radio.isChecked())

    def _on_random_toggled(self, checked: bool) -> None:
        self.seed_spin.setEnabled(not checked)
        self.randomize_button.setEnabled(not checked)

    def _on_randomize(self) -> None:
        self.seed_spin.setValue(random.randint(0, MAX_SEED))

    def _refresh_eligible(self) -> None:
        tiles = eligible_tiles(self._scenario, self._region, self.restrict_mode(), self.terrain_combo.currentData())
        if self.avoid_check.isChecked():
            if self._occupied is None:
                self._occupied = scatter.occupied_tiles(self._scenario)
            tiles = [t for t in tiles if t not in self._occupied]
        self._eligible = tiles
        mm = self._scenario.map_manager
        span = region_tiles(self._region, mm.map_width, mm.map_height)
        cols = len({x for x, _ in span})
        rows = len({y for _, y in span})
        self.count_label.setText(f"{len(tiles)} eligible tiles in a {cols}x{rows} region ({len(span)} tiles)")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(tiles))

    # -- results -----------------------------------------------------------

    def params(self) -> ScatterParams:
        if self.density_radio.isChecked():
            count: int | tuple[int, int] | None = None
            density: float | None = self.density_spin.value() / 100.0
        else:
            density = None
            lo = self.count_spin.value()
            if self.range_check.isChecked():
                count = (min(lo, self.count_hi_spin.value()), max(lo, self.count_hi_spin.value()))
            else:
                count = lo
        return ScatterParams(
            tiles=list(self._eligible),
            unit_const=self._unit_const,
            player=self.owner_combo.currentData(),
            count=count,
            density=density,
            seed=None if self.random_check.isChecked() else self.seed_spin.value(),
            jitter=self.jitter_spin.value(),
            min_spacing=self.spacing_spin.value(),
        )

    def state(self) -> dict:
        """Session-sticky values, restored by the next dialog. Nothing here
        is persisted to settings, so there is no MEMOIZED_GLOBALS key."""
        return {
            "restrict": self.restrict_mode(),
            "use_density": self.density_radio.isChecked(),
            "count": self.count_spin.value(),
            "use_range": self.range_check.isChecked(),
            "count_hi": self.count_hi_spin.value(),
            "density_percent": self.density_spin.value(),
            "seed": self.seed_spin.value(),
            "random_seed": self.random_check.isChecked(),
            "jitter": self.jitter_spin.value(),
            "min_spacing": self.spacing_spin.value(),
            "avoid": self.avoid_check.isChecked(),
        }

    def _restore(self, last: dict) -> None:
        restrict = last.get("restrict", RESTRICT_ANY)
        self.water_radio.setChecked(restrict == RESTRICT_WATER)
        self.terrain_radio.setChecked(restrict == RESTRICT_TERRAIN)
        self.any_radio.setChecked(restrict == RESTRICT_ANY)
        # The terrain choice is deliberately NOT restored: it always defaults
        # to whatever this region holds most of, which a sticky id from an
        # unrelated region would override.
        self.density_radio.setChecked(bool(last.get("use_density", False)))
        self.count_radio.setChecked(not last.get("use_density", False))
        self.count_spin.setValue(last.get("count", self.count_spin.value()))
        self.range_check.setChecked(bool(last.get("use_range", False)))
        self.count_hi_spin.setValue(last.get("count_hi", self.count_hi_spin.value()))
        self.density_spin.setValue(last.get("density_percent", self.density_spin.value()))
        self.seed_spin.setValue(last.get("seed", self.seed_spin.value()))
        self.random_check.setChecked(bool(last.get("random_seed", False)))
        self.jitter_spin.setValue(last.get("jitter", 0.0))
        self.spacing_spin.setValue(last.get("min_spacing", 0))
        self.avoid_check.setChecked(bool(last.get("avoid", True)))
        # The toggled signals above only fire on a real change, so the
        # enabled states are settled explicitly here.
        self._on_restrict_changed()
        self._on_amount_changed()
        self._on_random_toggled(self.random_check.isChecked())
