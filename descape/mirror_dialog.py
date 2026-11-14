"""The Mirror Map dialog, over mirror_tools' plans.

Split out of viewer.py; viewer_menus.py opens it. It names ViewerWindow in annotations
only, so it imports that under TYPE_CHECKING: a runtime import back would be
a cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
)

from descape import (
    player_fields,
)
from descape.mirror_tools import (
    ANGULAR_MODES,
    MODE_BY_ID,
    MODES,
    OVERLAPS_PLACE,
    OVERLAPS_REFUSE,
    OVERLAPS_SKIP,
    angular_mirror_axes,
    plan_mirror,
    plan_mirror_units,
)
from descape.viewer_common import (
    status_color,
)

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


class MirrorDialog(QDialog):
    """Map mirroring (Stage 1: terrain + elevation). The repo's first
    accept/reject dialog:
    SettingsDialog (settings_dialog.py) and DebugLogDialog (viewer_dialogs.py) are both Close-only and apply
    every change immediately on its own widget signal, so there is no OK/
    Apply/Cancel convention here to copy -- this establishes one, with a
    third "Preview" state in between.

    Preview applies the real edit as a normal undo record (via
    ViewerWindow.on_mirror(), same shape as on_fill()) and leaves the dialog
    open, so the user judges actual rendered pixels. Changing any option, or
    Cancel, undoes that preview first -- guarded by EditHistory.peek_undo()
    so it only auto-undoes if the preview's own record is still the top of
    the stack, never eating an edit made some other way in the meantime.
    Apply keeps whatever the last Preview already applied (or, if nothing
    was previewed under the current options, applies now) and closes.
    """

    def __init__(self, parent: ViewerWindow, last: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Mirror Map")
        self.resize(460, 460)
        self._window = parent
        # The restored slice, picked by the first _on_mode_changed() only.
        self._restore_slice: int | None = None
        # TASK-514: set when a read-only file forced Units off, so state() keeps the opened value.
        self._units_read_only = False
        # The DiffRecord Preview last pushed, or None if no preview is
        # currently applied under the CURRENTLY selected options (an option
        # change or Cancel undoes it and resets this to None -- see
        # _undo_own_preview()).
        self._preview_record = None

        layout = QVBoxLayout(self)

        note = QLabel(
            "The map renders as a diamond on screen (isometric projection) -- "
            "mode names below describe the screen tips they pair up, not raw "
            "array directions. The shaded area on the map is the source slice. "
            "The 3-way and 6-way modes split the map into equal wedges by tile distance, "
            "so the wedges look unequal on screen."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QGridLayout()
        row = 0
        form.addWidget(QLabel("Symmetry:"), row, 0)
        self.mode_combo = QComboBox()
        two_way = [m for m in MODES if len(m.group) == 2]
        four_way = [m for m in MODES if len(m.group) == 4]
        eight_way = [m for m in MODES if len(m.group) == 8]
        for mode in two_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        self.mode_combo.insertSeparator(self.mode_combo.count())
        for mode in four_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        self.mode_combo.insertSeparator(self.mode_combo.count())
        for mode in eight_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        # Angular modes have an empty `group`, so the length buckets above
        # never see them: they get their own labelled block.
        self.mode_combo.insertSeparator(self.mode_combo.count())
        self.mode_combo.addItem("Approximate (tile-space wedges):")
        header = self.mode_combo.model().item(self.mode_combo.count() - 1)
        header.setFlags(header.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
        for mode in ANGULAR_MODES:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        form.addWidget(self.mode_combo, row, 1)
        row += 1

        form.addWidget(QLabel("Source slice:"), row, 0)
        self.slice_combo = QComboBox()
        form.addWidget(self.slice_combo, row, 1)
        row += 1
        layout.addLayout(form)

        self.terrain_checkbox = QCheckBox("Terrain (type and blend layer)")
        self.terrain_checkbox.setChecked(True)
        layout.addWidget(self.terrain_checkbox)
        self.elevation_checkbox = QCheckBox("Elevation")
        self.elevation_checkbox.setChecked(True)
        self.elevation_checkbox.setToolTip(
            "Copy elevation too. Off by default for the 3-way and 6-way modes: resampling a "
            "slope skips rows, so almost every hilly map ends up with steps the game rejects"
        )
        layout.addWidget(self.elevation_checkbox)
        # Which kind of mode the Elevation default was last set for, so a
        # switch between lattice and angular modes re-applies it once.
        self._elevation_default_kind = "lattice"
        self.units_checkbox = QCheckBox("Units")
        self.units_checkbox.setChecked(False)
        self.units_checkbox.setToolTip(
            "Rewrite every other slice's units from the source slice. Gates are swapped to "
            "the orientation sibling the mirror turns them into"
        )
        layout.addWidget(self.units_checkbox)

        # TASK-250: indented under Units, by the checkbox indicator's own width.
        overlap_row = QHBoxLayout()
        style = self.style()
        overlap_row.addSpacing(
            style.pixelMetric(QStyle.PM_IndicatorWidth) + style.pixelMetric(QStyle.PM_CheckBoxLabelSpacing)
        )
        self.overlaps_label = QLabel("Overlapping copies:")
        self.overlaps_combo = QComboBox()
        for text, policy in (("Refuse", OVERLAPS_REFUSE), ("Skip them", OVERLAPS_SKIP), ("Place anyway", OVERLAPS_PLACE)):
            self.overlaps_combo.addItem(text, policy)
        self.overlaps_combo.setToolTip(
            "Where a unit's copy would land on the original across the axis: refuse the whole mirror, "
            "leave those copies out, or place them overlapping the original. Elevation seams and "
            "garrisoned units outside the source slice still refuse"
        )
        self.overlaps_label.setBuddy(self.overlaps_combo)
        for widget in (self.overlaps_label, self.overlaps_combo):
            widget.setEnabled(False)
            overlap_row.addWidget(widget)
        overlap_row.addStretch(1)
        layout.addLayout(overlap_row)

        # Ownership rotation is opt-in and off by default, matching the
        # in-game Map Copy tool's own Change Player option: there is no
        # per-mode rule that is right for every scenario, since the slice
        # count often does not divide the player count.
        owner_row = QHBoxLayout()
        self.ownership_checkbox = QCheckBox("Rotate ownership by")
        self.ownership_checkbox.setChecked(False)
        self.ownership_spin = QSpinBox()
        self.ownership_spin.setRange(1, 8)
        self.ownership_spin.setValue(1)
        self.ownership_spin.setEnabled(False)
        self.ownership_spin.setToolTip(
            "Players per slice, along the scenario's own defined-player list. GAIA is "
            "never rotated. With fewer defined players than slices, ownership wraps around"
        )
        owner_row.addWidget(self.ownership_checkbox)
        owner_row.addWidget(self.ownership_spin)
        owner_row.addStretch(1)
        layout.addLayout(owner_row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        btn_row.addStretch(1)
        btn_row.addWidget(self.preview_button)
        btn_row.addWidget(self.apply_button)
        btn_row.addWidget(self.cancel_button)
        layout.addLayout(btn_row)

        if last:
            self._restore(last)  # before the connections, so the open still plans once
        self._opened_units = self.units_checkbox.isChecked()

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.slice_combo.currentIndexChanged.connect(self._on_option_changed)
        self.terrain_checkbox.toggled.connect(self._on_option_changed)
        self.elevation_checkbox.toggled.connect(self._on_option_changed)
        self.units_checkbox.toggled.connect(self._on_option_changed)
        self.units_checkbox.toggled.connect(self.overlaps_label.setEnabled)
        self.units_checkbox.toggled.connect(self.overlaps_combo.setEnabled)
        self.overlaps_combo.currentIndexChanged.connect(self._on_option_changed)
        self.ownership_checkbox.toggled.connect(self.ownership_spin.setEnabled)
        self.ownership_checkbox.toggled.connect(self._on_option_changed)
        self.ownership_spin.valueChanged.connect(self._on_option_changed)
        self.preview_button.clicked.connect(self._on_preview)
        self.apply_button.clicked.connect(self._on_apply)
        self.cancel_button.clicked.connect(self.reject)
        self.finished.connect(self._on_finished)

        self._on_mode_changed()  # populates slice_combo and the initial overlay/summary

    def state(self) -> dict:
        """Session-sticky values, restored by the next dialog. Nothing here
        is persisted to settings, so there is no MEMOIZED_GLOBALS key."""
        return {
            "mode_id": self.mode_combo.currentData(),
            "slice": self.slice_combo.currentData(),
            "terrain": self.terrain_checkbox.isChecked(),
            "elevation": self.elevation_checkbox.isChecked(),
            "units": self._opened_units if self._units_read_only else self.units_checkbox.isChecked(),
            "overlaps": self.overlaps_combo.currentData(),
            "rotate_ownership": self.ownership_checkbox.isChecked(),
            "ownership_steps": self.ownership_spin.value(),
        }

    def _restore(self, last: dict) -> None:
        index = self.mode_combo.findData(last.get("mode_id")) if last.get("mode_id") is not None else -1
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        kind = self._current_mode().kind
        # Else the first _on_mode_changed() sees a kind switch and overwrites the restored Elevation.
        self._elevation_default_kind = kind
        self.terrain_checkbox.setChecked(bool(last.get("terrain", True)))
        self.elevation_checkbox.setChecked(bool(last.get("elevation", kind == "lattice")))
        self.units_checkbox.setChecked(bool(last.get("units", False)))
        index = self.overlaps_combo.findData(last.get("overlaps", OVERLAPS_REFUSE))
        if index >= 0:
            self.overlaps_combo.setCurrentIndex(index)
        self.ownership_checkbox.setChecked(bool(last.get("rotate_ownership", False)))
        self.ownership_spin.setValue(last.get("ownership_steps", 1))
        # The toggled signals are not connected yet, so the enabled states are settled here.
        for widget in (self.overlaps_label, self.overlaps_combo):
            widget.setEnabled(self.units_checkbox.isChecked())
        self.ownership_spin.setEnabled(self.ownership_checkbox.isChecked())
        self._restore_slice = last.get("slice")

    def _current_mode(self):
        return MODE_BY_ID[self.mode_combo.currentData()]

    def _on_mode_changed(self) -> None:
        self._undo_own_preview()
        mode = self._current_mode()
        if mode.kind != self._elevation_default_kind:
            # Measured over examples/: 6 of 285 hilly file x angular mode x
            # wedge cases had no seam violation, so angular defaults it off.
            self._elevation_default_kind = mode.kind
            self.elevation_checkbox.blockSignals(True)
            self.elevation_checkbox.setChecked(mode.kind == "lattice")
            self.elevation_checkbox.blockSignals(False)
        self.slice_combo.blockSignals(True)
        self.slice_combo.clear()
        for index, label in enumerate(mode.slice_labels):
            self.slice_combo.addItem(label, index)
        if self._restore_slice is not None:
            self.slice_combo.setCurrentIndex(max(0, min(self._restore_slice, self.slice_combo.count() - 1)))
            self._restore_slice = None
        self.slice_combo.blockSignals(False)
        self._on_option_changed()

    def _on_option_changed(self) -> None:
        self._undo_own_preview()
        self._refresh_overlay_and_summary()

    def _compute_plan(self):
        mode = self._current_mode()
        mm = self._window.scenario.map_manager
        slice_index = self.slice_combo.currentData()
        if slice_index is None:
            slice_index = 0
        return plan_mirror(
            mm,
            mode.mode_id,
            slice_index,
            self.terrain_checkbox.isChecked(),
            self.elevation_checkbox.isChecked(),
        )

    def _compute_unit_plan(self, plan):
        """The units half, or None when the Units box is off or this file's
        units are read-only. Shares plan_mirror()'s own source_indices rather
        than re-deriving which tiles are the source."""
        if not self.units_checkbox.isChecked():
            return None
        model = self._window._ensure_unit_edits()
        if model is None:
            # Read-only units: untick rather than just returning None, or
            # _ensure_unit_edits()' warning box reappears on every option
            # change for the rest of the dialog's life.
            self._units_read_only = True
            self.units_checkbox.setChecked(False)
            return None
        mode = self._current_mode()
        slice_index = self.slice_combo.currentData() or 0
        loaded = self._window.scenario
        return plan_mirror_units(
            loaded.map_manager,
            mode.mode_id,
            slice_index,
            loaded.unit_manager.units,
            plan.source_indices,
            player_ids=player_fields.defined_player_ids(loaded, self._window._pending_options()),
            ownership_steps=self.ownership_spin.value() if self.ownership_checkbox.isChecked() else 0,
            referencing=model.referencing,
            unreachable=plan.unreachable,
            overlaps=self.overlaps_combo.currentData(),
        )

    def _refresh_overlay_and_summary(self) -> None:
        if self._window.scenario is None:
            return
        plan = self._compute_plan()
        unit_plan = self._compute_unit_plan(plan)
        mm = self._window.scenario.map_manager
        n = mm.map_width
        tiles = [divmod(idx, n)[::-1] for idx in plan.source_indices]
        axes = self._axis_lines(self._current_mode(), n)
        self._window.map_view.show_mirror_overlay(tiles, axes)

        blockers = []
        if plan.elevation_violations:
            blockers.append(
                f"{len(plan.elevation_violations)} elevation seam violation(s) would exceed "
                f"the +/-1 limit"
            )
        if unit_plan is not None:
            if unit_plan.overlap_blocked:
                blockers.append(
                    f"{len(unit_plan.overlap_blockers)} unit(s) straddle the symmetry axis or a "
                    f"slice boundary (copies would land on them)"
                )
            if unit_plan.garrisoned_blockers:
                blockers.append(
                    f"{len(unit_plan.garrisoned_blockers)} unit(s) outside the source slice "
                    f"hold a garrison and cannot be replaced"
                )
        # Angular modes only: what the approximation leaves alone, reported
        # whether or not the plan is blocked.
        leftovers = []
        if plan.unreachable:
            leftovers.append(
                f"{len(plan.unreachable)} corner tile(s) have no source inside the map and stay as they are"
            )
        if unit_plan is not None and unit_plan.off_map:
            leftovers.append(
                f"{len(unit_plan.off_map)} unit image(s) would land off the map or in an untouched "
                f"corner and are skipped"
            )
        if blockers:
            self.summary_label.setText(
                f"{len(plan.changes)} tile(s) would change -- BLOCKED: "
                + "; ".join(blockers)
                + ". Pick a different mode, or move the offending objects."
                + "".join(f" {part[0].upper()}{part[1:]}." for part in leftovers)
            )
            # On the window, not a tab page. Modal, so a set-time colour is enough.
            self.summary_label.setStyleSheet(f"color: {status_color(False, on_tab_page=False)};")
        else:
            parts = [f"{len(plan.changes)} tile(s) will change"]
            if unit_plan is not None:
                parts.append(
                    f"{len(unit_plan.removals)} unit(s) replaced by {len(unit_plan.images)} image(s)"
                )
                if unit_plan.gates:
                    parts.append(f"{len(unit_plan.gates)} gate(s) reoriented")
                if unit_plan.unsquare_spans:
                    parts.append(
                        f"{len(unit_plan.unsquare_spans)} non-square footprint(s) cannot be "
                        f"reflected onto the other axis"
                    )
                if unit_plan.overlapping:
                    fate = "will be skipped" if unit_plan.overlaps == OVERLAPS_SKIP else "will be placed overlapping"
                    parts.append(f"{len(unit_plan.overlapping)} unit cop(ies) overlap the source and {fate}")
            parts.extend(leftovers)
            self.summary_label.setText("; ".join(parts))
            self.summary_label.setStyleSheet("")

    def _axis_lines(self, mode, n: int):
        """Straight symmetry-axis lines, Flat style only -- Stepped/Sloped's
        projected geometry makes a literal straight line non-trivial to
        place correctly, and the shaded source slice (tile-exact in every
        style via _tile_polygon) already conveys the boundary on its own.
        One line per reflection generator actually in the mode's group;
        pure-rotation modes (5, 6, 10, 11) have no reflection axis and draw
        none; mode 12 draws its three mirror axes."""
        if self._window.map_view._terrain_style != "flat":
            return []
        tp = self._window.map_view._tile_pixels
        size = n * tp
        lines = []
        if mode.angular is not None:
            # Scene space is tile space scaled, so a tile-space axis is a
            # straight scene line through the centre, clipped to the square.
            half = size / 2
            for dx, dy in angular_mirror_axes(mode.angular):
                t = half / max(abs(dx), abs(dy))
                lines.append((QPointF(half + t * dx, half + t * dy), QPointF(half - t * dx, half - t * dy)))
            return lines
        if "mx" in mode.group:  # u=0 -- the vertical centre line
            lines.append((QPointF(size / 2, 0), QPointF(size / 2, size)))
        if "my" in mode.group:  # v=0 -- the horizontal centre line
            lines.append((QPointF(0, size / 2), QPointF(size, size / 2)))
        if "d" in mode.group:  # v-u=0 -- the main diagonal, (0,0)-(n-1,n-1)
            lines.append((QPointF(0, 0), QPointF(size, size)))
        if "a" in mode.group:  # u+v=0 -- the anti-diagonal, (n-1,0)-(0,n-1)
            lines.append((QPointF(size, 0), QPointF(0, size)))
        return lines

    def _undo_own_preview(self) -> None:
        if self._preview_record is not None and self._window.edit_history.peek_undo() is self._preview_record:
            self._window.undo()
        self._preview_record = None

    def _on_preview(self) -> None:
        self._undo_own_preview()
        plan = self._compute_plan()
        dirty = self._window.on_mirror(plan, self._compute_unit_plan(plan))
        if dirty:
            self._preview_record = self._window.edit_history.peek_undo()
        self._refresh_overlay_and_summary()

    def _on_apply(self) -> None:
        if self._preview_record is not None and self._window.edit_history.peek_undo() is self._preview_record:
            self.accept()
            return
        plan = self._compute_plan()
        dirty = self._window.on_mirror(plan, self._compute_unit_plan(plan))
        if dirty is None:
            return  # refused (elevation violations) -- status already logged, stay open
        self.accept()

    def _on_finished(self, result: int) -> None:
        if result == QDialog.Rejected:
            self._undo_own_preview()
        self._window.map_view.clear_mirror_overlay()
