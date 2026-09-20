"""UnitsPanel: the Units-mode left page.

A vertical splitter: the object catalog and Owner combo -- the placement
vocabulary, used on every click -- on top, and a small selected-unit(s)
inspector at the bottom, sized like TriggerPanel's own property form. This
replaces both the modal CatalogBrowseDialog the param toolbar used to open
and the toolbar's own Object/Owner pair, which is why the catalog and
inspector live in one class: they share the splitter, the MIN_USEFUL_WIDTH
this mode needs, and (via the sticky pending object const) the moment a
double-click or Enter should ask the window to place something.

Dumb and callback-driven like TriggerPanel/PlayersPanel: it never touches
EditHistory or a UnitEditModel itself, reporting "the user set field X of
the (single) selected unit to raw value V" or "the user asked to place
object N" through the callbacks it is constructed with. Selection itself --
which unit(s) show here at all -- and the write path stay entirely on
ViewerWindow: see its own _refresh_selection_view() docstring for why that
reconciliation point has to stay one piece rather than splitting across a
panel boundary.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from descape import object_catalog, unit_fields, unit_rotation
from descape.constant_picker import HIDDEN_LABEL, catalog_preview, items_for
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.unit_stats_table import unit_stats
from descape.value_picker import ValuePickerView

# unit_fields.UnitFieldSpec.conditional -> the rule it names. The specs stay
# Qt-free and data-only by naming a rule as a string; this is the one place
# that resolves it, so the rule itself stays independently testable.
_FIELD_CONDITIONALS = {"rotation_is_angle": unit_rotation.rotation_is_angle}

# The full caveat, unconditionally available via the Rotation caption's
# tooltip (see _build_inspector_pane) regardless of which const is selected.
# unit_rotation_note below shows the same text, but only when the selected
# const's rotation genuinely isn't a real angle -- see
# _apply_conditional_fields.
_ROTATION_TOOLTIP = (
    "Rotation is shown raw, in radians. It is editable only for units whose "
    "rotation is a real facing -- for most GAIA objects, walls and gates it "
    "is a graphic-variant index, not an angle. Trees, plants and scenery can "
    "change variant with Edit > Cycle Variant."
)

# The in-game editor's own Units-tab selection panel set (docs/
# INGAME_EDITOR_REFERENCE.md's "Selecting: shows HP/Attack/Armour") plus
# Range, which the original stats request named explicitly. unit_stats()
# keys use genieutils' own British spelling; captions don't have to.
_STAT_FIELDS = (
    ("hp", "Hit points"),
    ("attack", "Attack"),
    ("melee_armour", "Melee armour"),
    ("pierce_armour", "Pierce armour"),
    ("range", "Range"),
)

# Shortened per the vertical-budget mitigations: one line at MIN_USEFUL_WIDTH
# rather than the full sentence, which is still one tooltip away.
_STATS_NOTE_SHORT = "Base values: no civ bonuses, tech upgrades or trigger effects applied."
_STATS_NOTE_FULL = (
    "Base values from the game's own unit table: no civ bonuses, tech "
    "upgrades or trigger effects applied."
)


class UnitsPanel(QWidget):
    # Measured at MIN_USEFUL_WIDTH: 64 px sprite preview + ~16 px scrollbar +
    # ~280 px tree viewport + margins. Between messages/map_options (300) and
    # diplomacy (430) -- see tools/gen_units_panel_eyeball.py's screenshot
    # pass for confirmation; adjust there; not by argument.
    MIN_USEFUL_WIDTH = 380

    def __init__(self, on_unit_field=None, on_place_requested=None):
        super().__init__()
        # No-op defaults so the panel stays constructible on its own, the
        # same contract MapOptionsPanel/TriggerPanel/PlayersPanel's
        # callbacks have.
        self._on_unit_field = on_unit_field or (lambda *args: None)
        self._on_place_requested = on_place_requested or (lambda *args: None)
        # True while show_unit()/clear() are populating widgets
        # programmatically -- suppresses _field_changed() the same way
        # PlayersPanel._changed()'s own _populating guard does, so a
        # populate never looks like a user edit and never records a phantom
        # undo step.
        self._populating = False
        # Sticky, mirroring CatalogLineEdit._value (never cleared by a
        # filter or selection change that lands on nothing): the object the
        # next Place click/Enter places. See _on_catalog_current_changed.
        self._pending_object_const: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Vertical)
        # Both here and on the horizontal content_splitter: without this, a
        # user dragging this splitter to the bottom of its travel collapses
        # the inspector to zero height while every isVisibleTo() assertion
        # in tests/test_unit_selection_viewer.py still passes -- collapsed-
        # to-zero is not the same thing as hidden as far as Qt is concerned.
        self.splitter.setChildrenCollapsible(False)
        layout.addWidget(self.splitter, stretch=1)

        self.splitter.addWidget(self._build_catalog_pane())
        self.splitter.addWidget(self._build_inspector_pane())
        # (1, 0), not TriggerPanel's own (1, 1): the catalog absorbs
        # vertical resize, the inspector keeps whatever height it was given.
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        # Qt normalizes setSizes() to a ratio of whatever height the
        # splitter ends up with, even called before the first show/resize --
        # this is the initial default, not a floor; dragging overrides it.
        self.splitter.setSizes([2, 1])

        self._show_inspector_fields(False)

    # -- catalog pane ------------------------------------------------------

    def _build_catalog_pane(self) -> QWidget:
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        self.catalog_view = ValuePickerView(
            items_for(object_catalog.objects()),
            default_group="Units",
            show_values=False,
            hidden_label=HIDDEN_LABEL,
            preview=catalog_preview,
        )
        # ClickFocus, not the tree's Qt default: this tree sits in an
        # always-visible sidebar, and MapView's own arrow-key unit nudge
        # (map_view.py's keyPressEvent) only fires while MapView itself has
        # focus. A focus-follows-click default would silently break nudging
        # the moment a user clicks into the catalog and forgets to click
        # back onto the map -- see _on_catalog_activated, which is the
        # common flow's own way back.
        self.catalog_view.tree.setFocusPolicy(Qt.ClickFocus)
        self.catalog_view.current_changed.connect(self._on_catalog_current_changed)
        self.catalog_view.activated.connect(self._on_catalog_activated)
        pane_layout.addWidget(self.catalog_view, stretch=1)

        self.placing_label = QLabel("Placing: (none)")
        pane_layout.addWidget(self.placing_label)

        owner_row = QHBoxLayout()
        owner_row.setContentsMargins(0, 0, 0, 0)
        owner_row.addWidget(QLabel("Owner:"))
        self.owner_combo = QComboBox()
        self.owner_combo.addItem("GAIA", GAIA_PLAYER_ID)
        for player_id in range(1, MAX_PLAYER_ID + 1):
            self.owner_combo.addItem(f"Player {player_id}", player_id)
        self.owner_combo.setCurrentIndex(1)  # Player 1, preserved default
        owner_row.addWidget(self.owner_combo, stretch=1)
        pane_layout.addLayout(owner_row)
        return pane

    def _on_catalog_current_changed(self, value) -> None:
        """Updates the sticky pending value, but only when the new current
        row is a real, visible object -- ValuePickerView.current_value()
        (what feeds this signal) already folds in the group-heading and
        hidden-row guards, returning None for either, so a filter that hides
        the previously-picked row leaves the pending value untouched rather
        than silently blanking "choose an object first"."""
        if value is None:
            return
        self._set_pending_object(value)

    def _on_catalog_activated(self, value: int) -> None:
        self._set_pending_object(value)
        self._on_place_requested(value)

    def _set_pending_object(self, value: int) -> None:
        self._pending_object_const = value
        name = object_catalog.name_for(object_catalog.objects(), value) or str(value)
        self.placing_label.setText(f"Placing: {name}")
        # A one-line label doesn't wrap or elide, so a long name (the
        # longest object names run 36 characters) silently clips at
        # MIN_USEFUL_WIDTH -- caught by the screenshot pass, not by any
        # widget assertion. The tooltip is the cheap fix that keeps the
        # label one line.
        self.placing_label.setToolTip(name)

    def selected_object_const(self) -> int | None:
        return self._pending_object_const

    def select_object(self, const: int) -> None:
        self.catalog_view.select(const)

    def owner_id(self) -> int:
        return self.owner_combo.currentData()

    def select_owner(self, player_id: int) -> bool:
        index = self.owner_combo.findData(player_id)
        if index < 0:
            return False
        self.owner_combo.setCurrentIndex(index)
        return True

    # -- inspector pane ------------------------------------------------------

    def _build_inspector_pane(self) -> QWidget:
        """Wrapped in a QScrollArea(widgetResizable=True), exactly like
        TriggerPanel._build_property_page(): nothing outside it may have a
        large intrinsic minimum, because QStackedWidget.minimumSizeHint() is
        the max over every page and would make this page a floor for the
        whole window otherwise."""
        self.inspector_area = QScrollArea()
        self.inspector_area.setWidgetResizable(True)
        self.inspector_host = QWidget()
        self.inspector_layout = QVBoxLayout(self.inspector_host)

        self.unit_inspector_empty = QLabel("No unit selected")
        self.unit_inspector_empty.setWordWrap(True)
        self.inspector_layout.addWidget(self.unit_inspector_empty)

        self.unit_inspector_grid = QWidget()
        grid = QGridLayout(self.unit_inspector_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        self.unit_field_labels: dict[str, QLabel] = {}
        self.unit_field_editors: dict[str, QWidget] = {}
        for row, spec in enumerate(unit_fields.FIELDS):
            caption = QLabel(f"{spec.label}:")
            grid.addWidget(caption, row, 0)
            if spec.field_id == "rotation":
                # The "shown raw, in radians" half of the old always-visible
                # note, made always-accessible for zero vertical cost
                # instead -- see unit_rotation_note below for the other half.
                caption.setToolTip(_ROTATION_TOOLTIP)
                self.unit_rotation_label = caption
            # A conditional field gets BOTH widgets, in the same cell: the
            # editor for consts its rule admits, and the plain read-only label
            # every other const keeps. _apply_conditional_fields() shows one
            # and hides the other per selection, and a hidden widget takes no
            # layout space, so the row never doubles in height.
            if not spec.editable or spec.conditional:
                value = QLabel("")
                value.setWordWrap(True)
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(value, row, 1)
                self.unit_field_labels[spec.field_id] = value
            if not spec.editable:
                continue
            if spec.kind == unit_fields.PLAYER:
                combo = QComboBox()
                combo.addItem("GAIA", GAIA_PLAYER_ID)
                for player_id in range(1, MAX_PLAYER_ID + 1):
                    combo.addItem(f"Player {player_id}", player_id)
                combo.currentIndexChanged.connect(
                    lambda _, s=spec, w=combo: self._field_changed(s, w.currentData())
                )
                grid.addWidget(combo, row, 1)
                self.unit_field_editors[spec.field_id] = combo
            else:  # FLOAT
                spin = QDoubleSpinBox()
                spin.setDecimals(spec.decimals)
                spin.setRange(spec.minimum, spec.maximum)
                if spec.field_id == "rotation":
                    spin.setSuffix(" rad")
                # setKeyboardTracking(False) is load-bearing the same way
                # viewer_common._make_spinbox's own comment explains: without
                # it, typing a multi-digit value over an old one fires
                # valueChanged once per digit, recording one undo step per
                # keystroke instead of one per commit (trap 3).
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(
                    lambda value, s=spec: self._field_changed(s, value)
                )
                grid.addWidget(spin, row, 1)
                self.unit_field_editors[spec.field_id] = spin
        self.inspector_layout.addWidget(self.unit_inspector_grid)

        # Conditional (only for a const whose rotation isn't a real angle) --
        # see _apply_conditional_fields, which owns this widget's visibility.
        # A top-level AGENTS.md hard rule, and this panel is the first place
        # it becomes user-visible: for ~65% of GAIA objects `rotation` is a
        # tree/doodad graphic-variant index (values like 7..53), not an angle.
        # Same text as the Rotation caption's tooltip above -- both name the
        # same fact, one always reachable, the other surfaced only when it
        # actually applies to the selected const.
        self.unit_rotation_note = QLabel(_ROTATION_TOOLTIP)
        self.unit_rotation_note.setWordWrap(True)
        self.inspector_layout.addWidget(self.unit_rotation_note)

        # A separate grid, not appended to unit_field_labels/unit_fields.FIELDS:
        # these are derived reference data (empires2_x2_p1.dat, committed as
        # unit_stats.json) and will never be editable, unlike every row above.
        # Whole block hidden when unit_stats() has nothing for the const (no
        # combat block, e.g. a pure decorative) -- see _apply_stats.
        self.unit_stats_header = QLabel("<b>Base stats</b>")
        self.inspector_layout.addWidget(self.unit_stats_header)

        self.unit_stats_grid = QWidget()
        stats_grid = QGridLayout(self.unit_stats_grid)
        stats_grid.setContentsMargins(0, 0, 0, 0)
        # Both labels per row, not just the value: ~1,169 of 2,642 consts
        # have no type_50 combat block at all (a tree, say), and hiding only
        # the value would leave four rows reading "Attack:", "Range:" with
        # nothing beside them instead of disappearing outright.
        self.unit_stat_rows: dict[str, tuple[QLabel, QLabel]] = {}
        for row, (field_id, caption) in enumerate(_STAT_FIELDS):
            cap_label = QLabel(f"{caption}:")
            value_label = QLabel("")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            stats_grid.addWidget(cap_label, row, 0)
            stats_grid.addWidget(value_label, row, 1)
            self.unit_stat_rows[field_id] = (cap_label, value_label)
        self.inspector_layout.addWidget(self.unit_stats_grid)

        self.unit_stats_note = QLabel(_STATS_NOTE_SHORT)
        self.unit_stats_note.setWordWrap(True)
        self.unit_stats_note.setToolTip(_STATS_NOTE_FULL)
        self.inspector_layout.addWidget(self.unit_stats_note)

        self.inspector_layout.addStretch(1)
        self.inspector_area.setWidget(self.inspector_host)
        return self.inspector_area

    def _show_inspector_fields(self, visible: bool) -> None:
        self.unit_inspector_grid.setVisible(visible)
        if not visible:
            # Shown again only if _apply_conditional_fields() re-admits it
            # for whatever gets selected next -- no unit selected means no
            # per-const caveat to show.
            self.unit_rotation_note.setVisible(False)
        self.unit_inspector_empty.setVisible(not visible)

    def show_unit(self, entry) -> None:
        """Populates every field from `entry` (a unit_pick index entry), or
        blanks and hides the grid for None -- the "nothing selected" state.
        Never touches unit_inspector_empty's text; pair a None call with
        show_selection_count() when the caller has a count to report."""
        if entry is None:
            self._show_inspector_fields(False)
            for label in self.unit_field_labels.values():
                label.setText("")
            self._apply_stats(None)
            self._fit_inspector_height()
            return
        unit = entry.unit
        texts = {
            "name": object_catalog.display_name(unit.unit_const),
            "unit_const": str(unit.unit_const),
            "rotation": f"{unit.rotation:g}",
            "reference_id": str(unit.reference_id),
            "garrisoned_in_id": str(getattr(unit, "garrisoned_in_id", -1)),
        }
        for field, text in texts.items():
            self.unit_field_labels[field].setText(text)
        # Guards the editors below against _field_changed recording a
        # phantom undo step from this programmatic populate -- the same
        # _populating guard players_panel._changed() uses.
        self._populating = True
        try:
            self.unit_field_editors["player"].setCurrentIndex(
                max(self.unit_field_editors["player"].findData(entry.player_id), 0)
            )
            self.unit_field_editors["x"].setValue(unit.x)
            self.unit_field_editors["y"].setValue(unit.y)
            self.unit_field_editors["z"].setValue(getattr(unit, "z", 0.0))
            self._apply_conditional_fields(unit)
        finally:
            self._populating = False
        self._apply_stats(unit.unit_const)
        self._show_inspector_fields(True)
        self._fit_inspector_height()

    def _apply_conditional_fields(self, unit) -> None:
        """Swaps each conditional field between its editor and its read-only
        label for THIS unit's const -- today only Rotation, whose rule is
        unit_rotation.rotation_is_angle. Also owns unit_rotation_note's
        visibility (shown only when the const's rotation is NOT a real
        angle): the caveat appears exactly on the consts where the value is
        a lie-in-waiting, rather than always.

        Only ever called with the populating guard already held: it sets an
        editor's value, and the resulting valueChanged must not record a
        phantom undo step.

        The value is normalized through rotate_step(..., 0) on the way in
        rather than handed to setValue raw. An angle const can still carry a
        stored value outside [0, 2*pi) (7.0 appears 574 times in the corpus),
        and the spinbox would silently CLAMP that to its own maximum -- i.e.
        display a number the file does not contain.
        """
        for spec in unit_fields.FIELDS:
            if not spec.conditional:
                continue
            allowed = _FIELD_CONDITIONALS[spec.conditional](unit.unit_const)
            editor = self.unit_field_editors[spec.field_id]
            editor.setVisible(allowed)
            self.unit_field_labels[spec.field_id].setVisible(not allowed)
            if spec.field_id == "rotation":
                self.unit_rotation_note.setVisible(not allowed)
            if allowed and spec.field_id == "rotation":
                editor.setValue(
                    unit_rotation.rotate_step(
                        unit.rotation, unit_rotation.angle_count_for(unit.unit_const), 0
                    )
                )

    def _apply_stats(self, unit_const: int | None) -> None:
        """Shows/hides each Base stats row against key presence in
        unit_stats(unit_const), and hides the whole block (header, grid,
        note) when the dict is empty -- an unrecognized const (a modded
        scenario), or a real one with no HP and no combat block at all
        (roughly a fifth of the table: cliffs, rocks, other decoratives).
        `unit_const=None` (no selection) is treated the same as "no stats"."""
        stats = unit_stats(unit_const) if unit_const is not None else {}
        has_stats = bool(stats)
        self.unit_stats_header.setVisible(has_stats)
        self.unit_stats_grid.setVisible(has_stats)
        self.unit_stats_note.setVisible(has_stats)
        for field_id, (cap_label, value_label) in self.unit_stat_rows.items():
            present = field_id in stats
            cap_label.setVisible(present)
            value_label.setVisible(present)
            # :g so 4.0 renders as "4" -- a melee unit's range is 0, not "0.0".
            value_label.setText(f"{stats[field_id]:g}" if present else "")

    def clear(self) -> None:
        """Blanks the inspector with no count to report -- the mode-switch
        paths (on_mode_changed, _force_mode), which reset the selection
        outright rather than reconciling it."""
        self.show_unit(None)

    def show_selection_count(self, count: int) -> None:
        """0 or 2+ selected units: hides the field grids (never call this
        for exactly 1 -- show_unit() shows its real fields instead) and
        shows a "No unit selected" / "N units selected" readout."""
        self.show_unit(None)
        self.unit_inspector_empty.setText(f"{count} units selected" if count else "No unit selected")

    def _fit_inspector_height(self) -> None:
        """Tell the scroll area how tall the wrapped content actually is --
        copied from TriggerPanel._fit_property_height() verbatim, including
        its activate()-first trap (measuring before Qt has laid out rows
        added moments ago answers for the PREVIOUS content instead). Three
        word-wrapping labels inside a widgetResizable scroll area reproduce
        the exact defect that method documents: rows draw on top of each
        other while every measurement still reads correct."""
        self.inspector_layout.activate()
        height = self.inspector_layout.minimumSize().height()
        width = self.inspector_area.viewport().width()
        if width > 0 and self.inspector_layout.hasHeightForWidth():
            height = max(height, self.inspector_layout.heightForWidth(width))
        self.inspector_host.setMinimumHeight(height)

    def resizeEvent(self, event) -> None:
        """Re-fit the inspector height, whose wrap-driven minimum is a
        function of the pane width -- see TriggerPanel.resizeEvent()."""
        super().resizeEvent(event)
        self._fit_inspector_height()

    # -- reporting an edit ---------------------------------------------------

    def _field_changed(self, spec: unit_fields.UnitFieldSpec, value) -> None:
        if self._populating:
            return
        self._on_unit_field(spec, value)
