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
the selected unit(s) to raw value V" or "the user asked to place
object N" through the callbacks it is constructed with. Selection itself --
which unit(s) show here at all -- and the write path stay entirely on
ViewerWindow: see its own _refresh_selection_view() docstring for why that
reconciliation point has to stay one piece rather than splitting across a
panel boundary.
"""

from __future__ import annotations

import random

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QFont, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import (
    catalog_thumbs,
    facing_dial,
    gate_orientation,
    object_catalog,
    place_random,
    player_labels,
    settings,
    unit_fields,
    unit_rotation,
    unit_variant,
)
from descape.catalog_browser import CatalogBrowserView
from descape.terrain_palette import PLAYER_COLORS
from descape.unit_card import _MIXED_TEXT, UnitCard
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.unit_model import capture_flag_of
from descape.viewer_common import (
    FontScaledWidth,
    _add_player_item,
    _fit_combo_width,
    _relabel_player_rows,
)

# unit_fields.UnitFieldSpec.conditional -> the rule it names. The specs stay
# Qt-free and data-only by naming a rule as a string; this is the one place
# that resolves it, so the rule itself stays independently testable.
_FIELD_CONDITIONALS = {"rotation_is_editable": unit_variant.is_rotation_editable}

# Always on the Rotation caption's tooltip. GH #147: a lone read-only unit gets a one-line per-kind tooltip on its value
# instead of this inline. GH #198: a group's note leads the same tooltip (see _apply_conditional_fields).
_ROTATION_TOOLTIP = (
    "Rotation is shown as a facing number, 0 to one less than the unit's "
    "direction count, stored in radians (hover the field). Trees, plants and "
    "scenery show their graphic variant number instead, and gates their "
    "orientation, 0 to 3 in the order Rotate steps. Walls and cliffs store a "
    "graphic-variant index the game derives itself, and single-frame objects "
    "have nothing to change: both are shown raw and not editable. For facings "
    "and gates, the dial beside the number points the way the unit faces on "
    "the map: click or drag it to turn, or scroll over it to step."
)

# GH #186: how a CHOICE row reads its raw value. getattr and capture_flag_of
# because a test fake has neither field and a pre-1.59 unit has no capture flag.
_CHOICE_READERS = {
    "status": lambda unit: getattr(unit, "status", 2),
    "capture_flag": capture_flag_of,
}
# A stored value the row has no label for: shown, selected, never rewritten.
_OTHER_CHOICE = "Other ({value})"
# UnitFieldSpec.requires_codec_field -> why the row is disabled in a file without it.
_CODEC_FIELD_TOOLTIPS = {
    "capture_flag": "Capture Flag needs a scenario saved by the September 2026 game update (version 1.59)",
}

# unit_fields.FIELDS rows the header shows instead of the field grid (GH #148).
_HEADER_FIELDS = ("name", "unit_const")

# Four rows plus the header, measured at the test font: the corpus never puts
# more than 6 units in one host, and a taller list would push Add/Delete below
# the fold of an inspector pane that already scrolls.
_GARRISON_TREE_MAX_PX = 120

_GARRISON_OVER_CAPACITY = (
    "This file puts {count} units in a host the game gives {capacity} places. "
    "Shown as it is; DEscape never changes it."
)
_GARRISON_WRONG_TYPE = "{count} of these cannot garrison here in game. Shown as it is; DEscape never changes it."
# GH #204: on each row a pending Cut marked, which is also drawn italic and dimmed.
_GARRISON_CUT_TIP = "Cut: Paste inside another host moves it there"

# Group-mode notes on the Rotation caption's tooltip (GH #198): {skipped} of {total} selected.
# GH #71: leads with what a typed value does, since "won't rotate" read as "trees can't".
_GROUP_ROTATION_NOTE = (
    "Typing a Rotation turns the {turned} selected {units} with a facing. "
    "The other {skipped} of {total} (trees, walls, gates, buildings, scenery) keep theirs."
)
_GROUP_NO_ROTATION_NOTE = (
    "Typing a Rotation changes none of the {total} selected: walls, cliffs, "
    "gates and single-frame objects keep their stored value."
)
# Appended when Rotate would step a member the typed value skips.
_GROUP_ROTATE_TAIL = " Rotate{keys} still {does}."
_ROTATE_TAIL_TREES = "steps trees and scenery through their variants"
_ROTATE_TAIL_GATES = "turns gates through their four orientations"
# GH #61: gate_orientation's cycle order, as the game names it (graphic tokens _ne_/_e_/_se_/_n_).
_GATE_ORIENTATION_NAMES = ("SW to NE", "W to E", "NW to SE", "N to S")
_GROUP_VARIANT_NOTE = (
    "Typing a Rotation picks that variant for the {turned} selected {objects} with graphic variants. "
    "The other {skipped} of {total} (walls, cliffs, gates, single-frame objects) keep theirs."
)
# GH #154: on X, Y and Z while a group is shown.
_GROUP_STEP_TOOLTIP = "Arrows move every selected unit by one step. A typed value sets them all."


class _CoordSpinBox(QDoubleSpinBox):
    """X/Y/Z's editor. In a group (GH #154) every step, from the arrows, the
    keys or the wheel, is reported as a relative `stepped` delta instead of
    moving the value, so the selection keeps its formation."""

    stepped = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        self.relative_steps = False

    def stepBy(self, steps: int) -> None:
        if not self.relative_steps:
            super().stepBy(steps)
            return
        # Text typed but not committed lands first, as Qt's own stepBy() does.
        self.interpretText()
        self.stepped.emit(steps * self.singleStep())

    def stepEnabled(self):
        # The viewer owns the edge rule; both arrows stay live on the "(mixed)" sentinel.
        if self.relative_steps and not self.isReadOnly():
            return QAbstractSpinBox.StepUpEnabled | QAbstractSpinBox.StepDownEnabled
        return super().stepEnabled()


def _rotate_tail(units) -> str:
    """" Rotate (. and ,) still ..." for the members Rotate steps, or "" if none."""
    does = []
    if any(unit_variant.is_cyclable(unit.unit_const) for unit in units):
        does.append(_ROTATE_TAIL_TREES)
    if any(gate_orientation.is_gate(unit.unit_const) for unit in units):
        does.append(_ROTATE_TAIL_GATES)
    if not does:
        return ""
    # Read live: both keys are rebindable, and either may be cleared.
    bound = [key for key in (settings.get_keybind("unit_rotate_cw"), settings.get_keybind("unit_rotate_ccw")) if key]
    keys = f" ({' and '.join(bound)})" if bound else ""
    return _GROUP_ROTATE_TAIL.format(keys=keys, does=" and ".join(does))


def _gate_index(unit) -> int:
    """A gate's orientation index: its const's place in gate_orientation's cycle order."""
    return gate_orientation.orientation_siblings(unit.unit_const).index(unit.unit_const)


def _variant_of(unit) -> int:
    """A cyclable unit's current variant, via the one shared reader."""
    const = unit.unit_const
    return unit_variant.variant_of(
        unit.rotation, unit_rotation.angle_count_for(const), unit_variant.variant_count_for(const)
    )


def _readonly_rotation_tooltip(unit_const: int) -> str:
    """GH #147: why a lone unit's Rotation is read-only, in two sentences; the caption keeps the full text."""
    if unit_rotation.semantics_for(unit_const) == unit_rotation.INERT:
        return "Single-frame object: nothing to rotate. Shown raw, not editable."
    # A lone gate never reaches here: it takes gate mode.
    if unit_const in unit_variant.excluded_consts():
        return "Graphic variant index the game derives from neighbouring pieces. Shown raw, not editable."
    return "Only one graphic variant: nothing to change. Shown raw, not editable."


class UnitsPanel(QWidget):
    # Re-measured for GH #68 at the test font: the inspector holds it, the
    # selection card's 334 px without an install plus 34 px of frame,
    # scrollbar and margins (368); the catalog browser's class row needs 342.
    # Between messages/map_options (300) and diplomacy (430) -- see
    # tools/gen_units_panel_eyeball.py's screenshot pass for confirmation;
    # adjust there; not by argument. Scales with the app font (FontScaledWidth, GH #142).
    MIN_USEFUL_WIDTH = FontScaledWidth(380)

    def __init__(
        self,
        on_unit_field=None,
        on_place_requested=None,
        on_pending_changed=None,
        on_garrison_add=None,
        on_garrison_delete=None,
        on_garrison_navigate=None,
        map_size=None,
        on_garrison_pick=None,
        on_garrison_copy=None,
        on_garrison_paste=None,
        on_garrison_unload=None,
        on_garrison_owner=None,
        on_garrison_cut=None,
        codec_field_supported=None,
        rng=None,
        on_unit_field_step=None,
    ):
        super().__init__()
        # GH #164: rolls a random pick's real const; injectable so a test can seed it.
        self._rng = rng if rng is not None else random.Random()
        # The real const a random pending pick places next; None for a real pick.
        self._rolled_const: int | None = None
        # GH #68: one loader, so the list and both cards share one thumbnail memo.
        self.thumb_loader = catalog_thumbs.ThumbLoader(self)
        # (width, height) of the loaded map, or None: queried at populate time
        # so a load or resize needs no push (GH #118).
        self._map_size = map_size or (lambda: None)
        # GH #186: does the loaded file store this Unit field? Pulled at populate time too.
        self._codec_field_supported = codec_field_supported or (lambda _name: True)
        # No-op defaults so the panel stays constructible on its own, the
        # same contract MapOptionsPanel/TriggerPanel/PlayersPanel's
        # callbacks have.
        self._on_unit_field = on_unit_field or (lambda *args: None)
        # GH #154: a group's X/Y/Z step, as (spec, delta): move every member by delta.
        self._on_unit_field_step = on_unit_field_step or (lambda *args: None)
        self._on_place_requested = on_place_requested or (lambda *args: None)
        # GH #128: the picked object or the owner changed, so Place Unit's ghost re-resolves.
        self._on_pending_changed = on_pending_changed or (lambda *args: None)
        self._on_garrison_add = on_garrison_add or (lambda *args: None)
        self._on_garrison_delete = on_garrison_delete or (lambda *args: None)
        self._on_garrison_navigate = on_garrison_navigate or (lambda *args: None)
        # GH #115: Pick from map (checked), Copy/Unload (row ids), Paste, and Owner (row ids, pid).
        self._on_garrison_pick = on_garrison_pick or (lambda *args: None)
        self._on_garrison_copy = on_garrison_copy or (lambda *args: None)
        self._on_garrison_paste = on_garrison_paste or (lambda *args: None)
        self._on_garrison_unload = on_garrison_unload or (lambda *args: None)
        self._on_garrison_owner = on_garrison_owner or (lambda *args: None)
        # GH #204: Cut (row ids). It marks rows; the next Paste moves them.
        self._on_garrison_cut = on_garrison_cut or (lambda *args: None)
        # True while show_unit()/show_group()/clear() are populating widgets
        # programmatically -- suppresses _field_changed() the same way
        # PlayersPanel._changed()'s own _populating guard does, so a
        # populate never looks like a user edit and never records a phantom
        # undo step.
        self._populating = False
        # Sticky, mirroring CatalogLineEdit._value (never cleared by a
        # filter or selection change that lands on nothing): the object the
        # next Place click/Enter places. See _on_catalog_current_changed.
        self._pending_object_const: int | None = None
        self._pending_name: str | None = None
        # GH #130: what the owner combos show, pushed by refresh_player_labels().
        self._player_labels = player_labels.DEFAULT_LABELS
        self._player_colors = None

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
        self._sync_team_colors()

    # -- catalog pane ------------------------------------------------------

    def _build_catalog_pane(self) -> QWidget:
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        # GH #68: category buttons, a class combo and three display modes,
        # sharing the cards' thumbnail loader; it lists the GH #164 random
        # rows too. Its list takes ClickFocus, not Qt's default: MapView's own
        # arrow-key unit nudge (map_view.py's keyPressEvent) only fires while
        # MapView has focus, and _on_catalog_activated is the common flow's
        # way back to it.
        self.catalog_view = CatalogBrowserView(loader=self.thumb_loader)
        self.catalog_view.current_changed.connect(self._on_catalog_current_changed)
        self.catalog_view.activated.connect(self._on_catalog_activated)
        pane_layout.addWidget(self.catalog_view, stretch=1)

        owner_row = QHBoxLayout()
        owner_row.setContentsMargins(0, 0, 0, 0)
        owner_row.addWidget(QLabel("Owner:"))
        self.owner_combo = QComboBox()
        for player_id in range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1):
            _add_player_item(self.owner_combo, player_id, self._player_labels, self._player_colors, player_id)
        self.owner_combo.setCurrentIndex(1)  # Player 1, preserved default
        self.owner_combo.currentIndexChanged.connect(lambda _index: self._on_pending_changed())
        self.owner_combo.currentIndexChanged.connect(lambda _index: self._sync_team_colors())
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
        if place_random.is_random(value):
            # GH #164: rolled now, so the ghost already shows what the next click places.
            self._rolled_const = place_random.roll(value, self._rng)
            name = place_random.label(value)
            # The card stands for the whole pick with one fixed const, not the roll; Unit ID lists every choice.
            shown = place_random.preview_const(value)
            const_text = " / ".join(str(const) for const in place_random.choices(value)) + " (random)"
        else:
            self._rolled_const = None
            name = object_catalog.name_for(object_catalog.objects(), value) or str(value)
            shown = value
            const_text = None
        self._pending_name = name
        self.placing_card.show_const(shown, prefix="Placing: ", name=name, const_text=const_text)
        # The name line elides a long name (the longest run 36 characters),
        # so the full one stays reachable by hover.
        self.placing_label.setToolTip(name)
        self._on_pending_changed()

    def selected_object_const(self) -> int | None:
        """The real const the next placement adds: a random pick's rolled one (GH #164)."""
        if place_random.is_random(self._pending_object_const):
            return self._rolled_const
        return self._pending_object_const

    def selected_object_choices(self) -> tuple[int, ...] | None:
        """Every const the pending pick may place (Scatter rolls per copy), or None with nothing picked."""
        if self._pending_object_const is None:
            return None
        return place_random.choices(self._pending_object_const)

    def selected_object_label(self) -> str | None:
        """The pending pick's name as "Placing:" shows it, or None with nothing picked."""
        return self._pending_name

    def reroll_pending(self) -> None:
        """After a placement: a random pick rolls its next const and the ghost follows. No-op for a real pick."""
        if not place_random.is_random(self._pending_object_const):
            return
        self._rolled_const = place_random.roll(self._pending_object_const, self._rng)
        self._on_pending_changed()

    def select_object(self, const: int) -> None:
        self.catalog_view.select(const)

    def pick_object(self, const: int) -> bool:
        """select_object() that also lands on a row the catalog filter hides,
        by clearing that filter (GH #125's wall button), or on an editor-hidden
        row, by ticking the show-hidden box. True if `const` is now the pending
        object; False, changing nothing, for a const the catalog doesn't list."""
        view = self.catalog_view
        entry = object_catalog.entry(object_catalog.objects(), const)
        if entry is None:
            return False
        # Before select(), as TerrainPanel.set_terrain does: a row the box hides never reads as picked.
        if entry.hidden and view.show_hidden_checkbox is not None:
            view.show_hidden_checkbox.setChecked(True)
        view.select(const)
        if view.current_value() != const and view.filter_edit.text():
            view.filter_edit.clear()
        # A row that became current while hidden emits nothing when the filter unhides it.
        if view.current_value() == const and self._pending_object_const != const:
            self._set_pending_object(const)
        return self.selected_object_const() == const

    def owner_id(self) -> int:
        return self.owner_combo.currentData()

    def refresh_player_labels(self, labels, colors) -> None:
        """GH #130: relabel the Place owner combo and the inspector's Owner
        field in place, keeping each one's selection and "(mixed)" row."""
        self._player_labels = labels
        self._player_colors = colors
        _relabel_player_rows(self.owner_combo, labels, colors)
        _relabel_player_rows(self.garrison_owner_combo, labels, colors)
        owner = self.unit_field_editors.get("player")
        if owner is not None:
            _relabel_player_rows(owner, labels, colors)
        self._sync_team_colors()

    def _team_rgb(self, player_id: int | None) -> tuple[int, int, int]:
        """The owner swatch's colour; GAIA's for None (a mixed-owner group)."""
        colors = self._player_colors if self._player_colors is not None else PLAYER_COLORS
        return colors[GAIA_PLAYER_ID if player_id is None else player_id]

    def _sync_team_colors(self) -> None:
        """Portrait fills: the Owner combo's for the pick and the list, the inspector Owner's for the selection."""
        placing = self._team_rgb(self.owner_combo.currentData())
        self.placing_card.set_team_color(placing)
        self.catalog_view.set_team_color(placing)
        self.selection_card.set_team_color(self._team_rgb(self.unit_field_editors["player"].currentData()))

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

        # GH #148's follow-up: the pick's portrait, Unit ID and base stats. It
        # shares one slot with the selection card below, shown only while
        # nothing is selected, so the two never compete for the pane's height.
        self.placing_card = UnitCard(stable=True, loader=self.thumb_loader)
        self.placing_card.clear("Placing: (none)")
        self.placing_label = self.placing_card.name_label
        self.inspector_layout.addWidget(self.placing_card)

        self.unit_field_labels: dict[str, QLabel] = {}
        self.unit_field_editors: dict[str, QWidget] = {}
        self.unit_field_captions: dict[str, QLabel] = {}
        # GH #148: portrait, name, Unit ID and stats; the names below are aliases of its widgets.
        self.selection_card = UnitCard(loader=self.thumb_loader)
        self.unit_header = self.selection_card
        self.unit_portrait = self.selection_card.portrait
        self.unit_stats_grid = self.selection_card.stats_grid
        self.unit_stat_rows = self.selection_card.stat_rows
        self.unit_field_labels["name"] = self.selection_card.name_label
        self.unit_field_labels["unit_const"] = self.selection_card.const_label
        self.unit_field_captions["unit_const"] = self.selection_card.const_caption
        self.inspector_layout.addWidget(self.selection_card)

        self.unit_inspector_grid = QWidget()
        grid = QGridLayout(self.unit_inspector_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        # Extra pane width goes to the editors: unstretched, Qt widens the captions first (TASK-247).
        grid.setColumnStretch(1, 1)
        # Group mode's "(mixed)" spinbox value, one step below the live minimum.
        # A FLOAT's is set per group by _set_spin_mixed (None when not mixed).
        self._mixed_sentinels: dict[str, float | None] = {}
        grid_specs = [spec for spec in unit_fields.FIELDS if spec.field_id not in _HEADER_FIELDS]
        for row, spec in enumerate(grid_specs):
            caption = QLabel(f"{spec.label}:")
            grid.addWidget(caption, row, 0)
            self.unit_field_captions[spec.field_id] = caption
            if spec.field_id == "rotation":
                # The full caveat, always reachable at zero vertical cost; never shown inline (GH #147).
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
                for player_id in range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1):
                    _add_player_item(combo, player_id, self._player_labels, self._player_colors, player_id)
                combo.currentIndexChanged.connect(
                    lambda _, s=spec, w=combo: self._field_changed(s, w.currentData())
                )
                combo.currentIndexChanged.connect(lambda _index: self._sync_team_colors())
                grid.addWidget(combo, row, 1)
                self.unit_field_editors[spec.field_id] = combo
            elif spec.kind == unit_fields.FACING:
                # GH #61: whole facings, one per wheel notch, wrapping n-1 -> 0.
                # The per-const range is set by _apply_conditional_fields.
                spin = QSpinBox()
                spin.setWrapping(True)
                spin.setKeyboardTracking(False)  # see the FLOAT branch below
                # TASK-023: the dial only drives the spinbox, which stays the one emitter.
                dial = facing_dial.FacingDial()
                dial.on_commit = spin.setValue
                dial.on_step = spin.stepBy
                # Before _field_changed, and read live: an edit's re-populate may set a newer value.
                spin.valueChanged.connect(lambda _value: self._sync_rotation_dial_value())
                spin.valueChanged.connect(
                    lambda value, s=spec: self._field_changed(s, value)
                )
                cell = QWidget()
                cell_layout = QHBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.addWidget(spin, 1)
                cell_layout.addWidget(dial, 0)
                grid.addWidget(cell, row, 1)
                self.unit_field_editors[spec.field_id] = spin
                self.unit_rotation_dial = dial
                self.unit_rotation_cell = cell
                self._mixed_sentinels[spec.field_id] = -1
            elif spec.kind == unit_fields.CHOICE:
                # GH #186. Never in _mixed_sentinels: a -1 sentinel would swallow
                # Capture Flag's "Default". Its "(mixed)" is an item with data None.
                combo = QComboBox()
                for raw, label in spec.choices:
                    combo.addItem(label, raw)
                # The longest label would otherwise set the pane's minimum width.
                _fit_combo_width(combo)
                combo.currentIndexChanged.connect(
                    lambda _, s=spec, w=combo: self._field_changed(s, w.currentData())
                )
                grid.addWidget(combo, row, 1)
                self.unit_field_editors[spec.field_id] = combo
            else:  # FLOAT: X, Y or Z
                spin = _CoordSpinBox()
                spin.setDecimals(spec.decimals)
                # Initial only: the map-derived range is applied at populate.
                spin.setRange(spec.minimum, spec.maximum)
                # setKeyboardTracking(False) is load-bearing the same way
                # viewer_common._make_spinbox's own comment explains: without
                # it, typing a multi-digit value over an old one fires
                # valueChanged once per digit, recording one undo step per
                # keystroke instead of one per commit (trap 3).
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(
                    lambda value, s=spec: self._field_changed(s, value)
                )
                spin.stepped.connect(lambda delta, s=spec: self._field_stepped(s, delta))
                grid.addWidget(spin, row, 1)
                self.unit_field_editors[spec.field_id] = spin
                self._mixed_sentinels[spec.field_id] = None
        self.inspector_layout.addWidget(self.unit_inspector_grid)

        # A separate grid (GH #42), on the stats grid's own pattern and for
        # the same reason: a garrison is a list of OTHER units, not a field of
        # the selected one. The panel stays model-free -- show_garrison() is
        # handed finished rows by the viewer, which owns the model.
        self.garrison_header = QLabel("<b>Garrison</b>")
        self.inspector_layout.addWidget(self.garrison_header)

        self.garrison_tree = QTreeWidget()
        self.garrison_tree.setHeaderLabels(["Unit", "Owner"])
        self.garrison_tree.setRootIsDecorated(False)
        self.garrison_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.garrison_tree.setMaximumHeight(_GARRISON_TREE_MAX_PX)
        self.garrison_tree.setToolTip(
            "Units inside this one. They are hidden on the map unless "
            "Filters > Show Garrisoned Units is on; double-click a row to select one"
        )
        self.garrison_tree.itemSelectionChanged.connect(self._garrison_selection_changed)
        self.garrison_tree.itemDoubleClicked.connect(self._garrison_double_clicked)
        self.inspector_layout.addWidget(self.garrison_tree)

        self.garrison_note = QLabel("")
        self.garrison_note.setWordWrap(True)
        self.inspector_layout.addWidget(self.garrison_note)

        # Three rows of at most three (GH #115; GH #204 regrouped them around Cut): a fourth
        # button overflows the inspector at MIN_USEFUL_WIDTH or leaves 8 px (measured offscreen).
        self.garrison_buttons = QWidget()
        rows_layout = QVBoxLayout(self.garrison_buttons)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        self.garrison_button_rows = (QWidget(), QWidget(), QWidget())
        for row in self.garrison_button_rows:
            QHBoxLayout(row).setContentsMargins(0, 0, 0, 0)
            rows_layout.addWidget(row)
        self.garrison_add_button = QPushButton("Add...")
        self.garrison_add_button.clicked.connect(lambda *_: self._on_garrison_add())
        self.garrison_pick_button = QPushButton("Pick from map")
        self.garrison_pick_button.setCheckable(True)
        self.garrison_pick_button.setToolTip("Click units on the map to put them inside; Esc or right-click to finish")
        self.garrison_pick_button.toggled.connect(self._garrison_pick_toggled)
        self.garrison_cut_button = QPushButton("Cut")
        self.garrison_cut_button.setToolTip(
            "Mark the selected rows to move: Paste inside another host moves the same units there, "
            "so triggers naming them keep working"
        )
        self.garrison_cut_button.clicked.connect(lambda *_: self._garrison_rows_clicked(self._on_garrison_cut))
        self.garrison_copy_button = QPushButton("Copy")
        self.garrison_copy_button.setToolTip("Copy the selected rows, to paste inside this or another host")
        self.garrison_copy_button.clicked.connect(lambda *_: self._garrison_rows_clicked(self._on_garrison_copy))
        self.garrison_paste_button = QPushButton("Paste")
        self.garrison_paste_button.setToolTip(
            "Add copies of the copied rows inside this host, or move the cut rows here; each keeps its own owner"
        )
        self.garrison_paste_button.clicked.connect(lambda *_: self._on_garrison_paste())
        self.garrison_unload_button = QPushButton("Unload")
        self.garrison_unload_button.setToolTip("Take the selected rows out, onto the tile beside this host")
        self.garrison_unload_button.clicked.connect(lambda *_: self._garrison_rows_clicked(self._on_garrison_unload))
        self.garrison_delete_button = QPushButton("Delete")
        self.garrison_delete_button.clicked.connect(lambda *_: self._garrison_delete_clicked())
        self.garrison_owner_combo = QComboBox()
        self.garrison_owner_combo.setToolTip("Give the selected rows a new owner")
        self.garrison_owner_combo.addItem("Owner...", None)
        for player_id in range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1):
            _add_player_item(self.garrison_owner_combo, player_id, self._player_labels, self._player_colors, player_id)
        self.garrison_owner_combo.activated.connect(self._garrison_owner_activated)
        groups = (
            (self.garrison_add_button, self.garrison_pick_button),
            (self.garrison_cut_button, self.garrison_copy_button, self.garrison_paste_button),
            (self.garrison_unload_button, self.garrison_delete_button, self.garrison_owner_combo),
        )
        for row, widgets in zip(self.garrison_button_rows, groups, strict=True):
            for widget in widgets:
                row.layout().addWidget(widget)
            row.layout().addStretch(1)
        self._set_garrison_row_actions_enabled(False)
        self.inspector_layout.addWidget(self.garrison_buttons)
        self.hide_garrison()

        self.inspector_layout.addStretch(1)
        self.inspector_area.setWidget(self.inspector_host)
        return self.inspector_area

    @property
    def portrait_source(self) -> str | None:
        """The selection card's: "icon", "sprite" or None."""
        return self.selection_card.portrait_source

    def reload_catalog(self) -> None:
        """After a game dataset switch (GH #67): the catalog's rows, re-read from
        object_catalog. The pending pick stays, under its new name, while the
        new catalog lists its const as the same object (object_catalog.
        same_object_in_aoe2), ticking the show-hidden box for a hidden row as
        pick_object() does. It is dropped otherwise: an empty slot (Mule Cart
        1808 under RoR) or a reused one (93, Spearman and Hoplite)."""
        self.catalog_view.reload_entries()
        pending = self._pending_object_const
        if pending is None or place_random.is_random(pending):
            return
        entry = object_catalog.entry(object_catalog.objects(), pending)
        if entry is not None and object_catalog.same_object_in_aoe2(pending):
            if entry.hidden and self.catalog_view.show_hidden_checkbox is not None:
                self.catalog_view.show_hidden_checkbox.setChecked(True)
            self._set_pending_object(pending)
            return
        self._pending_object_const = None
        self._rolled_const = None
        self._pending_name = None
        self.placing_card.clear("Placing: (none)")
        self.placing_label.setToolTip("")
        self._on_pending_changed()

    def refresh_thumbnails(self) -> None:
        """After an install change: the list's thumbnails and modes, and both cards' stat icons and portraits."""
        self.thumb_loader.clear()
        self.catalog_view.refresh_thumbnails()
        self.placing_card.refresh_thumbnails()
        self.selection_card.refresh_thumbnails()

    def _show_inspector_fields(self, visible: bool) -> None:
        self.unit_header.setVisible(visible)
        self.placing_card.setVisible(not visible)
        self.unit_inspector_grid.setVisible(visible)
        self.unit_inspector_empty.setVisible(not visible)

    def show_unit(self, entry) -> None:
        """Populates every field from `entry` (a unit_pick index entry), or
        blanks and hides the grid for None -- the "nothing selected" state.
        Never touches unit_inspector_empty's text; pair a None call with
        show_selection_count() when the caller has a count to report."""
        # Undo any group-mode state first: the "(mixed)" combo item, the
        # widened spin ranges, the hidden per-unit rows and the reworded note.
        self._populating = True
        try:
            self._leave_group_mode()
        finally:
            self._populating = False
        if entry is None:
            self._show_inspector_fields(False)
            for label in self.unit_field_labels.values():
                label.setText("")
            self.unit_field_labels["rotation"].setToolTip("")
            self.selection_card.clear()
            self.hide_garrison()
            self._fit_inspector_height()
            return
        unit = entry.unit
        # The viewer follows a show_unit() with show_garrison() when the
        # const is a host; anything else leaves the block hidden.
        self.hide_garrison()
        texts = {
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
            for field_id, value in (("x", unit.x), ("y", unit.y), ("z", getattr(unit, "z", 0.0))):
                spin = self.unit_field_editors[field_id]
                # Range before value, or setValue clamps silently to the
                # previous unit's range (GH #118).
                spin.setRange(*self._coordinate_range(field_id, [value]))
                spin.setValue(value)
            self._apply_choice_fields([unit])
            self._apply_conditional_fields([unit])
        finally:
            self._populating = False
        self.selection_card.show_const(unit.unit_const)
        self._show_inspector_fields(True)
        self._fit_inspector_height()

    def show_group(self, entries) -> None:
        """2+ selected units (GH #71): the same grid, each field showing the
        members' common value or "(mixed)". A typed value then applies to
        every member (ViewerWindow._on_unit_field_changed), while a step on
        X, Y or Z shifts every member by that step (GH #154, reported through
        on_unit_field_step). Reference ID and Garrisoned in are per-unit only,
        so their rows are hidden."""
        # A garrison is per-host, so the block has nothing to say about a
        # 2+ selection (GH #42), like Reference ID and Garrisoned in.
        self.hide_garrison()
        units = [entry.unit for entry in entries]
        consts = {unit.unit_const for unit in units}
        common_const = next(iter(consts)) if len(consts) == 1 else None
        # Stats and portrait only when every member shares one const.
        if common_const is None:
            self.selection_card.show_mixed(len(consts))
        else:
            self.selection_card.show_const(common_const)
        self.unit_field_labels["reference_id"].setText("")
        self.unit_field_labels["garrisoned_in_id"].setText("")
        coords = {
            "x": [unit.x for unit in units],
            "y": [unit.y for unit in units],
            "z": [getattr(unit, "z", 0.0) for unit in units],
        }
        self._populating = True
        try:
            self._set_group_owner({entry.player_id for entry in entries})
            for field_id, values in coords.items():
                self._set_group_spin(field_id, values)
                self.unit_field_editors[field_id].relative_steps = True
                self.unit_field_editors[field_id].setToolTip(_GROUP_STEP_TOOLTIP)
            self._apply_choice_fields(units)
            self._apply_conditional_fields(units)
        finally:
            self._populating = False
        self._show_inspector_fields(True)
        for field_id in ("reference_id", "garrisoned_in_id"):
            self.unit_field_captions[field_id].setVisible(False)
            self.unit_field_labels[field_id].setVisible(False)
        # _show_inspector_fields(True) hid the count line; group mode keeps it.
        self.unit_inspector_empty.setText(f"{len(entries)} units selected")
        self.unit_inspector_empty.setVisible(True)
        self._fit_inspector_height()

    def _leave_group_mode(self) -> None:
        """Restores single-unit state. Caller holds the populating guard:
        setRange clamping off the sentinel and removeItem both emit."""
        combo = self.unit_field_editors["player"]
        if combo.count() and combo.itemData(0) is None:
            combo.removeItem(0)
        for spec in unit_fields.FIELDS:
            if spec.kind == unit_fields.CHOICE:
                self._strip_choice_extras(spec)
        for field_id in self._mixed_sentinels:
            self._set_spin_mixed(field_id, False)
        for field_id in ("x", "y", "z"):
            self.unit_field_editors[field_id].relative_steps = False
            self.unit_field_editors[field_id].setToolTip("")
        for field_id in ("reference_id", "garrisoned_in_id"):
            self.unit_field_captions[field_id].setVisible(True)
            self.unit_field_labels[field_id].setVisible(True)
        self.unit_rotation_label.setToolTip(_ROTATION_TOOLTIP)

    def _set_group_owner(self, players: set[int]) -> None:
        combo = self.unit_field_editors["player"]
        has_mixed_item = combo.count() and combo.itemData(0) is None
        if len(players) == 1:
            if has_mixed_item:
                combo.removeItem(0)
            combo.setCurrentIndex(max(combo.findData(next(iter(players))), 0))
            return
        if not has_mixed_item:
            combo.insertItem(0, _MIXED_TEXT, None)
        combo.setCurrentIndex(0)

    def _strip_choice_extras(self, spec: unit_fields.UnitFieldSpec) -> None:
        """Takes a CHOICE combo back to its spec's own items: drops a leading
        "(mixed)" and a trailing "Other (N)". Caller holds the populating guard."""
        combo = self.unit_field_editors[spec.field_id]
        if combo.count() and combo.itemData(0) is None:
            combo.removeItem(0)
        while combo.count() > len(spec.choices):
            combo.removeItem(combo.count() - 1)

    def _apply_choice_fields(self, units) -> None:
        """Populates each CHOICE row (GH #186) for the selected unit(s): the
        shared raw value's label, "Other (N)" for a shared value the spec has
        no label for, or "(mixed)", repeated as the combo's tooltip. A row
        whose field the loaded file does not store is disabled on "Default"
        with a tooltip saying why.

        Only ever called with the populating guard already held, like
        _apply_conditional_fields: every step here emits currentIndexChanged.
        """
        for spec in unit_fields.FIELDS:
            if spec.kind != unit_fields.CHOICE:
                continue
            combo = self.unit_field_editors[spec.field_id]
            self._strip_choice_extras(spec)
            gate = spec.requires_codec_field
            supported = gate is None or self._codec_field_supported(gate)
            combo.setEnabled(supported)
            self.unit_field_captions[spec.field_id].setToolTip("" if supported else _CODEC_FIELD_TOOLTIPS[gate])
            if not supported:
                # The first choice is the field's default, the only value such a file can hold.
                combo.setCurrentIndex(0)
                combo.setToolTip(_CODEC_FIELD_TOOLTIPS[gate])
                continue
            values = {_CHOICE_READERS[spec.field_id](unit) for unit in units}
            if len(values) > 1:
                combo.insertItem(0, _MIXED_TEXT, None)
                index = 0
            else:
                value = next(iter(values))
                index = combo.findData(value)
                if index < 0:
                    combo.addItem(_OTHER_CHOICE.format(value=value), value)
                    index = combo.count() - 1
            combo.setCurrentIndex(index)
            # The closed combo cuts the longest label at MIN_USEFUL_WIDTH, so the full one is on hover.
            combo.setToolTip(combo.currentText())

    def _coordinate_range(self, field_id: str, values) -> tuple[float, float]:
        """X/Y/Z's live range (GH #118): the loaded map's bounds, or the
        spec's wide fallback with no scenario, widened over `values`."""
        size = self._map_size()
        if size is None:
            spec = unit_fields.FIELDS_BY_ID[field_id]
            bounds = (spec.minimum, spec.maximum)
        else:
            bounds = unit_fields.coordinate_bounds(field_id, *size)
        return unit_fields.widen(bounds, values)

    def _set_spin_mixed(self, field_id: str, mixed: bool, values=()) -> None:
        """"(mixed)" is the special value text, shown only at minimum(), so
        the range widens down to the sentinel and the value parks there.
        `values` widens a FLOAT's live range over the selected units'."""
        spec = unit_fields.FIELDS_BY_ID[field_id]
        spin = self.unit_field_editors[field_id]
        # A facing's maximum is per-const (set by _apply_conditional_fields),
        # so only its minimum toggles here.
        if spec.kind == unit_fields.FACING:
            minimum, maximum = 0, spin.maximum()
            step = 1
        else:
            minimum, maximum = self._coordinate_range(field_id, values)
            # One display step, not singleStep() (1.0): nothing off-map is
            # typeable between the sentinel and the minimum.
            step = 10 ** -spec.decimals
        if mixed:
            spin.setRange(minimum - step, maximum)
            if spec.kind != unit_fields.FACING:
                # Qt's rounded bound, which is what valueChanged will emit.
                self._mixed_sentinels[field_id] = spin.minimum()
            spin.setSpecialValueText(_MIXED_TEXT)
            spin.setValue(self._mixed_sentinels[field_id])
        else:
            # Cleared too: special text would otherwise show for a real 0.
            spin.setSpecialValueText("")
            spin.setRange(minimum, maximum)
            if spec.kind != unit_fields.FACING:
                self._mixed_sentinels[field_id] = None

    def _set_group_spin(self, field_id: str, values: list[float]) -> None:
        # Compared at the spinbox's own decimals: raw == would read units
        # rotated or moved by different paths as "(mixed)" forever.
        spec = unit_fields.FIELDS_BY_ID[field_id]
        facing = spec.kind == unit_fields.FACING
        decimals = 0 if facing else spec.decimals
        # A facing's range is per-const, so only a FLOAT widens over values.
        range_values = () if facing else values
        if len({round(value, decimals) for value in values}) == 1:
            self._set_spin_mixed(field_id, False, range_values)
            self.unit_field_editors[field_id].setValue(values[0])
        else:
            self._set_spin_mixed(field_id, True, range_values)

    def _apply_conditional_fields(self, units) -> None:
        """Swaps each conditional field between its editor and its read-only
        label for the selected unit(s) -- today only Rotation, whose rule is
        unit_variant.is_rotation_editable. The editor shows if ANY unit's const
        admits it, or if every selected unit is a gate (rotation_field_mode's
        "gate" mode); for a group, the Rotation caption's tooltip leads with a
        note if a typed value would skip any member, never an inline label
        (GH #198). A lone read-only unit gets a tooltip on its value (GH #147).

        Only ever called with the populating guard already held: it sets an
        editor's value, and the resulting valueChanged must not record a
        phantom undo step.

        Rotation has three modes (unit_variant.rotation_field_mode). As a facing
        (GH #61), rotation_to_facing wraps a junk stored value like 7.0 (574
        corpus placements) onto its own frame, and a group whose members
        differ in direction count edits on the finest grid. As a variant
        (GH #123), on a selection with no ANGLE member, the range is the fewest
        variants any member has. As a gate orientation (GH #61), on an all-gate
        selection, 0..3 in Rotate's order. Every way the range is set before the value,
        or the value would clamp to the previous selection's maximum.
        """
        group = len(units) > 1
        for spec in unit_fields.FIELDS:
            if not spec.conditional:
                continue
            rule = _FIELD_CONDITIONALS[spec.conditional]
            admitted = [unit for unit in units if rule(unit.unit_const)]
            # Over every unit, as the viewer asks it: "gate" needs the whole selection.
            mode = unit_variant.rotation_field_mode(unit.unit_const for unit in units)
            allowed = bool(admitted) or (spec.field_id == "rotation" and mode == "gate")
            editor = self.unit_field_editors[spec.field_id]
            editor.setVisible(allowed)
            label = self.unit_field_labels[spec.field_id]
            label.setVisible(not allowed)
            if spec.field_id != "rotation":
                continue
            # Counted from the active mode's writable set, not the rule's
            # admitted one: in facing mode a typed value skips a tree too.
            if mode == "facing":
                writable = [unit for unit in admitted if unit_rotation.rotation_is_angle(unit.unit_const)]
            elif mode == "gate":
                writable = list(units)
            else:
                writable = admitted
            skipped = len(units) - len(writable)
            # GH #147: a lone read-only unit explains itself in a tooltip on its value.
            readonly_single = not group and not allowed
            label.setToolTip(_readonly_rotation_tooltip(units[0].unit_const) if readonly_single else "")
            # Only for a group a typed value would skip part of; "gate" mode skips nobody.
            note = ""
            if group and skipped > 0:
                if mode == "variant":
                    turned = len(writable)
                    note = _GROUP_VARIANT_NOTE.format(
                        turned=turned, objects="object" if turned == 1 else "objects", skipped=skipped, total=len(units)
                    )
                    # Tail over the skipped members only: the typed value already steps the trees.
                    note += _rotate_tail([unit for unit in units if not unit_variant.is_cyclable(unit.unit_const)])
                elif mode == "facing":
                    turned = len(writable)
                    note = _GROUP_ROTATION_NOTE.format(
                        turned=turned, units="unit" if turned == 1 else "units", skipped=skipped, total=len(units)
                    )
                    note += _rotate_tail(units)
                else:  # mode is None: nothing in the group takes a typed value
                    note = _GROUP_NO_ROTATION_NOTE.format(total=len(units)) + _rotate_tail(units)
            # GH #198: the note leads the caption's tooltip instead of taking an inline line.
            self.unit_rotation_label.setToolTip(f"{note}\n\n{_ROTATION_TOOLTIP}" if note else _ROTATION_TOOLTIP)
            if group and not allowed:
                label.setText("(n/a)")
            if mode == "facing":
                scale = unit_rotation.facing_scale(unit.unit_const for unit in writable)
                editor.setRange(editor.minimum(), scale - 1)
                values = [unit_rotation.rotation_to_facing(unit.rotation, scale) for unit in writable]
                self._set_group_spin(spec.field_id, values)
                editor.setToolTip(self._facing_tooltip(writable, scale))
            elif mode == "variant":
                scale = unit_variant.variant_scale(unit.unit_const for unit in writable)
                editor.setRange(editor.minimum(), scale - 1)
                self._set_group_spin(spec.field_id, [_variant_of(unit) for unit in writable])
                editor.setToolTip(self._variant_tooltip(writable, scale))
            elif mode == "gate":
                editor.setRange(editor.minimum(), len(_GATE_ORIENTATION_NAMES) - 1)
                self._set_group_spin(spec.field_id, [_gate_index(unit) for unit in writable])
                editor.setToolTip(self._gate_tooltip(writable))
            # TASK-023: shown and hidden with the spinbox, so a read-only label never sits under an empty cell.
            self.unit_rotation_cell.setVisible(allowed)
            dial = self.unit_rotation_dial
            dial_shown = allowed and mode in ("facing", "gate")
            dial.setVisible(dial_shown)
            if dial_shown:
                dial.set_mode(mode, editor.maximum() + 1)
                dial.setToolTip(editor.toolTip())
                self._sync_rotation_dial_value()

    def _sync_rotation_dial_value(self) -> None:
        """The dial follows the spinbox; its "(mixed)" sentinel (-1) shows no pointer."""
        value = self.unit_field_editors["rotation"].value()
        self.unit_rotation_dial.set_value(None if value < 0 else value)

    def set_rotation_projection(self, flat: bool) -> None:
        """TASK-023: draw the dial as a plain circle (Flat) or with the iso 2:1 squash."""
        self.unit_rotation_dial.set_flat(flat)

    @staticmethod
    def _facing_tooltip(units, scale: int) -> str:
        if len(units) == 1:
            unit = units[0]
            facing = unit_rotation.rotation_to_facing(unit.rotation, scale)
            return f"Facing {facing} of {scale} (stored: {unit.rotation:.4f} rad)"
        counts = sorted({unit_rotation.angle_count_for(unit.unit_const) for unit in units})
        if len(counts) == 1:
            return f"Facing 0 to {scale - 1} of {scale} for every selected unit"
        others = ", ".join(str(count) for count in counts[:-1])
        return (
            f"Facing on a {scale}-direction scale. Units with fewer directions "
            f"({others}) turn to their nearest frame."
        )

    @staticmethod
    def _gate_tooltip(units) -> str:
        order = ", ".join(f"{i} {name}" for i, name in enumerate(_GATE_ORIENTATION_NAMES))
        if len(units) == 1:
            index = _gate_index(units[0])
            head = f"Orientation {index} of 4: {_GATE_ORIENTATION_NAMES[index]} (const {units[0].unit_const})"
        else:
            head = "Orientation for every selected gate"
        return f"{head}. Order {order}, the order Rotate steps."

    @staticmethod
    def _variant_tooltip(units, scale: int) -> str:
        if len(units) == 1:
            unit = units[0]
            return f"Variant {_variant_of(unit)} of {scale} (stored: {unit.rotation:g})"
        counts = sorted({unit_variant.variant_count_for(unit.unit_const) for unit in units})
        if len(counts) == 1:
            return f"Variant 0 to {scale - 1} of {scale} for every selected object"
        return (
            f"Variant 0 to {scale - 1}: the fewest any selected object has. "
            f"Others have {', '.join(str(count) for count in counts[1:])}."
        )

    def clear(self) -> None:
        """Blanks the inspector with no count to report -- the mode-switch
        paths (on_mode_changed, _force_mode), which reset the selection
        outright rather than reconciling it."""
        self.show_unit(None)

    def show_selection_count(self, count: int) -> None:
        """Hides the field grids and shows a "No unit selected" / "N units
        selected" readout. The viewer calls it only for an empty selection:
        one unit goes to show_unit(), 2+ to show_group()."""
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

    # -- garrison block (GH #42) ---------------------------------------------

    def show_garrison(self, rows, capacity: int, wrong_type: int = 0, can_paste: bool = False, cut=()) -> None:
        """The Garrison block for the one selected host.

        `rows` is a sequence of (label, owner, reference_id) the viewer built
        from the model -- this panel never reads a model itself. `capacity` is
        the base .dat figure, and `wrong_type` counts occupants the game
        itself would not admit; both are reported, never corrected, since a
        file is shown as it is. `can_paste` is the viewer's call (GH #115): a
        garrison clipboard holding something, and a host that can hold it.
        `cut` is the reference_ids a pending Cut marked here (GH #204).
        Pick from map's checked state is not touched; see set_garrison_pick_armed().
        """
        rows = list(rows)
        self.garrison_header.setText(f"<b>Garrison ({len(rows)} / {capacity})</b>")
        self.garrison_tree.clear()
        for label, owner, reference_id in rows:
            item = QTreeWidgetItem([label, owner])
            item.setData(0, Qt.UserRole, reference_id)
            self.garrison_tree.addTopLevelItem(item)
        self.set_garrison_cut(cut)
        notes = []
        if len(rows) > capacity:
            notes.append(_GARRISON_OVER_CAPACITY.format(count=len(rows), capacity=capacity))
        if wrong_type:
            notes.append(_GARRISON_WRONG_TYPE.format(count=wrong_type))
        self.garrison_note.setText(" ".join(notes))
        self.garrison_note.setVisible(bool(notes))
        self.garrison_add_button.setEnabled(len(rows) < capacity)
        self.garrison_add_button.setToolTip(
            "" if len(rows) < capacity else f"Full: the game gives this one {capacity} places"
        )
        # An armed pick stays enabled so it can still be switched off.
        self.garrison_pick_button.setEnabled(len(rows) < capacity or self.garrison_pick_button.isChecked())
        self.garrison_paste_button.setEnabled(can_paste)
        self._set_garrison_row_actions_enabled(False)
        for widget in (self.garrison_header, self.garrison_tree, self.garrison_buttons):
            widget.setVisible(True)
        self._fit_inspector_height()

    def hide_garrison(self) -> None:
        """Whenever the selection isn't a single unit that can hold one."""
        self.garrison_tree.clear()
        for widget in (self.garrison_header, self.garrison_tree, self.garrison_note, self.garrison_buttons):
            widget.setVisible(False)

    def garrison_selection(self) -> list[int]:
        """The reference_ids of the selected occupant rows."""
        return [item.data(0, Qt.UserRole) for item in self.garrison_tree.selectedItems()]

    def set_garrison_can_paste(self, can_paste: bool) -> None:
        """Paste's enablement alone, so a Copy or a Cut keeps its rows selected."""
        self.garrison_paste_button.setEnabled(can_paste)

    def set_garrison_cut(self, reference_ids) -> None:
        """Draws the rows a pending Cut holds italic and dimmed, and every
        other row plain, in place: a Cut keeps the rows it cut selected (GH #204)."""
        cut = set(reference_ids)
        italic = QFont()
        italic.setItalic(True)
        dimmed = QBrush(self.garrison_tree.palette().color(QPalette.Disabled, QPalette.Text))
        for index in range(self.garrison_tree.topLevelItemCount()):
            item = self.garrison_tree.topLevelItem(index)
            marked = item.data(0, Qt.UserRole) in cut
            for column in range(self.garrison_tree.columnCount()):
                # None clears the role, back to the tree's own font, colour and tooltip.
                item.setData(column, Qt.FontRole, italic if marked else None)
                item.setData(column, Qt.ForegroundRole, dimmed if marked else None)
                item.setData(column, Qt.ToolTipRole, _GARRISON_CUT_TIP if marked else None)

    def set_garrison_pick_armed(self, armed: bool) -> None:
        """The viewer's word on Pick from map, shown without reporting back."""
        self.garrison_pick_button.blockSignals(True)
        self.garrison_pick_button.setChecked(armed)
        self.garrison_pick_button.blockSignals(False)
        if armed:
            self.garrison_pick_button.setEnabled(True)

    def _set_garrison_row_actions_enabled(self, enabled: bool) -> None:
        for widget in (
            self.garrison_cut_button,
            self.garrison_copy_button,
            self.garrison_unload_button,
            self.garrison_delete_button,
            self.garrison_owner_combo,
        ):
            widget.setEnabled(enabled)

    def _garrison_selection_changed(self) -> None:
        self._set_garrison_row_actions_enabled(bool(self.garrison_tree.selectedItems()))

    def _garrison_pick_toggled(self, checked: bool) -> None:
        self._on_garrison_pick(checked)

    def _garrison_rows_clicked(self, callback) -> None:
        selected = self.garrison_selection()
        if selected:
            callback(selected)

    def _garrison_owner_activated(self, index: int) -> None:
        player_id = self.garrison_owner_combo.itemData(index)
        # Back to the prompt row first, so picking the same owner again still fires.
        self.garrison_owner_combo.setCurrentIndex(0)
        selected = self.garrison_selection()
        if player_id is not None and selected:
            self._on_garrison_owner(selected, player_id)

    def _garrison_delete_clicked(self) -> None:
        selected = self.garrison_selection()
        if selected:
            self._on_garrison_delete(selected)

    def _garrison_double_clicked(self, item, _column: int) -> None:
        self._on_garrison_navigate(item.data(0, Qt.UserRole))

    # -- reporting an edit ---------------------------------------------------

    def _field_changed(self, spec: unit_fields.UnitFieldSpec, value) -> None:
        if self._populating:
            return
        # Landing on a group-mode "(mixed)" placeholder is not an edit.
        if value is None or value == self._mixed_sentinels.get(spec.field_id):
            return
        self._on_unit_field(spec, value)

    def _field_stepped(self, spec: unit_fields.UnitFieldSpec, delta: float) -> None:
        if self._populating:
            return
        self._on_unit_field_step(spec, delta)
