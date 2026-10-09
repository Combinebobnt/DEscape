"""Edit > Find and Replace (GH #144): a modeless, tabbed results dialog.

Dumb the same way AnalysisDialog is: it never touches the map. Every search
and every operation goes through the FindCallbacks it was built with, and
the window decides what "find", "delete" or "select" means. Rows hold keys,
never Unit objects, so a stale row can only fail to resolve.

Staleness: the window calls mark_stale() after any unit edit (undo included).
One QTimer.singleShot(0) re-runs the last search while the dialog is shown,
so a drag coalesces to one re-run; a hidden dialog re-runs on its next show.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import find_triggers, object_catalog, player_labels
from descape.constant_picker import CatalogBrowseDialog, CatalogLineEdit
from descape.find_objects import CATEGORIES, FindCriteria, FindRow, rows_to_csv
from descape.find_text import FindPatternError, compile_pattern
from descape.find_triggers import TriggerFindCriteria, TriggerHit, hits_to_csv
from descape.scenario_write import WriteBlockedError
from descape.value_picker import _configure_scrolling
from descape.viewer_common import _add_player_item, _player_icon, _relabel_player_rows

_KIND_LABELS = (
    ("trees", "Trees"),
    ("eye_candy", "Eye candy"),
    ("walls", "Walls and gates"),
    ("invisible", "Invisible"),
)
OBJECT_COLUMNS = ("Name", "ID", "Type id", "Owner", "X", "Y", "Z", "Caption", "In", "Holds", "Triggers", "Note")
_COL_TRIGGERS = OBJECT_COLUMNS.index("Triggers")
_COL_NOTE = OBJECT_COLUMNS.index("Note")
_TRIGGERS_COLUMN_TIP = "Trigger conditions/effects naming this object. Blank until triggers are read (open the Triggers tab)"
_GREY = QBrush(QColor(128, 128, 128))
_ERROR_STYLE = "color: #c62828;"
AREA_WHOLE, AREA_REGION, AREA_CUSTOM = 0, 1, 2


def confirm_with_choices(
    parent,
    title: str,
    text: str,
    choices: Sequence[tuple[str, str, bool]] = (),
    actions: Sequence[tuple[str, Callable[[], None]]] = (),
) -> dict | None:
    """A modal OK/Cancel box with optional checkboxes. None when cancelled,
    else {key: checked}. `choices` is (key, label, checked by default);
    `actions` are (label, callback) buttons that run without closing it."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(text)
    label.setWordWrap(True)
    layout.addWidget(label)
    boxes = {}
    for key, choice_label, checked in choices:
        box = QCheckBox(choice_label)
        box.setChecked(checked)
        layout.addWidget(box)
        boxes[key] = box
    if actions:
        row = QHBoxLayout()
        for action_label, callback in actions:
            button = QPushButton(action_label)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, c=callback: c())
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec_() != QDialog.Accepted:
        return None
    return {key: box.isChecked() for key, box in boxes.items()}


@dataclass
class FindCallbacks:
    """What the window lends the dialog. Keys are (player, reference_id)."""

    find_objects: Callable[[FindCriteria], list[FindRow]]
    select_keys: Callable[[list], tuple[int, int]] = lambda keys: (0, 0)
    centre_on_key: Callable[[tuple[int, int]], bool] = lambda key: False
    delete_keys: Callable[[list], tuple[int, int, int]] = lambda keys: (0, 0, 0)
    replace_preflight: Callable[[list, int], dict] = lambda keys, const: {}
    replace_keys: Callable[..., tuple[int, dict]] = lambda keys, const, follow_writes=(): (0, {})
    reassign_keys: Callable[[list, int], dict] = lambda keys, player: {}
    region: Callable[[], tuple[int, int, int, int] | None] = lambda: None
    warn: Callable[[str, str], None] = lambda title, text: None
    confirm: Callable[..., dict | None] = lambda title, text, choices=(), actions=(): {}
    # -- Part B: triggers --
    trigger_status: Callable[[], tuple[str | None, bool]] = lambda: ("No map open.", False)
    find_triggers: Callable[[TriggerFindCriteria], list[TriggerHit]] = lambda criteria: []
    text_replace: Callable[..., dict] = lambda hits, text, regex, case, replacement: {}
    type_replace: Callable[..., dict] = lambda hits, old_consts, new_const: {}
    set_enabled: Callable[[dict, bool], int] = lambda parents, value: 0
    delete_triggers: Callable[[dict], int] = lambda parents: 0
    reveal: Callable[[int, str, int], None] = lambda trigger_index, kind, entry_index: None
    follow_preview: Callable[[list, int], tuple] = lambda keys, const: (None, "")
    trigger_refs_for: Callable[[list], int | None] = lambda keys: None


def _player_text(labels, pid: int) -> str:
    return labels.text[pid] if 0 <= pid < len(labels.text) else f"P{pid}"


class _ObjectsTab(QWidget):
    def __init__(self, dialog: FindDialog):
        super().__init__(dialog)
        self._dialog = dialog
        self._labels = player_labels.DEFAULT_LABELS
        self._colors = None
        self.type_consts: list[int] = []
        self.last_criteria: FindCriteria | None = None
        self.rows: list[FindRow] = []
        self.checked: set[tuple[int, int]] = set()
        self.notes: dict[tuple[int, int], str] = {}
        self._filling = False

        # -- search panel ------------------------------------------------
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name, or #id")
        self.name_edit.returnPressed.connect(self.run_find)
        self.name_edit.textChanged.connect(self._validate)
        self.regex_box = QCheckBox("Regex")
        self.case_box = QCheckBox("Match case")
        self.captions_box = QCheckBox("Captions")
        for box in (self.regex_box, self.case_box):
            box.toggled.connect(self._validate)
        self.pattern_error = QLabel("")
        self.pattern_error.setStyleSheet(_ERROR_STYLE)
        self.pattern_error.setVisible(False)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        name_row.addWidget(self.name_edit, stretch=1)
        name_row.addWidget(self.regex_box)
        name_row.addWidget(self.case_box)
        name_row.addWidget(self.captions_box)

        self.owner_list = QListWidget()
        # Wrapping columns, so nine owners fit beside the kind boxes without a scrollbar.
        self.owner_list.setFlow(QListWidget.TopToBottom)
        self.owner_list.setWrapping(True)
        self.owner_list.setResizeMode(QListWidget.Adjust)
        self.owner_list.setMaximumHeight(130)
        for pid in range(9):
            item = QListWidgetItem(_player_text(self._labels, pid))
            item.setData(Qt.UserRole, pid)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.owner_list.addItem(item)
        owner_box = QGroupBox("Owner")
        owner_layout = QVBoxLayout(owner_box)
        owner_layout.addWidget(self.owner_list)

        self.category_boxes = {name: QCheckBox(name) for name in CATEGORIES}
        self.kind_boxes = {key: QCheckBox(label) for key, label in _KIND_LABELS}
        kind_box = QGroupBox("Category / kind")
        kind_layout = QGridLayout(kind_box)
        for i, box in enumerate(self.category_boxes.values()):
            box.setChecked(True)
            kind_layout.addWidget(box, i, 0)
        for i, box in enumerate(self.kind_boxes.values()):
            box.setToolTip("Only this kind (none ticked = any kind)")
            kind_layout.addWidget(box, i, 1)

        self.types_button = QPushButton("Types…")
        self.types_button.clicked.connect(self._pick_type)
        self.types_clear = QPushButton("Clear")
        self.types_clear.clicked.connect(self.clear_types)
        self.chips = QHBoxLayout()
        self.chips.setContentsMargins(0, 0, 0, 0)
        types_row = QHBoxLayout()
        types_row.addWidget(self.types_button)
        types_row.addLayout(self.chips)
        types_row.addStretch(1)
        types_row.addWidget(self.types_clear)

        self.area_combo = QComboBox()
        self.area_combo.addItems(["Whole map", "Current selection region", "Custom"])
        self.area_combo.currentIndexChanged.connect(self._area_changed)
        self.area_spins = []
        for _ in range(4):
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.setVisible(False)
            self.area_spins.append(spin)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Area:"))
        area_row.addWidget(self.area_combo)
        for spin, tip in zip(self.area_spins, ("x0", "y0", "x1 (exclusive)", "y1 (exclusive)"), strict=True):
            spin.setToolTip(tip)
            area_row.addWidget(spin)
        area_row.addStretch(1)

        self.garrisoned_box = QCheckBox("Include garrisoned")
        self.garrisoned_box.setChecked(True)
        self.off_map_box = QCheckBox("Include off-map")
        self.off_map_box.setChecked(True)
        self.find_button = QPushButton("Find")
        self.find_button.setDefault(True)
        self.find_button.clicked.connect(self.run_find)
        find_row = QHBoxLayout()
        find_row.addWidget(self.garrisoned_box)
        find_row.addWidget(self.off_map_box)
        find_row.addStretch(1)
        find_row.addWidget(self.find_button)

        facets = QHBoxLayout()
        facets.addWidget(owner_box, stretch=1)
        facets.addWidget(kind_box, stretch=1)

        # -- results -------------------------------------------------------
        self.count_label = QLabel("")
        self.unselectable_label = QLabel("")
        self.check_all = QPushButton("Check all")
        self.check_all.clicked.connect(lambda: self._set_all_checked(True))
        self.check_none = QPushButton("Check none")
        self.check_none.clicked.connect(lambda: self._set_all_checked(False))
        header_row = QHBoxLayout()
        header_row.addWidget(self.count_label)
        header_row.addWidget(self.unselectable_label)
        header_row.addStretch(1)
        header_row.addWidget(self.check_all)
        header_row.addWidget(self.check_none)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(OBJECT_COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        _configure_scrolling(self.tree)
        # Fixed starting widths, never ResizeToContents: a map can hold ~14k rows.
        for col, chars in enumerate((22, 7, 7, 10, 7, 7, 5, 14, 7, 6, 8, 30)):
            self.tree.setColumnWidth(col, self.fontMetrics().averageCharWidth() * chars + 24)
        self.tree.headerItem().setToolTip(_COL_TRIGGERS, _TRIGGERS_COLUMN_TIP)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)

        # -- operations ----------------------------------------------------
        self.select_button = QPushButton("Select on Map")
        self.select_button.clicked.connect(self.select_on_map)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_checked)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self.export_csv)
        self.usages_button = QPushButton("Show trigger usages")
        self.usages_button.setToolTip("List the triggers that name the checked objects, in the Triggers tab")
        self.usages_button.clicked.connect(self.show_trigger_usages)
        self.replace_edit = CatalogLineEdit(object_catalog.objects(), default_category="Units", placeholder="(object)")
        self.replace_button = QPushButton("Replace")
        self.replace_button.clicked.connect(self.replace_checked)
        self.owner_combo = QComboBox()
        for pid in range(9):
            _add_player_item(self.owner_combo, pid, self._labels, self._colors, pid)
        self.owner_apply = QPushButton("Apply")
        self.owner_apply.clicked.connect(self.change_owner)
        ops_row = QHBoxLayout()
        ops_row.addWidget(self.select_button)
        ops_row.addWidget(self.delete_button)
        ops_row.addWidget(self.export_button)
        ops_row.addStretch(1)
        ops_row.addWidget(self.usages_button)
        replace_row = QHBoxLayout()
        replace_row.addWidget(QLabel("Replace with:"))
        replace_row.addWidget(self.replace_edit, stretch=1)
        replace_row.addWidget(self.replace_button)
        replace_row.addSpacing(12)
        replace_row.addWidget(QLabel("Owner:"))
        replace_row.addWidget(self.owner_combo)
        replace_row.addWidget(self.owner_apply)

        layout = QVBoxLayout(self)
        layout.addLayout(name_row)
        layout.addWidget(self.pattern_error)
        layout.addLayout(facets)
        layout.addLayout(types_row)
        layout.addLayout(area_row)
        layout.addLayout(find_row)
        layout.addLayout(header_row)
        layout.addWidget(self.tree, stretch=1)
        layout.addLayout(ops_row)
        layout.addLayout(replace_row)
        self.set_region_available(False)
        self._rebuild_chips()
        self._update_ops()

    # -- search panel ----------------------------------------------------------

    def _validate(self) -> bool:
        try:
            compile_pattern(self.name_edit.text(), self.regex_box.isChecked(), self.case_box.isChecked())
        except FindPatternError as e:
            self.pattern_error.setText(str(e))
            self.pattern_error.setVisible(True)
            self.find_button.setEnabled(False)
            return False
        self.pattern_error.setVisible(False)
        self.find_button.setEnabled(True)
        return True

    def refresh_player_labels(self, labels, colors) -> None:
        self._labels, self._colors = labels, colors
        for i in range(self.owner_list.count()):
            item = self.owner_list.item(i)
            pid = item.data(Qt.UserRole)
            item.setText(labels.text[pid])
            item.setToolTip(labels.full[pid])
            item.setIcon(_player_icon(pid, colors))
        _relabel_player_rows(self.owner_combo, labels, colors)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            row = item.data(0, Qt.UserRole)
            if isinstance(row, FindRow):
                item.setText(3, _player_text(labels, row.player))

    def set_region_available(self, available: bool) -> None:
        model = self.area_combo.model()
        model.item(AREA_REGION).setEnabled(available)
        if not available and self.area_combo.currentIndex() == AREA_REGION:
            self.area_combo.setCurrentIndex(AREA_WHOLE)

    def _area_changed(self, index: int) -> None:
        for spin in self.area_spins:
            spin.setVisible(index == AREA_CUSTOM)

    def add_type_const(self, const: int) -> None:
        if const in self.type_consts:
            return
        self.type_consts.append(const)
        self._rebuild_chips()

    def clear_types(self) -> None:
        self.type_consts.clear()
        self._rebuild_chips()

    def _remove_type(self, const: int) -> None:
        if const in self.type_consts:
            self.type_consts.remove(const)
        self._rebuild_chips()

    def _rebuild_chips(self) -> None:
        _fill_chips(self.chips, self.type_consts, self._remove_type)
        self.types_clear.setEnabled(bool(self.type_consts))

    def _pick_type(self) -> None:
        dialog = CatalogBrowseDialog(object_catalog.objects(), default_category="Units", parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_id() is not None:
            self.add_type_const(dialog.selected_id())

    def criteria(self) -> FindCriteria | None:
        """The panel as a FindCriteria, or None when the area is invalid."""
        players = frozenset(
            self.owner_list.item(i).data(Qt.UserRole)
            for i in range(self.owner_list.count())
            if self.owner_list.item(i).checkState() == Qt.Checked
        )
        categories = frozenset(name for name, box in self.category_boxes.items() if box.isChecked())
        kinds = frozenset(key for key, box in self.kind_boxes.items() if box.isChecked())
        area = None
        if self.area_combo.currentIndex() == AREA_REGION:
            area = self._dialog.callbacks.region()
            if area is None:
                return None
        elif self.area_combo.currentIndex() == AREA_CUSTOM:
            area = tuple(spin.value() for spin in self.area_spins)
        return FindCriteria(
            text=self.name_edit.text(),
            regex=self.regex_box.isChecked(),
            case_sensitive=self.case_box.isChecked(),
            match_captions=self.captions_box.isChecked(),
            players=None if len(players) == 9 else players,
            categories=None if len(categories) == len(CATEGORIES) else categories,
            kinds=kinds or None,
            consts=frozenset(self.type_consts) or None,
            area=area,
            include_garrisoned=self.garrisoned_box.isChecked(),
            include_off_map=self.off_map_box.isChecked(),
        )

    def run_find(self) -> None:
        if not self._validate():
            return
        criteria = self.criteria()
        if criteria is None:
            self._dialog.callbacks.warn("Find and Replace", "There is no selected region to search.")
            return
        self.last_criteria = criteria
        self.checked.clear()
        self.notes.clear()
        self._fill(criteria, check_all=True)

    def rerun(self) -> None:
        """Re-runs the last search, keeping the checked keys and refusal notes."""
        if self.last_criteria is not None:
            self._fill(self.last_criteria, check_all=False)

    def _fill(self, criteria: FindCriteria, check_all: bool) -> None:
        try:
            rows = self._dialog.callbacks.find_objects(criteria)
        except FindPatternError as e:
            self.pattern_error.setText(str(e))
            self.pattern_error.setVisible(True)
            return
        self.rows = rows
        live = {row.key for row in rows if not row.duplicate_ref}
        self.checked = set(live) if check_all else self.checked & live
        self._filling = True
        self.tree.setUpdatesEnabled(False)
        self.tree.setSortingEnabled(False)
        try:
            self.tree.clear()
            items = [self._item(row) for row in rows]
            self.tree.addTopLevelItems(items)
        finally:
            self.tree.setUpdatesEnabled(True)
            self._filling = False
        unselectable = sum(1 for row in rows if row.duplicate_ref or row.off_map)
        self.count_label.setText(f"{len(rows)} found")
        self.unselectable_label.setText(f"({unselectable} not selectable)" if unselectable else "")
        self._update_ops()

    def _item(self, row: FindRow) -> QTreeWidgetItem:
        item = QTreeWidgetItem(
            [
                row.name,
                str(row.reference_id),
                str(row.unit_const),
                _player_text(self._labels, row.player),
                f"{row.x:g}",
                f"{row.y:g}",
                f"{row.z:g}",
                row.caption,
                "" if row.garrisoned_in == -1 else str(row.garrisoned_in),
                str(row.occupants) if row.occupants else "",
                "" if row.trigger_refs is None else str(row.trigger_refs),
                self.notes.get(row.key, ""),
            ]
        )
        item.setData(0, Qt.UserRole, row)
        if row.duplicate_ref:
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            tip = "Its reference id is placed more than once, so every operation refuses it"
        else:
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if row.key in self.checked else Qt.Unchecked)
            tip = "Off the map: Select on Map and Replace refuse it" if row.off_map else ""
        if tip:
            for col in range(len(OBJECT_COLUMNS)):
                item.setForeground(col, _GREY)
                item.setToolTip(col, tip)
        note = self.notes.get(row.key)
        if note:
            item.setToolTip(_COL_NOTE, note)
        return item

    # -- results ----------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._filling or column != 0:
            return
        row = item.data(0, Qt.UserRole)
        if not isinstance(row, FindRow) or row.duplicate_ref:
            return
        if item.checkState(0) == Qt.Checked:
            self.checked.add(row.key)
        else:
            self.checked.discard(row.key)
        self._update_ops()

    def _set_all_checked(self, checked: bool) -> None:
        self._filling = True
        try:
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                row = item.data(0, Qt.UserRole)
                if isinstance(row, FindRow) and not row.duplicate_ref:
                    item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        finally:
            self._filling = False
        self.checked = {row.key for row in self.rows if not row.duplicate_ref} if checked else set()
        self._update_ops()

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        row = item.data(0, Qt.UserRole)
        if isinstance(row, FindRow):
            self._dialog.callbacks.centre_on_key(row.key)

    def checked_rows(self) -> list[FindRow]:
        return [row for row in self.rows if row.key in self.checked and not row.duplicate_ref]

    def _update_ops(self) -> None:
        has = bool(self.checked)
        for button in (
            self.select_button,
            self.delete_button,
            self.export_button,
            self.replace_button,
            self.owner_apply,
            self.usages_button,
        ):
            button.setEnabled(has)

    def remap_checked(self, old_to_new: Mapping[tuple[int, int], tuple[int, int]]) -> None:
        self.checked = {old_to_new.get(key, key) for key in self.checked}
        self.notes = {old_to_new.get(key, key): note for key, note in self.notes.items()}

    # -- operations --------------------------------------------------------------

    def select_on_map(self) -> None:
        keys = [row.key for row in self.checked_rows()]
        if keys:
            self._dialog.callbacks.select_keys(keys)

    def delete_checked(self) -> None:
        rows = self.checked_rows()
        if not rows:
            return
        callbacks = self._dialog.callbacks
        keys = [row.key for row in rows]
        text = f"Delete {len(rows)} objects? Ctrl+Z undoes."
        # Warned, never refused: a dangling reference is a Map Analysis finding, legal on disk.
        refs = callbacks.trigger_refs_for(keys)
        if refs:
            text += f"\n{refs} trigger entries reference these objects and will point at nothing."
        if callbacks.confirm("Delete objects", text) is None:
            return
        callbacks.delete_keys(keys)

    def show_trigger_usages(self) -> None:
        refs = sorted({row.reference_id for row in self.checked_rows()})
        if refs:
            self._dialog.show_usages(refs)

    def replace_checked(self) -> None:
        rows = self.checked_rows()
        if not rows:
            return
        callbacks = self._dialog.callbacks
        new_const = self.replace_edit.value()
        if new_const is None:
            callbacks.warn("Replace", "Choose the object to replace with first.")
            return
        keys = [row.key for row in rows]
        refused = callbacks.replace_preflight(keys, new_const)
        self._mark_refused(refused)
        ok = len(keys) - len(refused)
        if ok == 0:
            callbacks.warn("Replace", f"All {len(keys)} checked objects refuse this replacement; see the Note column.")
            return
        name = object_catalog.display_name(new_const)
        text = f"Replace {ok} objects with {name}? Ctrl+Z undoes."
        if refused:
            text += f"\n{len(refused)} refused, marked in the Note column, are skipped."
        plan, note = callbacks.follow_preview(keys, new_const)
        choices, actions = [], []
        if note:
            text += f"\n\n{note}"
        if plan is not None and plan.entries:
            paired = plan.tier(find_triggers.TIER_PAIRED)
            filter_only = plan.tier(find_triggers.TIER_FILTER_ONLY)
            skipped = plan.tier(find_triggers.TIER_MIXED) + plan.tier(find_triggers.TIER_CLASS)
            report = plan.tier(find_triggers.TIER_REPORT)
            text += "\n\nTrigger effects that filter these objects by type:"
            if paired:
                choices.append(("paired", f"Retarget {len(paired)} paired trigger effects", True))
            if filter_only:
                summary = "; ".join(plan.filter_only_summary())
                choices.append(
                    ("filter_only", f"Also retarget {len(filter_only)} type-filter-only effects ({summary})", False)
                )
            if skipped:
                text += f"\n{len(skipped)} entries can't follow and are skipped."
                actions.append(("Show skipped", lambda e=skipped: self._dialog.show_follow_entries(e)))
            if report:
                text += f"\n{len(report)} other trigger fields name this type."
                actions.append(("Show others", lambda e=report: self._dialog.show_follow_entries(e)))
        answer = callbacks.confirm("Replace objects", text, choices, actions)
        if answer is None:
            return
        follow_writes = []
        if plan is not None and answer.get("paired", False):
            follow_writes = plan.writes(include_filter_only=answer.get("filter_only", False))
        elif plan is not None and answer.get("filter_only", False):
            follow_writes = [
                w for w in plan.writes(include_filter_only=True) if w not in plan.writes(include_filter_only=False)
            ]
        _replaced, refused = callbacks.replace_keys(keys, new_const, follow_writes)
        self._mark_refused(refused)

    def _mark_refused(self, refused: Mapping[tuple[int, int], str]) -> None:
        for key in [k for k in self.notes if k in self.checked]:
            del self.notes[key]
        self.notes.update(refused)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            row = item.data(0, Qt.UserRole)
            if isinstance(row, FindRow):
                note = self.notes.get(row.key, "")
                item.setText(_COL_NOTE, note)
                item.setToolTip(_COL_NOTE, note)

    def change_owner(self) -> None:
        rows = self.checked_rows()
        if not rows:
            return
        player = self.owner_combo.currentData()
        old_to_new = self._dialog.callbacks.reassign_keys([row.key for row in rows], player)
        self.remap_checked(old_to_new)

    def ask_csv_path(self) -> str:
        path, _filter = QFileDialog.getSaveFileName(self, "Export CSV", "find_results.csv", "CSV files (*.csv)")
        return path

    def export_csv(self) -> None:
        rows = self.checked_rows()
        if not rows:
            return
        path = self.ask_csv_path()
        if not path:
            return
        try:
            count = rows_to_csv(rows, Path(path))
        except (WriteBlockedError, OSError) as e:
            self._dialog.callbacks.warn("Export CSV", str(e))
            return
        self.count_label.setText(f"{len(self.rows)} found; exported {count}")


_HIT_ROLE = Qt.UserRole
_SPANS_ROLE = Qt.UserRole + 1
_HIGHLIGHT = QColor(255, 214, 0, 140)
TRIGGER_COLUMNS = ("#", "Trigger / entry", "Field", "Value", "Enabled", "Note")
_TCOL_VALUE = TRIGGER_COLUMNS.index("Value")
_TCOL_NOTE = TRIGGER_COLUMNS.index("Note")
_SCOPE_LABELS = (
    ("names", "Names", True),
    ("descriptions", "Descriptions", True),
    ("messages", "Messages", True),
    ("identifiers", "Identifiers", False),
    ("xs", "XS", False),
)
_READ_ONLY_TIP = "Triggers are read-only for this file"


class _SpanDelegate(QStyledItemDelegate):
    """Paints a hit's Value cell with its matched spans highlighted."""

    def paint(self, painter, option, index) -> None:
        spans = index.data(_SPANS_ROLE)
        if not spans:
            super().paint(painter, option, index)
            return
        text = index.data(Qt.DisplayRole) or ""
        self.initStyleOption(option, index)
        option.text = ""
        option.widget.style().drawControl(QStyle.CE_ItemViewItem, option, painter, option.widget)
        metrics = option.fontMetrics
        rect = option.rect.adjusted(4, 0, -4, 0)
        painter.save()
        painter.setClipRect(option.rect)
        for start, end in spans:
            x0 = rect.left() + metrics.horizontalAdvance(text[:start])
            width = metrics.horizontalAdvance(text[start:end])
            painter.fillRect(x0, rect.top() + 1, width, rect.height() - 2, _HIGHLIGHT)
        painter.drawText(rect, int(Qt.AlignVCenter | Qt.AlignLeft), text)
        painter.restore()


def _one_line(text: str) -> str:
    """Same length as `text` (so spans still index it), newlines shown as ↵."""
    return text.replace("\n", "↵")


def _entry_label(hit: TriggerHit) -> str:
    if hit.kind == "trigger":
        return "Trigger"
    return f"{hit.kind.title()} #{hit.entry_index} {hit.entry_name.replace('_', ' ')}"


class _TriggersTab(QWidget):
    """Text, type and placed-object search over triggers (Step 9)."""

    def __init__(self, dialog: FindDialog):
        super().__init__(dialog)
        self._dialog = dialog
        self.initialized = False
        self.writable = False
        self.type_consts: list[int] = []
        self.last_criteria: TriggerFindCriteria | None = None
        self.hits: list[TriggerHit] = []
        # Checked by location; parents by trigger index.
        self.checked_children: set[tuple] = set()
        self.checked_parents: set[int] = set()
        self.notes: dict[tuple, str] = {}
        self._filling = False
        # What the dialog's own op does to the checks once its re-run lands ("keep" or "clear").
        self.own_op: str | None = None

        self.unsupported = QLabel("")
        self.unsupported.setWordWrap(True)
        self.unsupported.setVisible(False)

        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("Text")
        self.text_edit.returnPressed.connect(self.run_find)
        self.text_edit.textChanged.connect(self._validate)
        self.regex_box = QCheckBox("Regex")
        self.case_box = QCheckBox("Match case")
        for box in (self.regex_box, self.case_box):
            box.toggled.connect(self._validate)
        self.pattern_error = QLabel("")
        self.pattern_error.setStyleSheet(_ERROR_STYLE)
        self.pattern_error.setVisible(False)
        text_row = QHBoxLayout()
        text_row.addWidget(QLabel("Text:"))
        text_row.addWidget(self.text_edit, stretch=1)
        text_row.addWidget(self.regex_box)
        text_row.addWidget(self.case_box)

        self.scope_boxes = {}
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("In:"))
        for key, label, checked in _SCOPE_LABELS:
            box = QCheckBox(label)
            box.setChecked(checked)
            self.scope_boxes[key] = box
            scope_row.addWidget(box)
        scope_row.addStretch(1)

        self.types_button = QPushButton("References type…")
        self.types_button.clicked.connect(self._pick_type)
        self.types_clear = QPushButton("Clear")
        self.types_clear.clicked.connect(self.clear_types)
        self.chips = QHBoxLayout()
        self.chips.setContentsMargins(0, 0, 0, 0)
        types_row = QHBoxLayout()
        types_row.addWidget(self.types_button)
        types_row.addLayout(self.chips)
        types_row.addStretch(1)
        types_row.addWidget(self.types_clear)

        self.refs_edit = QLineEdit()
        self.refs_edit.setPlaceholderText("reference ids, e.g. 500, 501")
        self.refs_edit.returnPressed.connect(self.run_find)
        self.enabled_combo = QComboBox()
        self.enabled_combo.addItem("All", None)
        self.enabled_combo.addItem("Enabled", True)
        self.enabled_combo.addItem("Disabled", False)
        self.find_button = QPushButton("Find")
        self.find_button.setDefault(True)
        self.find_button.clicked.connect(self.run_find)
        refs_row = QHBoxLayout()
        refs_row.addWidget(QLabel("References object:"))
        refs_row.addWidget(self.refs_edit, stretch=1)
        refs_row.addWidget(QLabel("Show:"))
        refs_row.addWidget(self.enabled_combo)
        refs_row.addWidget(self.find_button)

        self.count_label = QLabel("")
        self.check_all = QPushButton("Check all")
        self.check_all.clicked.connect(lambda: self._set_all_checked(True))
        self.check_none = QPushButton("Check none")
        self.check_none.clicked.connect(lambda: self._set_all_checked(False))
        header_row = QHBoxLayout()
        header_row.addWidget(self.count_label)
        header_row.addStretch(1)
        header_row.addWidget(self.check_all)
        header_row.addWidget(self.check_none)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(TRIGGER_COLUMNS))
        self.tree.setUniformRowHeights(True)
        _configure_scrolling(self.tree)
        for col, chars in enumerate((5, 30, 18, 40, 7, 30)):
            self.tree.setColumnWidth(col, self.fontMetrics().averageCharWidth() * chars + 24)
        self.tree.setItemDelegateForColumn(_TCOL_VALUE, _SpanDelegate(self.tree))
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)

        self.replace_text_edit = QLineEdit()
        self.replace_text_edit.setPlaceholderText("replacement (a template in regex mode: \\1)")
        self.replace_text_button = QPushButton("Replace text")
        self.replace_text_button.clicked.connect(self.replace_text)
        self.replace_type_edit = CatalogLineEdit(object_catalog.objects(), default_category="Units", placeholder="(object)")
        self.replace_type_button = QPushButton("Replace type")
        self.replace_type_button.clicked.connect(self.replace_type)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_checked)
        self.enable_button = QPushButton("Enable")
        self.enable_button.clicked.connect(lambda: self.set_enabled(True))
        self.disable_button = QPushButton("Disable")
        self.disable_button.clicked.connect(lambda: self.set_enabled(False))
        self.export_button = QPushButton("Export CSV…")
        self.export_button.clicked.connect(self.export_csv)
        text_ops = QHBoxLayout()
        text_ops.addWidget(QLabel("Replace text with:"))
        text_ops.addWidget(self.replace_text_edit, stretch=1)
        text_ops.addWidget(self.replace_text_button)
        type_ops = QHBoxLayout()
        type_ops.addWidget(QLabel("Replace type with:"))
        type_ops.addWidget(self.replace_type_edit, stretch=1)
        type_ops.addWidget(self.replace_type_button)
        trigger_ops = QHBoxLayout()
        trigger_ops.addWidget(self.delete_button)
        trigger_ops.addWidget(self.enable_button)
        trigger_ops.addWidget(self.disable_button)
        trigger_ops.addStretch(1)
        trigger_ops.addWidget(self.export_button)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.addLayout(text_row)
        body.addWidget(self.pattern_error)
        body.addLayout(scope_row)
        body.addLayout(types_row)
        body.addLayout(refs_row)
        body.addLayout(header_row)
        body.addWidget(self.tree, stretch=1)
        body.addLayout(text_ops)
        body.addLayout(type_ops)
        body.addLayout(trigger_ops)
        layout = QVBoxLayout(self)
        layout.addWidget(self.unsupported)
        layout.addWidget(self.body, stretch=1)
        self._rebuild_chips()
        self._update_ops()

    # -- gating -------------------------------------------------------------------

    def ensure_initialized(self) -> bool:
        """Parses on first show, never when the dialog opens. False if unreadable."""
        if not self.initialized:
            reason, writable = self._dialog.callbacks.trigger_status()
            self.initialized = True
            self.writable = writable
            self.unsupported.setText(reason or "")
            self.unsupported.setVisible(reason is not None)
            self.body.setEnabled(reason is None)
            self._update_ops()
        return not self.unsupported.isVisible()

    # -- search --------------------------------------------------------------------

    def _validate(self) -> bool:
        try:
            compile_pattern(self.text_edit.text(), self.regex_box.isChecked(), self.case_box.isChecked())
        except FindPatternError as e:
            self.pattern_error.setText(str(e))
            self.pattern_error.setVisible(True)
            self.find_button.setEnabled(False)
            self._update_ops()
            return False
        self.pattern_error.setVisible(False)
        self.find_button.setEnabled(True)
        self._update_ops()
        return True

    def add_type_const(self, const: int) -> None:
        if const not in self.type_consts:
            self.type_consts.append(const)
            self._rebuild_chips()

    def clear_types(self) -> None:
        self.type_consts.clear()
        self._rebuild_chips()

    def _remove_type(self, const: int) -> None:
        if const in self.type_consts:
            self.type_consts.remove(const)
        self._rebuild_chips()

    def _rebuild_chips(self) -> None:
        _fill_chips(self.chips, self.type_consts, self._remove_type)
        self.types_clear.setEnabled(bool(self.type_consts))
        self._update_ops()

    def _pick_type(self) -> None:
        dialog = CatalogBrowseDialog(object_catalog.objects(), default_category="Units", parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_id() is not None:
            self.add_type_const(dialog.selected_id())

    def set_object_refs(self, refs: Sequence[int]) -> None:
        self.refs_edit.setText(", ".join(str(r) for r in refs))

    def _object_refs(self) -> frozenset[int] | None:
        parts = [p.strip() for p in self.refs_edit.text().replace(";", ",").split(",") if p.strip()]
        ids = [int(p) for p in parts if p.lstrip("-").isdigit()]
        return frozenset(ids) if ids else None

    def criteria(self) -> TriggerFindCriteria:
        return TriggerFindCriteria(
            text=self.text_edit.text(),
            regex=self.regex_box.isChecked(),
            case_sensitive=self.case_box.isChecked(),
            text_scopes=frozenset(key for key, box in self.scope_boxes.items() if box.isChecked()),
            type_consts=frozenset(self.type_consts) or None,
            object_refs=self._object_refs(),
            enabled=self.enabled_combo.currentData(),
        )

    def run_find(self) -> None:
        if not self.ensure_initialized() or not self._validate():
            return
        self.last_criteria = self.criteria()
        self.notes.clear()
        self.checked_children.clear()
        self.checked_parents.clear()
        self._fill(self.last_criteria, check_all=True)

    def rerun(self, keep_checks: bool) -> None:
        if self.last_criteria is None:
            return
        if not keep_checks:
            self.checked_children.clear()
            self.checked_parents.clear()
        self._fill(self.last_criteria, check_all=False)

    def _fill(self, criteria: TriggerFindCriteria, check_all: bool) -> None:
        try:
            hits = self._dialog.callbacks.find_triggers(criteria)
        except FindPatternError as e:
            self.pattern_error.setText(str(e))
            self.pattern_error.setVisible(True)
            return
        self.hits = hits
        by_trigger: dict[int, list[TriggerHit]] = {}
        for h in hits:
            by_trigger.setdefault(h.trigger_index, []).append(h)
        if check_all:
            self.checked_parents = set(by_trigger)
            self.checked_children = {h.location for h in hits if h.field_class != "trigger"}
        else:
            self.checked_parents &= set(by_trigger)
            self.checked_children &= {h.location for h in hits}
        self._filling = True
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            for index, group in by_trigger.items():
                first = group[0]
                children = [h for h in group if h.field_class != "trigger"]
                parent = QTreeWidgetItem(
                    [str(index), first.trigger_name, f"{len(children)} hits" if children else "", "",
                     "Yes" if first.enabled else "No", self.notes.get((index, "trigger", -1, ""), "")]
                )
                parent.setData(0, _HIT_ROLE, ("trigger", index, first.trigger_name, group))
                parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable)
                parent.setCheckState(0, Qt.Checked if index in self.checked_parents else Qt.Unchecked)
                for h in children:
                    child = QTreeWidgetItem(
                        ["", _entry_label(h), h.attribute, _one_line(h.value_display), "", self.notes.get(h.location, "")]
                    )
                    child.setData(0, _HIT_ROLE, h)
                    child.setData(_TCOL_VALUE, _SPANS_ROLE, list(h.spans))
                    child.setToolTip(_TCOL_VALUE, h.value_display)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Checked if h.location in self.checked_children else Qt.Unchecked)
                    parent.addChild(child)
                self.tree.addTopLevelItem(parent)
                parent.setExpanded(len(children) <= 50)
        finally:
            self.tree.setUpdatesEnabled(True)
            self._filling = False
        self.count_label.setText(f"{len(by_trigger)} triggers, {sum(1 for h in hits if h.field_class != 'trigger')} hits")
        self._update_ops()

    def show_entries(self, entries) -> None:
        """Lists Replace's follow-up entries (find_triggers.FollowEntry) as type hits."""
        if not self.ensure_initialized():
            return
        self.text_edit.clear()
        self.refs_edit.clear()
        self.clear_types()
        for const in sorted({e.old_const for e in entries}):
            self.add_type_const(const)
        self.enabled_combo.setCurrentIndex(0)
        self.run_find()
        wanted = {(e.trigger_index, e.kind, e.entry_index, e.attribute) for e in entries}
        self.checked_children &= wanted
        self.checked_parents &= {loc[0] for loc in wanted}
        self._fill(self.last_criteria, check_all=False)

    # -- checks ---------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._filling or column != 0:
            return
        data = item.data(0, _HIT_ROLE)
        checked = item.checkState(0) == Qt.Checked
        if isinstance(data, TriggerHit):
            (self.checked_children.add if checked else self.checked_children.discard)(data.location)
        elif isinstance(data, tuple):
            (self.checked_parents.add if checked else self.checked_parents.discard)(data[1])
        self._update_ops()

    def _set_all_checked(self, checked: bool) -> None:
        self._filling = True
        try:
            state = Qt.Checked if checked else Qt.Unchecked
            for i in range(self.tree.topLevelItemCount()):
                parent = self.tree.topLevelItem(i)
                parent.setCheckState(0, state)
                for j in range(parent.childCount()):
                    parent.child(j).setCheckState(0, state)
        finally:
            self._filling = False
        if checked:
            self.checked_parents = {h.trigger_index for h in self.hits}
            self.checked_children = {h.location for h in self.hits if h.field_class != "trigger"}
        else:
            self.checked_parents, self.checked_children = set(), set()
        self._update_ops()

    def checked_hits(self, field_classes) -> list[TriggerHit]:
        return [h for h in self.hits if h.location in self.checked_children and h.field_class in field_classes]

    def checked_parent_map(self) -> dict[int, tuple[str, list[TriggerHit]]]:
        parents: dict[int, tuple[str, list[TriggerHit]]] = {}
        for h in self.hits:
            if h.trigger_index in self.checked_parents:
                parents.setdefault(h.trigger_index, (h.trigger_name, []))[1].append(h)
        return parents

    def _update_ops(self) -> None:
        writable = self.initialized and self.writable
        text_hits = bool(self.checked_hits(("text", "xs")))
        has_text = bool(self.text_edit.text()) and not self.pattern_error.isVisible()
        self.replace_text_button.setEnabled(writable and text_hits and has_text)
        self.replace_type_button.setEnabled(writable and bool(self.type_consts) and bool(self.checked_hits(("type",))))
        for button in (self.delete_button, self.enable_button, self.disable_button):
            button.setEnabled(writable and bool(self.checked_parents))
        self.export_button.setEnabled(bool(self.checked_children or self.checked_parents))
        tip = "" if writable or not self.initialized else _READ_ONLY_TIP
        for button in (self.replace_text_button, self.replace_type_button, self.delete_button, self.enable_button,
                       self.disable_button):
            button.setToolTip(tip)

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, _HIT_ROLE)
        if isinstance(data, TriggerHit):
            self._dialog.callbacks.reveal(data.trigger_index, data.kind, data.entry_index)
        elif isinstance(data, tuple):
            self._dialog.callbacks.reveal(data[1], "trigger", -1)

    # -- operations -------------------------------------------------------------------

    def _mark(self, notes: Mapping[tuple, str]) -> None:
        self.notes.update(notes)
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                hit = child.data(0, _HIT_ROLE)
                if isinstance(hit, TriggerHit) and hit.location in self.notes:
                    child.setText(_TCOL_NOTE, self.notes[hit.location])

    def replace_text(self) -> None:
        hits = self.checked_hits(("text", "xs"))
        if not hits or not self.text_edit.text():
            return
        callbacks = self._dialog.callbacks
        triggers = len({h.trigger_index for h in hits})
        if callbacks.confirm("Replace text", f"Replace in {len(hits)} fields across {triggers} triggers? Ctrl+Z undoes.") is None:
            return
        self.own_op = "keep"
        notes = callbacks.text_replace(
            hits, self.text_edit.text(), self.regex_box.isChecked(), self.case_box.isChecked(),
            self.replace_text_edit.text(),
        )
        self._mark(notes)

    def replace_type(self) -> None:
        hits = self.checked_hits(("type",))
        new_const = self.replace_type_edit.value()
        callbacks = self._dialog.callbacks
        if not hits:
            return
        if new_const is None:
            callbacks.warn("Replace type", "Choose the object to replace with first.")
            return
        name = object_catalog.display_name(new_const)
        if callbacks.confirm("Replace type", f"Set {len(hits)} type fields to {name}? Ctrl+Z undoes.") is None:
            return
        self.own_op = "keep"
        self._mark(callbacks.type_replace(hits, frozenset(self.type_consts), new_const))

    def set_enabled(self, value: bool) -> None:
        parents = self.checked_parent_map()
        if parents:
            self.own_op = "keep"
            self._dialog.callbacks.set_enabled(parents, value)

    def delete_checked(self) -> None:
        parents = self.checked_parent_map()
        if not parents:
            return
        callbacks = self._dialog.callbacks
        if callbacks.confirm("Delete triggers", f"Delete {len(parents)} triggers? Ctrl+Z undoes.") is None:
            return
        self.own_op = "clear"
        callbacks.delete_triggers(parents)

    def ask_csv_path(self) -> str:
        path, _filter = QFileDialog.getSaveFileName(self, "Export CSV", "find_triggers.csv", "CSV files (*.csv)")
        return path

    def export_csv(self) -> None:
        hits = [h for h in self.hits if h.location in self.checked_children or (
            h.field_class == "trigger" and h.trigger_index in self.checked_parents)]
        if not hits:
            return
        path = self.ask_csv_path()
        if not path:
            return
        try:
            count = hits_to_csv(hits, Path(path))
        except (WriteBlockedError, OSError) as e:
            self._dialog.callbacks.warn("Export CSV", str(e))
            return
        self.count_label.setText(f"Exported {count} rows")


def _fill_chips(layout: QHBoxLayout, consts: Sequence[int], on_remove: Callable[[int], None]) -> None:
    """The shared Types… chips: one removable button per const."""
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
    for const in consts:
        chip = QToolButton()
        chip.setText(f"{object_catalog.display_name(const)} ✕")
        chip.setToolTip(f"Type {const}: click to remove")
        chip.clicked.connect(lambda _checked=False, c=const: on_remove(c))
        layout.addWidget(chip)


class FindDialog(QDialog):
    """Modeless; `finished` (Close, Escape, title-bar X) is the close hook."""

    def __init__(self, parent=None, callbacks: FindCallbacks | None = None, on_closed=None):
        super().__init__(parent)
        self.setWindowTitle("Find and Replace")
        self.setModal(False)
        self.resize(960, 760)
        self.callbacks = callbacks or FindCallbacks(find_objects=lambda criteria: [])
        self._on_closed = on_closed or (lambda: None)
        self.finished.connect(lambda _result: self._on_closed())
        self._stale: set[str] = set()
        self._rerun_queued = False

        self.tabs = QTabWidget()
        self.objects_tab = _ObjectsTab(self)
        self.triggers_tab = _TriggersTab(self)
        self.tabs.addTab(self.objects_tab, "Objects")
        self.tabs.addTab(self.triggers_tab, "Triggers")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        # Enter in a field must not close a modeless tool window.
        buttons.button(QDialogButtonBox.Close).setAutoDefault(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs, stretch=1)
        layout.addWidget(buttons)

    def refresh_player_labels(self, labels, colors) -> None:
        self.objects_tab.refresh_player_labels(labels, colors)

    def region_changed(self, region) -> None:
        self.objects_tab.set_region_available(region is not None)

    def _on_tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.triggers_tab and not self.triggers_tab.initialized:
            self.triggers_tab.ensure_initialized()
            # Triggers are parsed now, so the Objects tab's Triggers column can fill.
            self.objects_tab.rerun()

    def show_usages(self, refs: Sequence[int]) -> None:
        """Objects tab's Show trigger usages: fill References object, Find, switch."""
        tab = self.triggers_tab
        self.tabs.setCurrentWidget(tab)
        if not tab.ensure_initialized():
            return
        tab.text_edit.clear()
        tab.clear_types()
        tab.set_object_refs(refs)
        tab.run_find()

    def show_follow_entries(self, entries) -> None:
        self.tabs.setCurrentWidget(self.triggers_tab)
        self.triggers_tab.show_entries(entries)

    def mark_stale(self, kind: str = "units") -> None:
        """A unit ("units") or trigger ("triggers") edit, or its undo, happened:
        re-run once, now (coalesced) or on the next show. Both kinds stale both
        tabs: unit edits change placed-object hits, trigger edits the Triggers column."""
        self._stale.add(kind)
        if self.isVisible() and not self._rerun_queued:
            self._rerun_queued = True
            QTimer.singleShot(0, self._rerun_if_stale)

    def _rerun_if_stale(self) -> None:
        self._rerun_queued = False
        if not self._stale or not self.isVisible():
            return
        kinds, self._stale = self._stale, set()
        self.objects_tab.rerun()
        tab = self.triggers_tab
        # Indices renumber on delete and undo, so after any trigger change this
        # dialog didn't make every check clears; its own enable/disable/replace keep theirs.
        keep = "triggers" not in kinds or tab.own_op == "keep"
        tab.own_op = None
        if tab.initialized:
            tab.rerun(keep_checks=keep)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale:
            self._rerun_if_stale()
