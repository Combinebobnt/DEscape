"""TriggerPanel: the trigger browser and property editor, and the
VariablesDialog that hangs off it."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import (
    library_compat,
    messages_fields,
    object_catalog,
    trigger_fields,
    trigger_organize,
    unit_references,
)
from descape.constant_picker import CatalogLineEdit
from descape.scenario_io import (
    LoadedScenario,
    parse_triggers,
    repo_version_has_triggers,
    unsupported_version_sentence,
)
from descape.text_edits import ProseTextEdit, XsTextEdit
from descape.trigger_model import (
    EXEC_MODE_DISPLAY,
    EXEC_MODE_LEGACY,
    EXEC_MODE_UNKNOWN,
    exec_order_value,
    resolve_exec_mode,
)
from descape.value_picker import PickerItem, ValueLineEdit, _HScrollStableTreeWidget
from descape.viewer_common import (
    FontScaledWidth,
    IndeterminateDoubleSpinBox,
    _fit_combo_width,
    _IndeterminateMixin,
    _make_spinbox,
)


class VariablesDialog(QDialog):
    """The document's trigger variables: add, remove, and rename.

    Dumb the same way TriggerPanel is. It never imports TriggerEditModel and
    never touches EditHistory; it reports "add a variable called X", "remove
    the variable with id N", or "rename the variable with id N to Y" through
    the callback it is constructed with, and the window decides what that
    means.

    A dialog rather than a third pane in the panel. Variables belong to the
    document, not to the selected trigger, and 4b.6b already found the panel's
    vertical budget over-subscribed at MIN_USEFUL_WIDTH (144 picker rows in
    five visible ones). A list that is opened rarely should not cost the
    trigger and entry trees their rows permanently.
    """

    _EMPTY = "This scenario has no trigger variables."

    def __init__(self, parent=None, on_structural=None):
        super().__init__(parent)
        self.setWindowTitle("Trigger Variables")
        self.resize(420, 360)
        self._on_structural = on_structural or (lambda *args: None)
        self._editable = False

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["ID", "Name"])
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.tree.itemDoubleClicked.connect(lambda *_: self._request_rename())

        self.status = QLabel(self._EMPTY)
        self.status.setWordWrap(True)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New variable name…")
        self.name_edit.textChanged.connect(lambda *_: self._update_buttons())
        self.name_edit.returnPressed.connect(self._request_add)
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("Add a variable with the next free id")
        self.add_button.clicked.connect(lambda checked=False: self._request_add())
        self.rename_button = QPushButton("Rename")
        self.rename_button.setToolTip("Rename the selected variable (or double-click its row)")
        self.rename_button.clicked.connect(lambda checked=False: self._request_rename())
        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the selected variable")
        self.remove_button.clicked.connect(lambda checked=False: self._request_remove())

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.addWidget(self.name_edit, stretch=1)
        add_row.addWidget(self.add_button)
        add_row.addWidget(self.rename_button)
        add_row.addWidget(self.remove_button)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tree, stretch=1)
        layout.addWidget(self.status)
        layout.addLayout(add_row)
        layout.addWidget(buttons)

        self.set_variables([], editable=False)

    def set_variables(self, variables, editable: bool) -> None:
        """Repopulate from `variables`, a sequence of (variable_id, name).

        Called on open and again after every accepted edit, since the dialog
        holds no model of its own. Selection is restored by variable id rather
        than by row: a removal renumbers no id but does shift every later row.
        """
        selected = self.selected_variable_id()
        self._editable = bool(editable)
        self.tree.clear()
        for variable_id, name in variables:
            item = QTreeWidgetItem([str(variable_id), name or ""])
            item.setData(0, Qt.UserRole, variable_id)
            self.tree.addTopLevelItem(item)
        self.status.setText(self._EMPTY if not variables else "")
        self.status.setVisible(not variables)
        if selected is not None:
            self.select_variable(selected)
        self._update_buttons()

    def select_variable(self, variable_id: int) -> None:
        for row in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(row)
            if item.data(0, Qt.UserRole) == variable_id:
                self.tree.setCurrentItem(item)
                return

    def selected_variable_id(self) -> int | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.UserRole)
        return value if isinstance(value, int) else None

    def _update_buttons(self) -> None:
        name = self.name_edit.text().strip()
        self.name_edit.setEnabled(self._editable)
        self.add_button.setEnabled(self._editable and bool(name))
        has_selection = self.selected_variable_id() is not None
        self.rename_button.setEnabled(self._editable and has_selection)
        self.remove_button.setEnabled(self._editable and has_selection)

    def _request_add(self) -> None:
        name = self.name_edit.text().strip()
        if not self._editable or not name:
            return
        # Cleared before the callback, not after: the window repopulates this
        # dialog from inside _on_structural, and clearing afterwards would
        # race that repopulate's own _update_buttons().
        self.name_edit.clear()
        self._on_structural("add", -1, name)

    def _request_rename(self) -> None:
        variable_id = self.selected_variable_id()
        if not self._editable or variable_id is None:
            return
        current_name = self.tree.currentItem().text(1)
        new_name, ok = QInputDialog.getText(
            self, "Rename Variable", "Name:", QLineEdit.Normal, current_name
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current_name:
            return
        self._on_structural("rename", variable_id, new_name)

    def _request_remove(self) -> None:
        variable_id = self.selected_variable_id()
        if not self._editable or variable_id is None:
            return
        self._on_structural("remove", variable_id, "")


class _TriggerTree(_HScrollStableTreeWidget):
    """The trigger list. With the tree focused, Qt's own Ctrl+A calls
    selectAll() before the window's Select All shortcut sees the key, and
    Qt's version skips collapsed sections, so it routes to the panel's."""

    def __init__(self, select_all) -> None:
        super().__init__()
        self._select_all = select_all

    def selectAll(self) -> None:
        self._select_all()


class TriggerPanel(QWidget):
    """Trigger browser and property editor (phases 4a.3 and 4b.6).

    Dumb by design. It never imports TriggerEditModel and never touches
    EditHistory: it reports "the user changed field X of condition Y of trigger
    Z" through the callbacks it is constructed with, and the window decides what
    that means. The model contract fails silently in every direction, so keeping
    it in one place outside this class is what makes it enforceable.

    Two panes, split vertically. The top is trigger names, grouped into 4c's
    divider-derived sections when the corpus's own `--- Section ---`
    convention is present (Display order only -- see show_scenario()). The
    bottom is the selected trigger's conditions and effects, plus a property
    form for whichever of them is selected.

    Condition and effect names come from library_compat.load_vocabulary(),
    which reads the library's own per-version JSON rather than its module-level
    attribute dicts. Those are rewritten on every load and describe whichever
    file was opened last.
    """

    # Both panes now scroll rather than compete for one row, so the browser is
    # readable at the map-statistics pane's own width. This was 560 in 4a.3,
    # when a name column and a detail column shared a single tree. Scales with
    # the app font (FontScaledWidth).
    MIN_USEFUL_WIDTH = FontScaledWidth(340)
    _PROPERTY_MIN_WIDTH = 240

    # The detail column never gets less than this much of the entry tree's
    # viewport, however long the widest condition name is. Content-sizing the
    # Item column alone would hand a long name the whole pane and push Detail
    # off the right edge, which is what its old fixed width was working around.
    _MIN_DETAIL_WIDTH = 120

    # A field the selected entries disagree on (GH #60). Never a value: the
    # widget showing it writes nothing until the user changes it.
    _DIFFERS = "(differs)"

    _NO_DOCUMENT = "No map open."
    # Why parse_triggers() gave None, keyed by _unsupported_key().
    _UNSUPPORTED: ClassVar[dict[str, str]] = {
        "library": (
            "This file's Triggers section can't be read.\n\n"
            "Scenario version 1.54 with trigger version 3.9 uses an older trigger "
            "format AoE2ScenarioParser can't parse. Open it in the in-game scenario "
            "editor and re-save it to upgrade the format, then reopen it here.\n\n"
            "Terrain and elevation editing work normally on this file."
        ),
        # {sentence} is scenario_io.unsupported_version_sentence(), the same
        # wording the "Failed to load" modal uses for a version with no
        # structure either. Only for a repo version with no vocabulary (v1.21).
        "repo_no_vocabulary": (
            "This file's Triggers section can't be read.\n\n"
            "{sentence} DEscape reads its map and units with its own structure "
            "definition, which does not cover triggers.\n\n"
            "Terrain, elevation and unit editing work normally on this file."
        ),
    }

    @staticmethod
    def _unsupported_key(loaded: LoadedScenario) -> str:
        if loaded.structure_source == "repo" and not repo_version_has_triggers(loaded.scenario_version):
            return "repo_no_vocabulary"
        return "library"

    _READ_ONLY_NOTE = "This file's triggers can be read but not written, so editing is off."
    _PASTE_TOOLTIP = "Paste the copied triggers below the current one (Edit > Copy Triggers copies them)"

    # The only synthetic row in the tree (4c): triggers before the first
    # divider. Never a real trigger -- see current_trigger_index()'s Qt.UserRole
    # sentinel below.
    _BEFORE_FIRST_SECTION = "(before the first section)"

    # A row's parsed [tag] (str | None), stashed at populate time. The tag
    # combo matches against this, never against the name column's text: the
    # rendered label substitutes "(unnamed)" and appends "  (disabled)", and
    # inheriting that false-positive class into a facet would be far more
    # visible than it already is in the free-text filter.
    _TAG_ROLE = Qt.UserRole + 1

    # The trigger tree's columns. Detail, picker and variables trees keep
    # their own indices.
    _COL_POS = 0
    _COL_ID = 1
    _COL_NAME = 2
    # Qt.UserRole and _TAG_ROLE are row data pinned to column 0, whichever
    # column that is; they do not follow the ID column.
    _ROLE_COL = 0

    # detail_stack pages.
    _DETAIL_FORM = 0
    _DETAIL_PICKER = 1
    _DETAIL_MULTI = 2

    # Position header per resolve_exec_mode() mode, 0-based like the ID column
    # and the on-disk trigger_display_order array.
    _POSITION_HEADERS: ClassVar[dict[str, tuple[str, str]]] = {
        EXEC_MODE_DISPLAY: (
            "Exec #",
            "Position in display order, which is also execution order in this file.",
        ),
        EXEC_MODE_LEGACY: (
            "Display #",
            (
                "Position in display order. This file executes in trigger-ID order "
                "(legacy), so the ID column is the execution order."
            ),
        ),
        EXEC_MODE_UNKNOWN: (
            "Display #",
            (
                "Position in display order. This file does not store which order "
                "its triggers execute in."
            ),
        ),
    }

    def __init__(
        self,
        on_trigger_field=None,
        on_entry_field=None,
        on_trigger_structural=None,
        on_entry_structural=None,
        on_variable_structural=None,
        on_selection_changed=None,
        on_tag_rename=None,
        on_tag_remove=None,
    ):
        super().__init__()
        # Callbacks, exactly as MapView takes them. All are no-ops by default
        # so the panel stays constructible on its own for screenshot tests.
        self._on_trigger_field = on_trigger_field or (lambda *args: None)
        self._on_entry_field = on_entry_field or (lambda *args: None)
        self._on_trigger_structural = on_trigger_structural or (lambda *args: None)
        self._on_entry_structural = on_entry_structural or (lambda *args: None)
        self._on_variable_structural = on_variable_structural or (lambda *args: None)
        # Fired when the trigger selection set changes, so the window can
        # re-gate actions that follow it (Edit > Copy Triggers).
        self._on_selection_changed = on_selection_changed or (lambda *args: None)
        self._on_tag_rename = on_tag_rename or (lambda *args: None)
        self._on_tag_remove = on_tag_remove or (lambda *args: None)
        # GH #41: fired after each entry-form sync, which a trigger change reaches too. Assigned by the window.
        self.on_focus_changed = lambda: None
        # Unit/Unit[] fields: placed-unit id -> label text, and Pick from map's
        # arm (a target tuple) or disarm (None) request. Assigned by the window.
        self.describe_unit_reference = lambda ref_id: ""
        self.on_pick_unit = lambda target: None
        self._armed_pick: tuple | None = None
        # Per populate: field -> (spec, refs, editor, refresh) and target -> Pick button.
        self._unit_ref_rows: dict[str, tuple] = {}
        self._pick_buttons: dict[tuple, QToolButton] = {}
        # Whether the window holds a pasteable trigger clipboard. Pushed in
        # through set_clipboard_state(); the panel never reads the window.
        self._clipboard_ready = False

        # Built on first open and then kept, so a variables edit can refresh a
        # dialog that is still showing. Closed and dropped by clear_document().
        self._variables_dialog: VariablesDialog | None = None

        # The trigger the picker was opened against, latched at open rather
        # than read back at accept: the trigger tree stays live while the
        # picker is showing, so a selection change in between would otherwise
        # add the condition to a different trigger than the user asked.
        self._picker_trigger_index: int | None = None
        # The entry being retyped, latched at open for the same reason, plus
        # its current type id so the "picked the type it already has" no-op
        # check needs no lookup at accept time. None means "Add", not "Change".
        self._picker_entry_ref: tuple[str, int] | None = None
        self._picker_entry_type: int = -1

        # True while widgets are being populated programmatically. Qt fires
        # valueChanged/currentIndexChanged on a programmatic set exactly as it
        # does on a user edit, so without this, selecting a row would record one
        # undo step per field and dirty the trigger. See _changed().
        self._populating = False
        self._loaded: LoadedScenario | None = None
        self._vocabulary = None
        self._editable = False
        self._rows: list[tuple] = []
        # The (kind, index) refs the property form was last built for. None
        # means "holds nothing", so the next sync always repopulates.
        self._form_refs: tuple[tuple[str, int], ...] | None = None
        # "display" (trigger_display_order, the in-game order -- default) or
        # "index" (raw list order, the pre-reordering-work behaviour). Pure
        # view state, session-only: sorting is a view concern, decoupled from
        # which order actually executes (that's legacy_exec_order, read-only
        # here and set from Map Options).
        self._sort_mode = "display"
        # Populated every show_scenario(): trigger list index -> its
        # QTreeWidgetItem, the inverse of each item's Qt.UserRole. By item
        # rather than row, unlike the pre-4c dict: a grouped tree's rows are
        # not all top-level, and an item's identity survives move_row()'s
        # take/insert whatever its row number becomes. Used by select_trigger()
        # and _refresh_labels().
        self._item_for_index: dict[int, QTreeWidgetItem] = {}
        # Populated every show_scenario(): the full display-order list (4c
        # grouping only ever reorders which *branch* a row sits under, never
        # this), and its inverse. Used by _update_reorder_buttons() for Move
        # Up/Down bounds, which are about adjacency in display order, not tree
        # geometry -- a grouped tree's adjacent row may be a different
        # section's top-level item.
        self._display_slots: list[int] = []
        self._display_position: dict[int, int] = {}
        # Whether the tree is currently the two-level grouped shape (4c) or
        # flat. Recomputed every show_scenario(); see that method's docstring
        # for when grouping applies.
        self._grouped = False
        # The grouped tree's sections, for Select Section; [] when flat.
        self._sections: list[trigger_organize.Section] = []
        # Display order's sections whatever the sort, and each trigger's
        # section header (None: the leading one; a divider maps to itself).
        # A rename crossing is_divider repopulates, so these stay current.
        self._display_sections: list[trigger_organize.Section] = []
        self._section_of: dict[int, int | None] = {}
        # The last show_scenario()'s pending exec-order value, reused by the
        # panel's own repopulates so they do not drop an unsaved flip.
        self._pending_exec_order: int | None = None
        # Sections the user collapsed, by trigger_organize.section_key(). Kept
        # across same-document rebuilds, dropped per document, never saved.
        self._collapsed_keys: set[tuple[str, int]] = set()
        # True while an active filter has force-expanded sections holding a
        # match, so clearing it knows to put _collapsed_keys back.
        self._filter_expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel(self._NO_DOCUMENT)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.splitter = QSplitter(Qt.Vertical)
        layout.addWidget(self.splitter, stretch=1)

        self.splitter.addWidget(self._build_trigger_pane())
        self.splitter.addWidget(self._build_detail_pane())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        # Below the splitter rather than in either pane's button rows: a
        # variable belongs to the document, not to the selected trigger or
        # entry, and both of those rows are already full at 340 px.
        self.variables_button = QPushButton("Variables…")
        self.variables_button.setToolTip("Add or remove this scenario's trigger variables")
        self.variables_button.clicked.connect(lambda checked=False: self._open_variables())
        layout.addWidget(self.variables_button)

        self.clear_document()

    # -- construction --------------------------------------------------------

    # (mode id, combo label), in display order. "display" is the default: it
    # matches what the in-game editor shows, which raw list order does not on
    # the 6 of 14 parseable corpus files carrying a non-identity
    # trigger_display_order.
    _SORT_MODES = (("display", "Display order"), ("index", "File order (trigger ID)"))

    def _build_trigger_pane(self) -> QWidget:
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter triggers…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.filter_edit.textChanged.connect(lambda *_: self._update_buttons())
        filter_row.addWidget(self.filter_edit, stretch=1)

        # 4c's tag facet: composes with the text filter above rather than
        # replacing it -- both are hide-only, so _apply_filter() gains a
        # second predicate rather than a second mechanism. Repopulated per
        # document in show_scenario().
        self.tag_combo = QComboBox()
        self.tag_combo.addItem("All tags", None)
        self.tag_combo.setEnabled(False)
        self.tag_combo.currentIndexChanged.connect(self._apply_filter)
        self.tag_combo.currentIndexChanged.connect(lambda *_: self._update_buttons())

        # Own row: beside the filter and sort at 340 px, every label elided.
        # One button, Remove behind its arrow. Acts on the facet's tag
        # whatever the text filter hides.
        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(0, 0, 0, 0)
        tag_row.addWidget(self.tag_combo, stretch=1)
        self.tag_rename_button = QToolButton()
        self.tag_rename_button.setText("Rename tag…")
        self.tag_rename_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.tag_rename_button.clicked.connect(lambda checked=False: self.request_tag_rename())
        tag_menu = QMenu(self.tag_rename_button)
        self.tag_remove_action = tag_menu.addAction("Remove tag…")
        self.tag_remove_action.triggered.connect(lambda checked=False: self.request_tag_remove())
        self.tag_rename_button.setMenu(tag_menu)
        tag_row.addWidget(self.tag_rename_button)

        # Collapse All / Expand All, then one jump per section. Shown only on a
        # grouped tree; the section actions are rebuilt per show_scenario().
        self.sections_button = QToolButton()
        self.sections_button.setText("Sections")
        self.sections_button.setToolTip("Collapse or expand every section, or jump to one")
        self.sections_button.setPopupMode(QToolButton.InstantPopup)
        self.sections_menu = QMenu(self.sections_button)
        self.collapse_all_action = self.sections_menu.addAction("Collapse All")
        self.collapse_all_action.triggered.connect(lambda checked=False: self.collapse_all_sections())
        self.expand_all_action = self.sections_menu.addAction("Expand All")
        self.expand_all_action.triggered.connect(lambda checked=False: self.expand_all_sections())
        self.sections_menu.addSeparator()
        self.sections_menu.aboutToShow.connect(self._update_section_actions)
        self._section_actions: list = []
        self.sections_button.setMenu(self.sections_menu)
        self.sections_button.setVisible(False)
        # On the tag row: beside the filter at 340 px it left the text box 45 px.
        tag_row.addWidget(self.sections_button)

        self.sort_combo = QComboBox()
        for mode_id, label in self._SORT_MODES:
            self.sort_combo.addItem(label, mode_id)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_mode_changed)
        filter_row.addWidget(self.sort_combo)
        pane_layout.addLayout(filter_row)
        pane_layout.addLayout(tag_row)

        self.tree = _TriggerTree(self.select_all)
        self.tree.setColumnCount(3)
        self._set_position_header(EXEC_MODE_UNKNOWN)
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(False)
        self._configure_scrolling(self.tree)
        # All content-sized, unlike the detail tree's capped Item/Detail
        # split: position and ID are a few digits wide, so giving Trigger the
        # same ResizeToContents treatment still leaves it effectively the whole
        # pane. No _fit_entry_columns()-style cap and no resizeEvent hook --
        # the sizing here has nothing viewport-dependent to react to.
        for column in (self._COL_POS, self._COL_ID, self._COL_NAME):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        # Multi-select (GH #27). Qt's own Ctrl/Shift gestures; clicking a
        # section header selects that one divider trigger only.
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Both fire on one click. The current item drives the entry tree; the
        # selection set drives only the detail page and the buttons.
        self.tree.currentItemChanged.connect(lambda *_: self._on_trigger_selected())
        self.tree.itemSelectionChanged.connect(self._on_trigger_selection_changed)
        self.tree.itemCollapsed.connect(lambda item: self._on_section_toggled(item, collapsed=True))
        self.tree.itemExpanded.connect(lambda item: self._on_section_toggled(item, collapsed=False))
        # Select Section / Select Tag: pure selection expanders, no model state.
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_trigger_context_menu)
        pane_layout.addWidget(self.tree, stretch=1)

        self.trigger_buttons = self._build_button_row(
            [
                ("trigger_new_button", "New", "Add a new trigger at the end of the list"),
                ("trigger_copy_button", "Copy", "Duplicate the selected triggers in place"),
                ("trigger_paste_button", "Paste", self._PASTE_TOOLTIP),
                ("trigger_delete_button", "Delete", "Remove the selected triggers"),
            ],
            lambda op: (lambda checked=False: self._request_trigger_op(op)),
            ("new", "copy", "paste", "delete"),
        )
        pane_layout.addLayout(self.trigger_buttons)

        # A second row rather than widening the first: the pane is ~340 px
        # and three buttons already fill it (the reason Conditions/Effects
        # share one New button in the entry row below). Move Up/Down are
        # display-order-only (ids are never renumbered, see
        # trigger_structural_edit()'s "move" op), so they are gated on the
        # sort mode too, not just on a trigger being selected.
        self.reorder_buttons = self._build_button_row(
            [
                ("trigger_move_up_button", "▲ Move Up", "Move the selected trigger up in display order"),
                ("trigger_move_down_button", "▼ Move Down", "Move the selected trigger down in display order"),
            ],
            lambda op: (lambda checked=False: self._request_trigger_op(op)),
            ("move_up", "move_down"),
        )
        pane_layout.addLayout(self.reorder_buttons)
        pane_layout.addLayout(self._build_section_row())
        return pane

    def _build_section_row(self) -> QHBoxLayout:
        """New Section (Rename Section behind its arrow) and Move to Section.
        A third row: beside Move Up/Down the four need ~490 px of a 340 px pane."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.section_new_button = QToolButton()
        self.section_new_button.setText("New Section…")
        self.section_new_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.section_new_button.clicked.connect(lambda checked=False: self.request_new_section())
        new_menu = QMenu(self.section_new_button)
        self.section_rename_action = new_menu.addAction("Rename Section…")
        self.section_rename_action.triggered.connect(lambda checked=False: self.request_section_rename())
        self.section_new_button.setMenu(new_menu)
        self.section_move_button = QToolButton()
        self.section_move_button.setText("Move to Section")
        self.section_move_button.setPopupMode(QToolButton.InstantPopup)
        self.section_move_menu = QMenu(self.section_move_button)
        self.section_move_menu.aboutToShow.connect(self._populate_move_menu)
        self.section_move_button.setMenu(self.section_move_menu)
        for button in (self.section_new_button, self.section_move_button):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(button)
        return row

    def _on_sort_mode_changed(self, combo_index: int) -> None:
        if self._populating:
            return
        mode = self.sort_combo.itemData(combo_index)
        if mode == self._sort_mode:
            return
        selected = self.current_trigger_index()
        self._sort_mode = mode
        self.show_scenario(self._loaded, self._pending_exec_order)
        if selected is not None:
            self.select_trigger(selected)

    def _build_detail_pane(self) -> QWidget:
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        self.entry_tree = _HScrollStableTreeWidget()
        self.entry_tree.setHeaderLabels(["Item", "Detail"])
        self.entry_tree.setColumnCount(2)
        self.entry_tree.setUniformRowHeights(True)
        self._configure_scrolling(self.entry_tree)
        # Multi-select (GH #60), the entry-level twin of the trigger tree's.
        self.entry_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # One click fires both; _sync_entry_form()'s latch makes it one populate.
        self.entry_tree.currentItemChanged.connect(lambda *_: self._on_entry_selected())
        self.entry_tree.itemSelectionChanged.connect(self._on_entry_selection_changed)
        self.entry_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.entry_tree.customContextMenuRequested.connect(self._show_entry_context_menu)
        pane_layout.addWidget(self.entry_tree, stretch=1)

        self.entry_buttons = self._build_button_row(
            [
                ("entry_new_button", "New", "Add a condition or effect to this trigger"),
                ("entry_copy_button", "Copy", "Duplicate the selected conditions and effects"),
                ("entry_delete_button", "Delete", "Remove the selected conditions and effects"),
                # "Type…" rather than "Change Type": 4b.6b measured that three
                # buttons fit the 340 px pane and five do not, so the fourth
                # has to be short. The tooltip carries the rest.
                (
                    "entry_retype_button",
                    "Type…",
                    "Change the selected condition or effect to a different type",
                ),
            ],
            lambda op: (lambda checked=False: self._request_entry_op(op)),
            ("new", "copy", "delete", "retype"),
        )
        pane_layout.addLayout(self.entry_buttons)

        # The property form and the vocabulary picker are two pages of one
        # stack rather than two stacked widgets: the picker is a 140-row tree
        # and a permanent home for it would halve the form's height on every
        # document, including the ones nobody ever adds a condition to.
        self.detail_stack = QStackedWidget()
        self.detail_stack.addWidget(self._build_property_page())  # _DETAIL_FORM
        self.detail_stack.addWidget(self._build_picker_page())  # _DETAIL_PICKER
        self.multi_label = QLabel("")
        self.multi_label.setAlignment(Qt.AlignCenter)
        self.detail_stack.addWidget(self.multi_label)  # _DETAIL_MULTI
        pane_layout.addWidget(self.detail_stack, stretch=1)
        return pane

    def _build_button_row(self, buttons, slot_for, ops) -> QHBoxLayout:
        """One row of equally-weighted push buttons, stored on self by name."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for (attribute, label, tip), op in zip(buttons, ops, strict=True):
            button = QPushButton(label)
            button.setToolTip(tip)
            # checked=False first: QPushButton.clicked passes a positional bool,
            # the same trap the mode actions document a few hundred lines down.
            button.clicked.connect(slot_for(op))
            setattr(self, attribute, button)
            row.addWidget(button)
        return row

    def _build_property_page(self) -> QWidget:
        self.property_area = QScrollArea()
        self.property_area.setWidgetResizable(True)
        self.property_area.setMinimumWidth(self._PROPERTY_MIN_WIDTH)
        self.property_host = QWidget()
        self.property_form = QFormLayout(self.property_host)
        self.property_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # Long-labelled rows put their editor on the line below instead of
        # beside a label column sized by "armour attack quantity". Measured at
        # 340 px: editors go from 128 px to 191-306 px, and the form's minimum
        # width drops from 413 to 227, so it stops being squeezed below its own
        # minimum. Short rows are unaffected and stay on one line.
        #
        # This only works together with the combo constraint in _build_widget():
        # with a combo demanding 430 px, every wrap policy pushed the form into
        # horizontal overflow instead, which is the opposite of the intent.
        self.property_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.property_area.setWidget(self.property_host)
        return self.property_area

    def _build_picker_page(self) -> QWidget:
        """4a.4's vocabulary picker: one filtered tree over both kinds.

        Conditions and effects share a single picker, and the group a row sits
        under is what decides which kind gets added. One picker rather than a
        New Condition and a New Effect button because three buttons fit the
        340 px pane and five do not.
        """
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        self.picker_filter = QLineEdit()
        self.picker_filter.setPlaceholderText("Filter conditions and effects…")
        self.picker_filter.setClearButtonEnabled(True)
        self.picker_filter.textChanged.connect(self._apply_picker_filter)
        page_layout.addWidget(self.picker_filter)

        self.picker_tree = _HScrollStableTreeWidget()
        self.picker_tree.setHeaderLabels(["Add to this trigger"])
        self.picker_tree.setColumnCount(1)
        self.picker_tree.setUniformRowHeights(True)
        self._configure_scrolling(self.picker_tree)
        # Content-sized rather than a fixed width. Safe here where it is not in
        # the two-column detail tree: ResizeToContents sizes to the single
        # longest row, and with only one column there is no second one for it to
        # push off the pane. Without it the column keeps Qt's ~100 px default and
        # clips every name to "accumu", "ai signa", "bring o".
        self.picker_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.picker_tree.itemDoubleClicked.connect(lambda *_: self._accept_pick())
        self.picker_tree.currentItemChanged.connect(lambda *_: self._update_buttons())
        page_layout.addWidget(self.picker_tree, stretch=1)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        self.picker_add_button = QPushButton("Add")
        self.picker_add_button.clicked.connect(lambda checked=False: self._accept_pick())
        row.addWidget(self.picker_add_button)
        self.picker_cancel_button = QPushButton("Cancel")
        self.picker_cancel_button.clicked.connect(lambda checked=False: self._close_picker())
        row.addWidget(self.picker_cancel_button)
        page_layout.addLayout(row)
        return page

    def _fit_entry_columns(self) -> None:
        """Size the detail tree's two columns to their content, capping Item.

        Neither column may be left at a fixed width, and for different reasons.
        Item was pinned at `_PROPERTY_MIN_WIDTH` (240) out of a 340 px pane, so
        Detail got 98 px and read "quantity=19...", "timer=1, inv...". Detail
        was left at Qt's 100 px default, which is worse than it looks: with
        `setStretchLastSection(False)` the column really is 100 px wide, so the
        rest of a 1263 px detail string was **unreachable at any scroll
        position** rather than merely off-screen.

        So Item is content-sized but capped to leave `_MIN_DETAIL_WIDTH`
        visible, and Detail is content-sized outright, which is what lets the
        horizontal scrollbar reach all of it. Called on every populate and on
        every resize, since the cap depends on the current viewport.
        """
        viewport = self.entry_tree.viewport().width()
        item_width = self.entry_tree.sizeHintForColumn(0)
        if viewport > 0:
            # max() so a very narrow pane still leaves Item something to show,
            # rather than collapsing it and relying on the scrollbar for both.
            item_width = min(item_width, max(80, viewport - self._MIN_DETAIL_WIDTH))
        self.entry_tree.setColumnWidth(0, item_width)
        self.entry_tree.setColumnWidth(1, self.entry_tree.sizeHintForColumn(1))

    def resizeEvent(self, event) -> None:
        """Re-fit the detail columns, whose cap is a function of the pane width.

        Without this, dragging the splitter narrower leaves Item at a width the
        pane can no longer afford, and dragging it wider leaves Item smaller
        than it could be. Both are only cosmetic -- the scrollbar still reaches
        everything -- but the cap is meaningless if it is only ever computed at
        the width the panel happened to be built at.
        """
        super().resizeEvent(event)
        if self._populating:
            return
        if self.entry_tree.topLevelItemCount():
            self._fit_entry_columns()
        # A narrower pane wraps more rows, so the form's height is a function
        # of the width too.
        self._fit_property_height()

    @staticmethod
    def _configure_scrolling(tree: QTreeWidget) -> None:
        """Make a tree scroll horizontally instead of eliding.

        All three calls are load-bearing together. With stretchLastSection on,
        a QTreeWidget never scrolls horizontally: it squeezes the columns and
        elides, which is the "s..."/"e..." defect 4a.3 hit, and which comes back
        as "there is a scrollbar but the text is still cut" if only the policy
        is set.
        """
        tree.header().setStretchLastSection(False)
        tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tree.setTextElideMode(Qt.ElideNone)

    # -- document state ------------------------------------------------------

    def _reset_views(self) -> None:
        """Empty both trees, the property form and the picker. Callers hold
        the _populating guard, since clearing a tree emits selection changes."""
        self.tree.clear()
        self.entry_tree.clear()
        self._clear_property_form()
        self._close_picker()

    def clear_document(self) -> None:
        self._populating = True
        try:
            self._reset_views()
            self.filter_edit.clear()
            self.filter_edit.setEnabled(False)
            self.tag_combo.clear()
            self.tag_combo.addItem("All tags", None)
            self.tag_combo.setEnabled(False)
            self._loaded = None
            self._vocabulary = None
            self._editable = False
            self._item_for_index = {}
            self._display_slots = []
            self._display_position = {}
            self._grouped = False
            self._sections = []
            self._display_sections = []
            self._section_of = {}
            self._collapsed_keys = set()
            self._filter_expanded = False
            self._populate_sections_menu([])
            self._pending_exec_order = None
            # Neutral, so an empty panel stops asserting the last file's mode.
            self._set_position_header(EXEC_MODE_UNKNOWN)
            self.status.setText(self._NO_DOCUMENT)
            # Closed, not just emptied: its contents belong to a document that
            # is no longer open, and leaving it up invites an edit against one.
            if self._variables_dialog is not None:
                self._variables_dialog.close()
                self._variables_dialog.deleteLater()
                self._variables_dialog = None
        finally:
            self._populating = False
        self._update_buttons()

    def show_scenario(
        self, loaded: LoadedScenario | None, pending_exec_order: int | None = None
    ) -> None:
        """Populate from `loaded`, parsing its Triggers section on first use.

        Calls parse_triggers() every time rather than caching a manager: the
        library's field gating is class-level and process-global, so a manager
        obtained before another file was opened is only safe to read after a
        re-parse. The call is memoized, so this costs a depoison(), not a
        re-parse. See scenario_io.parse_triggers().

        `pending_exec_order` is the caller-resolved value of an in-progress
        Map Options edit, not yet written to any retriever -- see
        _exec_order_readout().
        """
        same_document = loaded is not None and loaded is self._loaded
        keep = self._selection_state() if same_document else None
        self._pending_exec_order = pending_exec_order
        if not same_document:
            self._collapsed_keys = set()

        self._populating = True
        try:
            self._reset_views()
            if loaded is None:
                self._populating = False
                self.clear_document()
                return

            manager = parse_triggers(loaded)
            if manager is None:
                self._loaded = None
                self._editable = False
                self.filter_edit.setEnabled(False)
                self.tag_combo.clear()
                self.tag_combo.addItem("All tags", None)
                self.tag_combo.setEnabled(False)
                self._grouped = False
                self._sections = []
                self._display_sections = []
                self._section_of = {}
                self._populate_sections_menu([])
                self._set_position_header(EXEC_MODE_UNKNOWN)
                self.status.setText(
                    self._UNSUPPORTED[self._unsupported_key(loaded)].format(
                        sentence=unsupported_version_sentence(loaded.scenario_version)
                    )
                )
                # This return leaves via the finally below, skipping the tail,
                # so the buttons have to be settled here as well.
                self._update_buttons()
                self.refresh_variables()
                return

            self._loaded = loaded
            # Gated at populate time, never at file-open time: the flag is only
            # meaningful once parse_triggers() has actually run.
            self._editable = bool(loaded.trigger_write_supported)
            self.filter_edit.setEnabled(True)
            self._vocabulary = None
            if library_compat.vocabulary_is_available(loaded.scenario_version):
                self._vocabulary = library_compat.load_vocabulary(loaded.scenario_version)

            triggers = manager.triggers
            note = "" if self._editable else f". {self._READ_ONLY_NOTE}"
            self.status.setText(
                f"{len(triggers)} trigger{'s' if len(triggers) != 1 else ''} — "
                f"scenario {loaded.scenario_version}, trigger format "
                f"{loaded.trigger_version:g}{note}. "
                f"{self._exec_order_readout(loaded, pending_exec_order)}"
            )
            self._set_position_header(
                resolve_exec_mode(exec_order_value(loaded), pending_exec_order)[0]
            )
            # Always display order, whatever the sort: the position column and
            # Move Up/Down bounds both mean display position, not view row.
            self._display_slots = list(manager.trigger_display_order)
            self._display_position = {
                index: position for position, index in enumerate(self._display_slots)
            }
            # Display order by default -- the in-game order, and what raw
            # list order already disagrees with on 6 of 14 parseable corpus
            # files. "index" (raw list order) is the alternate sort a user can
            # pick, e.g. to match a file's trigger ids for scripting purposes.
            order = (
                list(self._display_slots)
                if self._sort_mode == "display"
                else list(range(len(triggers)))
            )

            names = [self._read(t, "name") or "" for t in triggers]
            self._populate_tag_combo(names)
            self._display_sections = trigger_organize.sections(names, self._display_slots)
            self._section_of = {}
            for section in self._display_sections:
                for index in section.member_indices:
                    self._section_of[index] = section.header_index
                if section.header_index is not None:
                    self._section_of[section.header_index] = section.header_index

            # Grouping (4c) exists only under Display order -- section
            # membership is order-derived, so under File order there are no
            # sections and the tree stays flat, whatever the names look like.
            if self._sort_mode == "display":
                parts = trigger_organize.sections(names, order)
            else:
                parts = [trigger_organize.Section("", None, tuple(order))]
            self._grouped = not (len(parts) == 1 and parts[0].header_index is None)
            self._sections = parts if self._grouped else []
            self._populate_sections_menu(self._sections)
            # False for the flat tree (its default -- no children ever exist,
            # so a branch indicator column would just be wasted indent). True
            # here only lets the user actually collapse a section.
            self.tree.setRootIsDecorated(self._grouped)

            self._item_for_index = {}
            if self._grouped:
                self._populate_grouped(triggers, parts)
            else:
                self._populate_flat(triggers, order)
            # No setColumnWidth here: the column is content-sized in
            # _build_trigger_pane(). Forcing it to MIN_USEFUL_WIDTH gave a list
            # of short names a horizontal scrollbar it did not need -- 340 px of
            # column for 246 px of longest name, on the shipped fixture.
            self._apply_filter()
        finally:
            self._populating = False

        if keep is not None:
            self._restore_selection(keep)
        else:
            self._select_default_trigger()
        self._update_buttons()
        # An undo can restore the variable list without touching a trigger, and
        # it refreshes through here, so this is not only for a document open.
        self.refresh_variables()

    def _populate_tag_combo(self, names: list[str]) -> None:
        """Repopulate the tag facet from `names`' distinct [tag] prefixes,
        preserving the current selection across a same-document rebuild (a
        sort-mode toggle, a structural edit's full repopulate) the same way
        the filter text field already does by simply never being cleared."""
        previous = self.tag_combo.currentData() if self.tag_combo.count() else None
        self.tag_combo.blockSignals(True)
        try:
            self.tag_combo.clear()
            self.tag_combo.addItem("All tags", None)
            tags = sorted({tag for name in names if (tag := trigger_organize.parse_tag(name)) is not None})
            for tag in tags:
                self.tag_combo.addItem(tag, tag)
            self.tag_combo.setEnabled(True)
            restored = self.tag_combo.findData(previous)
            self.tag_combo.setCurrentIndex(restored if restored >= 0 else 0)
        finally:
            self.tag_combo.blockSignals(False)

    _TAG_RENAME_TIP = (
        "Rename the chosen tag on every trigger carrying it, including ones the "
        "text filter hides. The arrow offers Remove tag."
    )
    _TAG_PICK_TIP = "Pick a tag to rename"

    def _tag_count(self, tag: str) -> int:
        """How many triggers carry `tag`, read off raw names."""
        manager = self._manager()
        if manager is None:
            return 0
        return sum(1 for t in manager.triggers if trigger_organize.parse_tag(self._read(t, "name") or "") == tag)

    @staticmethod
    def _triggers_text(count: int) -> str:
        return f"{count} trigger{'s' if count != 1 else ''}"

    def request_tag_rename(self) -> None:
        """Ask for a new name for the facet's tag, confirm a merge onto an
        existing tag, then report through on_tag_rename."""
        old = self.tag_combo.currentData()
        if old is None or not self._editable:
            return
        count = self._tag_count(old)
        text, ok = QInputDialog.getText(
            self,
            f'Rename tag "{old}"',
            f"{self._triggers_text(count)} carry this tag. New tag:",
            QLineEdit.Normal,
            old,
        )
        new = text.strip()
        if not ok or not new or new == old:
            return
        if self.tag_combo.findData(new) >= 0:
            answer = QMessageBox.question(
                self,
                "Merge tags",
                f'Merge tag "{old}" ({self._triggers_text(count)}) into existing tag '
                f'"{new}" ({self._triggers_text(self._tag_count(new))})? One undo reverses it.',
            )
            if answer != QMessageBox.Yes:
                return
        self._on_tag_rename(old, new)

    def request_tag_remove(self) -> None:
        """Confirm, then report stripping the facet's tag through on_tag_remove."""
        tag = self.tag_combo.currentData()
        if tag is None or not self._editable:
            return
        answer = QMessageBox.question(
            self,
            "Remove tag",
            f'Remove tag "{tag}" from {self._triggers_text(self._tag_count(tag))}? One undo reverses it.',
        )
        if answer == QMessageBox.Yes:
            self._on_tag_remove(tag)

    def current_tag(self) -> str | None:
        """The tag facet's selected tag, or None on "All tags"."""
        return self.tag_combo.currentData()

    def set_tag_filter(self, tag: str | None) -> None:
        """Point the tag facet at `tag`; a tag the document lacks is a no-op."""
        found = self.tag_combo.findData(tag)
        if found >= 0:
            self.tag_combo.setCurrentIndex(found)

    def _set_position_header(self, mode: str) -> None:
        text, tooltip = self._POSITION_HEADERS[mode]
        labels = ["", "ID", "Trigger"]
        labels[self._COL_POS] = text
        self.tree.setHeaderLabels(labels)
        self.tree.headerItem().setToolTip(self._COL_POS, tooltip)

    def _position_text(self, index: int) -> str:
        position = self._display_position.get(index)
        return "" if position is None else str(position)

    def _make_trigger_item(self, index: int, trigger) -> QTreeWidgetItem:
        columns = ["", "", ""]
        columns[self._COL_POS] = self._position_text(index)
        columns[self._COL_ID] = str(index)
        columns[self._COL_NAME] = self._trigger_label(trigger)
        item = QTreeWidgetItem(columns)
        item.setData(self._ROLE_COL, Qt.UserRole, index)
        item.setData(
            self._ROLE_COL, self._TAG_ROLE, trigger_organize.parse_tag(self._read(trigger, "name") or "")
        )
        return item

    def _populate_flat(self, triggers, order: list[int]) -> None:
        for index in order:
            item = self._make_trigger_item(index, triggers[index])
            self.tree.addTopLevelItem(item)
            self._item_for_index[index] = item

    def _populate_grouped(self, triggers, parts: list[trigger_organize.Section]) -> None:
        """The two-level tree (4c): one top-level row per section, whose own
        row IS the divider trigger -- never a synthetic row, see
        trigger_organize.Section's docstring -- except the single leading
        section when there are triggers before the first divider, which gets
        the one synthetic row in the whole tree."""
        for position, section in enumerate(parts):
            if section.header_index is None:
                columns = ["", "", ""]
                columns[self._COL_NAME] = self._BEFORE_FIRST_SECTION
                top = QTreeWidgetItem(columns)
                top.setData(self._ROLE_COL, Qt.UserRole, None)
                top.setData(self._ROLE_COL, self._TAG_ROLE, None)
                # Not a real trigger: selecting it must be impossible, not just
                # handled gracefully if it somehow gets selected.
                top.setFlags(top.flags() & ~Qt.ItemIsSelectable)
            else:
                top = self._make_trigger_item(section.header_index, triggers[section.header_index])
                self._item_for_index[section.header_index] = top
            for member_index in section.member_indices:
                child = self._make_trigger_item(member_index, triggers[member_index])
                top.addChild(child)
                self._item_for_index[member_index] = child
            self.tree.addTopLevelItem(top)
            top.setExpanded(trigger_organize.section_key(parts, position) not in self._collapsed_keys)

    # -- section navigation --------------------------------------------------

    # Menu labels are elided at this many average characters: the corpus's
    # p90 divider title is 38 long, its longest 133.
    _SECTION_LABEL_CHARS = 40

    def _populate_sections_menu(self, parts: list[trigger_organize.Section]) -> None:
        """One jump action per section after the fixed Collapse/Expand All,
        and the button shown only when there are sections at all."""
        for action in self._section_actions:
            self.sections_menu.removeAction(action)
            action.deleteLater()
        self._section_actions = []
        metrics = self.sections_menu.fontMetrics()
        width = metrics.averageCharWidth() * self._SECTION_LABEL_CHARS
        for section in parts:
            if section.header_index is None:
                label = self._BEFORE_FIRST_SECTION
                # The synthetic row is not selectable; jump to its first member,
                # the same row _select_default_trigger() lands on.
                target = section.member_indices[0]
            else:
                label = trigger_organize.section_label(section.title)
                target = section.header_index
            text = metrics.elidedText(label, Qt.ElideRight, width).replace("&", "&&")
            action = self.sections_menu.addAction(text)
            action.setData(target)
            action.triggered.connect(lambda checked=False, i=target: self.reveal_trigger(i))
            self._section_actions.append(action)
        self.sections_button.setVisible(bool(parts))

    def _update_section_actions(self) -> None:
        """Disable jumps to a row the filter currently hides."""
        for action in self._section_actions:
            item = self._item_for_index.get(action.data())
            action.setEnabled(item is not None and not item.isHidden())

    def _section_position(self, item) -> int | None:
        """`item`'s index in _sections when it is a grouped top-level row."""
        if not self._grouped or item is None or item.parent() is not None:
            return None
        position = self.tree.indexOfTopLevelItem(item)
        return position if 0 <= position < len(self._sections) else None

    def _on_section_toggled(self, item, collapsed: bool) -> None:
        """Remember a user's collapse or expand of one section. Populate and
        filter passes expand rows as a side effect, so they are ignored."""
        if self._populating or self._filter_expanded:
            return
        position = self._section_position(item)
        if position is None:
            return
        key = trigger_organize.section_key(self._sections, position)
        if collapsed:
            self._collapsed_keys.add(key)
        else:
            self._collapsed_keys.discard(key)

    def _sync_collapsed_keys_from_tree(self) -> None:
        self._collapsed_keys = {
            trigger_organize.section_key(self._sections, position)
            for position in range(min(len(self._sections), self.tree.topLevelItemCount()))
            if not self.tree.topLevelItem(position).isExpanded()
        }

    def collapse_all_sections(self) -> None:
        self._filter_expanded = False
        self.tree.collapseAll()
        self._sync_collapsed_keys_from_tree()
        # Re-expands what an active filter matches, so its hits stay visible.
        self._apply_filter()

    def expand_all_sections(self) -> None:
        self._filter_expanded = False
        self.tree.expandAll()
        self._sync_collapsed_keys_from_tree()
        self._apply_filter()

    def reveal_trigger(self, index: int) -> None:
        """Jump to the trigger at list `index`: expand its section, make it the
        current row and scroll it to the top of the view."""
        item = self._item_for_index.get(index)
        if item is None:
            return
        top = item.parent() or item
        position = self._section_position(top)
        if position is not None:
            top.setExpanded(True)
            # Explicit: under a filter the toggle handler is suppressed.
            self._collapsed_keys.discard(trigger_organize.section_key(self._sections, position))
        self._set_current_trigger_item(item)
        self.tree.scrollToItem(item, QAbstractItemView.PositionAtTop)

    def _retitle_section(self, trigger_index: int, title: str) -> None:
        """Follow a divider renamed to another divider: the menu label and the
        collapse keys, which a label patch alone would leave stale."""
        position = next(
            (p for p, s in enumerate(self._sections) if s.header_index == trigger_index), None
        )
        if position is None or self._sections[position].title == title:
            return
        collapsed = [
            p for p in range(len(self._sections))
            if trigger_organize.section_key(self._sections, p) in self._collapsed_keys
        ]
        self._sections[position] = dataclasses.replace(self._sections[position], title=title)
        self._display_sections = [
            dataclasses.replace(s, title=title) if s.header_index == trigger_index else s
            for s in self._display_sections
        ]
        self._collapsed_keys = {trigger_organize.section_key(self._sections, p) for p in collapsed}
        self._populate_sections_menu(self._sections)

    # -- section management --------------------------------------------------

    _NEW_SECTION_TIP = "Add a section header at the end of the current trigger's section"
    _MOVE_TO_SECTION_TIP = "Move the selected triggers to the end of another section"

    def _section_view_reason(self) -> str:
        """Why the section verbs are off for view reasons, or "": the same
        display-order, unfiltered precondition as Move Up/Down."""
        if self._sort_mode != "display":
            return " (switch to Display order to edit sections)"
        if self.filter_edit.text().strip() or self.tag_combo.currentData() is not None:
            return " (clear the filter or tag to edit sections)"
        return ""

    def _current_name(self) -> str | None:
        index = self.current_trigger_index()
        manager = self._manager()
        if index is None or manager is None or index >= len(manager.triggers):
            return None
        return self._read(manager.triggers[index], "name") or ""

    def _move_targets(self) -> list[tuple[str, int | None, bool]]:
        """(label, header index, enabled) per section the selection can move
        to, in display order. The leading section is offered even when empty,
        which sections() omits. A section already holding every selected
        trigger is disabled."""
        parts = list(self._display_sections)
        if parts and parts[0].header_index is not None:
            parts.insert(0, trigger_organize.Section("", None, ()))
        homes = {self._section_of.get(index) for index in self.selected_trigger_indices()}
        targets = []
        for section in parts:
            header = section.header_index
            label = (
                self._BEFORE_FIRST_SECTION
                if header is None
                else trigger_organize.section_label(section.title)
            )
            targets.append((label, header, homes != {header}))
        return targets

    def _move_blocker(self) -> str:
        """Why Move to Section is off for this selection, or ""."""
        indices = self.selected_trigger_indices()
        if not indices:
            return " (select a trigger first)"
        if any(self._section_of.get(index, -1) == index for index in indices):
            return " (a section header cannot move into another section)"
        if not any(enabled for _, _, enabled in self._move_targets()):
            return " (this file has no other section)"
        return ""

    def _populate_move_menu(self) -> None:
        self.section_move_menu.clear()
        metrics = self.section_move_menu.fontMetrics()
        width = metrics.averageCharWidth() * self._SECTION_LABEL_CHARS
        for label, header, enabled in self._move_targets():
            text = metrics.elidedText(label, Qt.ElideRight, width).replace("&", "&&")
            action = self.section_move_menu.addAction(text)
            # The header index, never the title: two sections can share one.
            action.setData(header)
            action.setEnabled(enabled)
            action.triggered.connect(lambda checked=False, h=header: self.request_move_to_section(h))

    def request_new_section(self) -> None:
        """Ask for a title, then report a new section after the current one."""
        if not self._editable or self._section_view_reason():
            return
        title, ok = QInputDialog.getText(self, "New Section", "Section title:", QLineEdit.Normal, "")
        title = title.strip()
        if not ok or not title:
            return
        current = self.current_trigger_index()
        self._on_trigger_structural("new_section", [] if current is None else [current], title)

    def request_section_rename(self) -> None:
        """Ask for a new title for the current divider and write it through
        the trigger name funnel, keeping the divider's own decoration."""
        index = self.current_trigger_index()
        name = self._current_name()
        if not self._editable or self._section_view_reason() or index is None or name is None:
            return
        parts = trigger_organize.split_divider(name)
        if parts is None:
            return
        text, ok = QInputDialog.getText(
            self, "Rename Section", "Section title:", QLineEdit.Normal, parts[1]
        )
        title = text.strip()
        if not ok or not title or title == parts[1]:
            return
        self.rename_section(index, title)

    def rename_section(self, index: int, title: str) -> None:
        """Retitle the divider at `index`. A title that would not read back
        (one ending in a divider character) is refused before any edit."""
        manager = self._manager()
        if manager is None or not 0 <= index < len(manager.triggers):
            return
        name = self._read(manager.triggers[index], "name") or ""
        if trigger_organize.split_divider(name) is None:
            return
        value = trigger_organize.retitle_divider(name, title)
        reason = trigger_organize.divider_title_error(value, title)
        if reason:
            QMessageBox.warning(self, "Cannot use that section title", f'"{title}": {reason}.')
            return
        if value == name:
            return
        spec = next(s for s in trigger_fields.TRIGGER_FIELDS if s.name == "name")
        self._on_trigger_field(index, spec, value)
        self._refresh_labels(index)
        current = self._current_entry()
        if current is not None and current[0] == "trigger":
            self._populate_property_form()

    def request_move_to_section(self, header_index: int | None) -> None:
        if not self._editable or self._section_view_reason() or self._move_blocker():
            return
        self._on_trigger_structural("move_to_section", self.selected_trigger_indices(), header_index)

    def _update_section_buttons(self, can_edit: bool) -> None:
        view = self._section_view_reason()
        self.section_new_button.setEnabled(can_edit and not view)
        self.section_new_button.setToolTip(self._NEW_SECTION_TIP + view)
        name = self._current_name() if can_edit and not view else None
        single = len(self.selected_trigger_indices()) == 1
        self.section_rename_action.setEnabled(
            single and name is not None and trigger_organize.split_divider(name) is not None
        )
        blocker = view or self._move_blocker()
        self.section_move_button.setEnabled(can_edit and not blocker)
        self.section_move_button.setToolTip(self._MOVE_TO_SECTION_TIP + blocker)

    def _select_default_trigger(self) -> None:
        """topLevelItem(0), or its first child when that row is the synthetic
        leading header -- selecting the header itself would show an empty
        entry pane on most real files (a before-first run is 1..6 triggers
        and never zero once dividers exist, per the corpus census)."""
        if not self.tree.topLevelItemCount():
            return
        first = self.tree.topLevelItem(0)
        if first.data(self._ROLE_COL, Qt.UserRole) is None and first.childCount():
            first = first.child(0)
        self._set_current_trigger_item(first)

    def _manager(self):
        """The live TriggerManager, re-fetched every call.

        Never cached, for the same process-global poisoning reason
        show_scenario() re-parses. Memoized upstream, so this is a depoison().
        """
        if self._loaded is None:
            return None
        return parse_triggers(self._loaded)

    # -- selection preservation ----------------------------------------------

    def _selection_state(self) -> tuple:
        """(current trigger path, selected trigger paths, entry paths,
        scroll) so a rebuild keeps the user's place. By list index / tree path, never by trigger_id: ids are
        positional and a reorder or removal renumbers the whole list.

        The entry element is (current entry path, the other selected entry
        paths), so a multi-entry selection (GH #60) survives a rebuild whole.
        """
        current_entry = self.entry_tree.currentItem()
        entry_path = self._entry_row_path(current_entry)
        other_entries = tuple(
            self._entry_row_path(item)
            for item in self.entry_tree.selectedItems()
            if item is not current_entry
        )
        current = self.tree.currentItem()
        selected = tuple(
            self._trigger_tree_path(item) for item in self.tree.selectedItems() if item is not current
        )
        return (
            self._trigger_tree_path(current),
            selected,
            (entry_path, other_entries),
            self.tree.verticalScrollBar().value(),
        )

    def _entry_row_path(self, item) -> tuple[int, int | None] | None:
        """(top-level row, child row) for `item` in self.entry_tree, or None;
        the same shape _trigger_tree_path() uses for the trigger tree."""
        if item is None:
            return None
        parent = item.parent()
        if parent is None:
            return (self.entry_tree.indexOfTopLevelItem(item), None)
        return (self.entry_tree.indexOfTopLevelItem(parent), parent.indexOfChild(item))

    def _entry_item_at_path(self, path):
        """The entry-tree item at an _entry_row_path() path, or None."""
        if path is None:
            return None
        top, child = path
        if not 0 <= top < self.entry_tree.topLevelItemCount():
            return None
        item = self.entry_tree.topLevelItem(top)
        if child is None:
            return item
        return item.child(child) if 0 <= child < item.childCount() else None

    def _trigger_tree_path(self, item) -> tuple[int, int] | None:
        """(top-level row, child row) for `item` in self.tree, or None.

        child row is None when `item` is itself a top-level row -- the same
        shape _selection_state() already uses for the entry tree above, so a
        flat document (no children anywhere) restores exactly as it did
        before 4c grouping existed.
        """
        if item is None:
            return None
        parent = item.parent()
        if parent is None:
            return (self.tree.indexOfTopLevelItem(item), None)
        return (self.tree.indexOfTopLevelItem(parent), parent.indexOfChild(item))

    def _restore_selection(self, state: tuple) -> None:
        trigger_path, selected_paths, (entry_path, other_entry_paths), scroll = state
        self._restore_trigger_path(trigger_path)
        # After the current item, never before: setCurrentItem() clears the
        # selection (ClearAndSelect).
        for path in selected_paths:
            item = self._item_at_path(path)
            if item is not None and item.data(self._ROLE_COL, Qt.UserRole) is not None:
                item.setSelected(True)
        self.tree.verticalScrollBar().setValue(scroll)
        item = self._entry_item_at_path(entry_path)
        if item is None:
            return
        self._set_current_entry_item(item)
        # The rest after the current one, for the trigger tree's reason above.
        for path in other_entry_paths:
            other = self._entry_item_at_path(path)
            if other is not None:
                other.setSelected(True)

    def _item_at_path(self, path: tuple[int, int] | None):
        """The trigger-tree item at a _trigger_tree_path() path, or None."""
        item = None
        if path is not None:
            top, child = path
            if 0 <= top < self.tree.topLevelItemCount():
                item = self.tree.topLevelItem(top)
                if child is not None and 0 <= child < item.childCount():
                    item = item.child(child)
        return item

    def _restore_trigger_path(self, path: tuple[int, int] | None) -> None:
        item = self._item_at_path(path)
        # Falls back to the default selection rather than the raw item on any
        # miss -- an out-of-range path, or a resolved item that turned out to
        # be the synthetic header (grouping can restructure which row a given
        # path now points at, e.g. after a rename crosses the divider
        # predicate -- see _changed()).
        if item is not None and item.data(self._ROLE_COL, Qt.UserRole) is not None:
            self._set_current_trigger_item(item)
        else:
            self._select_default_trigger()

    # -- the trigger list ----------------------------------------------------

    def _trigger_label(self, trigger) -> str:
        name = (self._read(trigger, "name") or "").strip() or "(unnamed)"
        return name if self._read(trigger, "enabled") else f"{name}  (disabled)"

    def _exec_order_readout(
        self, loaded: LoadedScenario, pending_exec_order: int | None
    ) -> str:
        """Read-only: which axis actually governs execution. Changing it is a
        Map Options edit, not a trigger edit, so this never touches
        `self.trigger_edits` (there is none -- the panel doesn't own one) or
        builds a TriggerEditModel -- see exec_order_value()'s docstring.
        `pending_exec_order` is the caller-resolved value of an in-progress
        edit, passed in rather than read through a model here for that same
        reason.
        """
        mode, unsaved = resolve_exec_mode(exec_order_value(loaded), pending_exec_order)
        if mode == EXEC_MODE_UNKNOWN:
            return "Execution order is not stored in this file."
        suffix = " - unsaved change." if unsaved else "."
        if mode == EXEC_MODE_LEGACY:
            return f"Executes in trigger-ID order (legacy){suffix}"
        return f"Executes in display order{suffix}"

    def current_trigger_index(self) -> int | None:
        """The trigger's stable list index, not its row -- these diverge under
        a display-order sort. Each item's Qt.UserRole carries the index set
        at populate time (show_scenario()); nothing here assumes row order.

        Also None when the selected row is 4c's one synthetic row (the
        "(before the first section)" header) -- its Qt.UserRole is None by
        construction (_populate_grouped()), which is what every consumer
        below already treats as "no trigger selected" without a special case.
        """
        item = self.tree.currentItem()
        if item is None:
            return None
        return item.data(self._ROLE_COL, Qt.UserRole)

    def selected_trigger_indices(self) -> list[int]:
        """Every selected trigger's list index, ordered by display position
        whatever the sort mode: the order a paste reproduces and Move acts in.

        Skips 4c's synthetic header (Qt.UserRole None) and any row a filter
        hides, so a filtered-out row is never swept into a bulk verb.
        """
        indices = []
        for item in self.tree.selectedItems():
            index = item.data(self._ROLE_COL, Qt.UserRole)
            if index is None or item.isHidden():
                continue
            indices.append(index)
        return sorted(indices, key=lambda index: self._display_position.get(index, index))

    def _on_trigger_selected(self) -> None:
        if self._populating:
            return
        if len(self.selected_trigger_indices()) > 1:
            # The selection handler owns the multi state; populating the
            # current trigger's entries here would only be cleared again.
            self._show_multi_page()
        else:
            self._populate_entry_tree()
        self._update_buttons()

    def _on_trigger_selection_changed(self) -> None:
        """The detail page and buttons follow the selection *set*. The
        entry tree is repopulated only when leaving the multi page, since
        currentItemChanged does not fire when the current item stays put."""
        if self._populating:
            return
        if len(self.selected_trigger_indices()) > 1:
            self._show_multi_page()
        elif self.detail_stack.currentIndex() == self._DETAIL_MULTI:
            self.detail_stack.setCurrentIndex(self._DETAIL_FORM)
            self._populate_entry_tree()
        self._update_buttons()
        self._on_selection_changed()

    def _show_multi_page(self) -> None:
        """The Units-mode precedent: N selected shows a count, not a form.
        The entry tree is cleared too, or it would keep showing (and its
        buttons keep acting on) the current trigger alone."""
        if self.detail_stack.currentIndex() == self._DETAIL_PICKER:
            self._close_picker()
        count = len(self.selected_trigger_indices())
        self.multi_label.setText(f"{count} triggers selected")
        self._populating = True
        try:
            self.entry_tree.clear()
            self._clear_property_form()
        finally:
            self._populating = False
        self.detail_stack.setCurrentIndex(self._DETAIL_MULTI)

    def _populate_entry_tree(self) -> None:
        self._populating = True
        try:
            self.entry_tree.clear()
            self._clear_property_form()
            manager = self._manager()
            index = self.current_trigger_index()
            if manager is None or index is None or index >= len(manager.triggers):
                return
            trigger = manager.triggers[index]

            root = QTreeWidgetItem(["Trigger", self._trigger_label(trigger)])
            root.setData(0, Qt.UserRole, ("trigger", -1))
            self.entry_tree.addTopLevelItem(root)

            for kind, label, entries in (
                ("condition", "Conditions", list(self._read(trigger, "conditions") or [])),
                ("effect", "Effects", list(self._read(trigger, "effects") or [])),
            ):
                group = QTreeWidgetItem([f"{label} ({len(entries)})", ""])
                # Marked rather than identified by row position: reading these
                # back as topLevelItem(1)/(2) works today and breaks the moment
                # the tree grows a row.
                group.setData(0, Qt.UserRole, ("group", kind))
                self.entry_tree.addTopLevelItem(group)
                for entry_index, entry in enumerate(entries):
                    child = QTreeWidgetItem(list(self._describe(kind, entry)))
                    child.setData(0, Qt.UserRole, (kind, entry_index))
                    group.addChild(child)
                group.setExpanded(True)
            self._fit_entry_columns()
        finally:
            self._populating = False
        self._set_current_entry_item(self.entry_tree.topLevelItem(0))

    def _vocab_for(self, kind: str):
        """(entries map, presentation map, type attribute) for a kind."""
        if self._vocabulary is None:
            return (None, {}, "condition_type" if kind == "condition" else "effect_type")
        if kind == "condition":
            return (
                self._vocabulary.conditions,
                self._vocabulary.condition_presentation,
                "condition_type",
            )
        return (self._vocabulary.effects, self._vocabulary.effect_presentation, "effect_type")

    def _definition_for(self, kind: str, entry):
        entries, _, type_attribute = self._vocab_for(kind)
        if entries is None:
            return None
        return entries.get(self._read(entry, type_attribute))

    def _describe(self, kind: str, entry) -> tuple[str, str]:
        """(name, detail) for one condition or effect row."""
        _, _, type_attribute = self._vocab_for(kind)
        definition = self._definition_for(kind, entry)
        if definition is None:
            return (f"{kind} type {self._read(entry, type_attribute)}", "")

        details = []
        for attribute in definition.attributes:
            if attribute == type_attribute:
                continue
            value = self._read(entry, attribute)
            # -1 is the library's "unset" sentinel across almost every numeric
            # trigger field, so showing it would bury the fields that are set.
            if value is None or value == -1 or value == "" or value == []:
                continue
            details.append(f"{attribute}={value}")
        return (definition.name.replace("_", " "), ", ".join(details))

    # -- the property form ---------------------------------------------------

    def _clear_property_form(self) -> None:
        """Empty the form, unparenting each widget immediately.

        setParent(None) rather than deleteLater() alone: deleteLater defers
        destruction to the next event-loop pass, so the old rows stay visible
        and keep their geometry, and the next populate draws its rows on top of
        them. That renders as every label overlapping every other one, which no
        widget-level assertion notices.
        """
        self._rows = []
        self._form_refs = None
        self._unit_ref_rows = {}
        self._pick_buttons = {}
        while self.property_form.count():
            item = self.property_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _current_entry(self) -> tuple[str, int, object] | None:
        """(kind, entry index, live object) for whatever the detail tree has
        selected, or None."""
        item = self.entry_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if data is None:
            return None
        kind, entry_index = data
        if kind == "group":
            # A Conditions/Effects heading. It carries its kind for the button
            # logic, but it is not an entry and has no fields of its own.
            return None
        manager = self._manager()
        trigger_index = self.current_trigger_index()
        if manager is None or trigger_index is None or trigger_index >= len(manager.triggers):
            return None
        trigger = manager.triggers[trigger_index]
        if kind == "trigger":
            return (kind, -1, trigger)
        entries = list(self._read(trigger, f"{kind}s") or [])
        if not 0 <= entry_index < len(entries):
            return None
        return (kind, entry_index, entries[entry_index])

    def selected_entry_refs(self) -> list[tuple[str, int]]:
        """(kind, index) for every selected condition or effect, in tree
        order: conditions then effects, each in list order.

        Drops the trigger row, the group headings and hidden rows (_picked()'s
        guard), so a row the user cannot see is never swept into a bulk verb.
        """
        rows = []
        for item in self.entry_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is None or item.isHidden() or data[0] in ("trigger", "group"):
                continue
            rows.append((self._entry_row_path(item), data))
        return [ref for _path, ref in sorted(rows)]

    def _form_refs_now(self) -> tuple[tuple[str, int], ...]:
        """The refs the property form should show: the selected entries, or
        the current row alone when none is selected (the trigger row's form)."""
        refs = self.selected_entry_refs()
        if refs:
            return tuple(refs)
        current = self._current_entry()
        return () if current is None else ((current[0], current[1]),)

    def _entries_for(self, refs) -> list[tuple[str, int, object]]:
        """(kind, index, live object) per ref, dropping any that no longer
        resolves against the current trigger. ("trigger", -1) is the trigger."""
        manager = self._manager()
        trigger_index = self.current_trigger_index()
        if manager is None or trigger_index is None or trigger_index >= len(manager.triggers):
            return []
        trigger = manager.triggers[trigger_index]
        resolved = []
        for kind, entry_index in refs:
            if kind == "trigger":
                resolved.append((kind, -1, trigger))
                continue
            entries = list(self._read(trigger, f"{kind}s") or [])
            if 0 <= entry_index < len(entries):
                resolved.append((kind, entry_index, entries[entry_index]))
        return resolved

    def _on_entry_selected(self) -> None:
        """currentItemChanged."""
        self._sync_entry_form()

    def _on_entry_selection_changed(self) -> None:
        """itemSelectionChanged: a Ctrl+click can change the set without
        moving the current row, so the form follows the set."""
        self._sync_entry_form()

    def _sync_entry_form(self) -> None:
        """Rebuild the form for the current selection, once per gesture.

        A click emits both entry-tree signals, and so does
        _populate_entry_tree()'s end-of-populate setCurrentItem() (#27 Trap 5).
        The latch lives here, not in _populate_property_form(), which the
        refresh paths call directly and must never skip.
        """
        if self._populating:
            return
        if self._form_refs_now() != self._form_refs:
            self._populate_property_form()
        self._update_buttons()
        self.on_focus_changed()

    def _specs_for(self, kind: str, entry) -> tuple:
        if kind == "trigger":
            return trigger_fields.TRIGGER_FIELDS
        definition = self._definition_for(kind, entry)
        if definition is None:
            return ()
        _, presentation, type_attribute = self._vocab_for(kind)
        specs = trigger_fields.field_specs(definition, presentation, type_attribute)
        if kind == "effect":
            # Which cluster slot is authoritative is a runtime decision, and
            # writing an inert one can corrupt the live one.
            specs = trigger_fields.apply_quantity_cluster_rule(specs, entry)
        return specs

    def _populate_property_form(self) -> None:
        self._populating = True
        refs = self._form_refs_now()
        try:
            self._clear_property_form()
            entries = self._entries_for(refs)
            self._form_refs = refs
            if not entries:
                return
            # N = 1 is the same code with a one-entry intersection (GH #60).
            specs = trigger_fields.shared_specs([self._specs_for(k, entry) for k, _i, entry in entries])
            if not specs:
                empty = QLabel(self._no_fields_text(len(entries)))
                # Wrapped: the N-entry wording is wider than the pane's minimum.
                empty.setWordWrap(True)
                self.property_form.addRow(empty)
                return
            objects = [entry for _k, _i, entry in entries]
            kind, entry_index, _entry = entries[0]
            for spec in specs:
                value = trigger_fields.shared_value(objects, spec.attribute, self._read)
                widget = self._build_widget(spec, refs, value)
                self.property_form.addRow(spec.label, widget)
                self._rows.append((spec, kind, entry_index, widget))
        finally:
            self._populating = False
        self._fit_property_height()

    @staticmethod
    def _no_fields_text(count: int) -> str:
        """Why the form is empty. A mixed selection is not refused; it says
        what it found (GH #60)."""
        if count == 1:
            return "This type has no editable fields."
        return f"No fields are shared by the {count} selected entries."

    def _fit_property_height(self) -> None:
        """Tell the scroll area how tall the wrapped form actually is.

        WrapLongRows makes a row's height depend on the width it is given, and
        a QScrollArea sizes its widget from sizeHint(), which is computed as if
        nothing wrapped. The area therefore hands the host the unwrapped height
        and every wrapped row draws over the next one -- rows visibly on top of
        each other, while host width, scrollbar state and field widths all
        still measure correct. Asking the layout for heightForWidth at the real
        viewport width is what makes the vertical scrollbar appear instead.
        """
        # activate() first, or this measures the *previous* form: the rows were
        # added moments ago and Qt has not laid them out yet, so heightForWidth
        # answers for whatever the form held before. That is how this first
        # landed as a 72 px minimum on a form needing 454.
        self.property_form.activate()
        height = self.property_form.minimumSize().height()
        width = self.property_area.viewport().width()
        if width > 0 and self.property_form.hasHeightForWidth():
            height = max(height, self.property_form.heightForWidth(width))
        self.property_host.setMinimumHeight(height)

    def _display_text(self, spec, value) -> str:
        """A read-only field's text, in the same terms its editor would use."""
        if value is trigger_fields.MIXED:
            return self._DIFFERS
        if value is None or value == trigger_fields.UNSET:
            return "(unset)"
        if spec.kind == trigger_fields.INT_LIST or spec.presentation == unit_references.LIST_PRESENTATION:
            return trigger_fields.format_int_list(value) or "(unset)"
        if spec.multiline == trigger_fields.XS:
            return trigger_fields.xs_to_display(value)
        if spec.multiline == trigger_fields.PROSE:
            return messages_fields.normalize_for_display("" if value is None else str(value))[0]
        if spec.kind == trigger_fields.BOOL:
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:g}"
        if spec.kind == trigger_fields.ENUM:
            for label, choice in spec.choices:
                if choice == value:
                    return f"{label} ({choice})"
        if spec.kind == trigger_fields.REFERENCE:
            if spec.presentation == "Unit":
                resolved = self.describe_unit_reference(value)
                if resolved.startswith(f"{value}:"):  # a dangling id already leads with itself
                    return resolved
                return f"{value}: {resolved}" if resolved else str(value)
            resolved = trigger_fields.resolve_reference(spec.presentation, value)
            return f"{resolved} ({value})" if resolved else str(value)
        return str(value)

    def _build_widget(self, spec, refs, value) -> QWidget:
        """One form row's editor, for the whole selection (GH #60).

        `value` is what every selected entry holds, or MIXED when they differ.
        An indeterminate widget shows "(differs)" and writes nothing until the
        user changes it, gated per widget kind. The gate is the whole risk:
        editingFinished fires on a bare focus-out, so an ungated blank box
        would write "" or [] to every selected entry for merely tabbing past.
        """
        indeterminate = value is trigger_fields.MIXED
        multi = len(refs) > 1
        editable = self._editable and not spec.read_only

        # Every locked field renders the same way, whatever its kind. A
        # disabled combo instead would show its first choice as though it were
        # the value: an inert armour_attack_class read "WONDER (0)" while the
        # int beside it, locked by the very same rule, honestly said "(unset)".
        if spec.read_only:
            label = QLabel(self._display_text(spec, value))
            label.setEnabled(False)
            return label

        if spec.kind == trigger_fields.BOOL:
            widget = QCheckBox()
            if indeterminate:
                # stateChanged, not toggled: PartiallyChecked already counts as
                # checked, so the first click would emit no toggled at all.
                widget.setTristate(True)
                widget.setCheckState(Qt.PartiallyChecked)
            else:
                widget.setChecked(bool(value))
            widget.setEnabled(editable)
            widget.stateChanged.connect(
                lambda state, s=spec, r=refs, w=widget: self._checkbox_changed(s, r, w, state)
            )
            return widget

        if spec.kind == trigger_fields.STR and spec.multiline == trigger_fields.XS:
            widget = XsTextEdit("" if indeterminate else trigger_fields.xs_to_display(value))
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            widget.editingFinished.connect(lambda s=spec, r=refs, w=widget: self._xs_changed(s, r, w))
            return widget

        if spec.kind == trigger_fields.STR and spec.multiline == trigger_fields.PROSE:
            display, token = messages_fields.normalize_for_display(
                "" if indeterminate or value is None else str(value)
            )
            widget = ProseTextEdit(display, token)
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            widget.editingFinished.connect(lambda s=spec, r=refs, w=widget: self._prose_changed(s, r, w))
            return widget

        if spec.kind == trigger_fields.STR:
            widget = QLineEdit("" if indeterminate or value is None else str(value))
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            # Show the start of the value, not its end. A freshly-set QLineEdit
            # leaves the cursor past the last character, so a name too long for
            # the field rendered as "ure: armour split".
            widget.setCursorPosition(0)
            widget.editingFinished.connect(
                lambda s=spec, r=refs, w=widget, m=multi: self._text_changed(s, r, w, m)
            )
            return widget

        # Unit[] by presentation, not kind: 9 effects (task_object among them) default it to None, not [].
        if spec.kind == trigger_fields.INT_LIST or spec.presentation == unit_references.LIST_PRESENTATION:
            widget = QLineEdit("" if indeterminate else trigger_fields.format_int_list(value))
            widget.setReadOnly(not editable)
            widget.setPlaceholderText(self._DIFFERS if indeterminate else "comma separated")
            widget.editingFinished.connect(
                lambda s=spec, r=refs, w=widget, m=multi: self._list_changed(s, r, w, m)
            )
            if spec.presentation != unit_references.LIST_PRESENTATION or multi:
                return widget
            return self._unit_reference_host(spec, refs, widget, editable)

        if spec.kind == trigger_fields.ENUM:
            if trigger_fields.wants_picker(spec):
                return self._build_enum_picker_widget(spec, refs, value, editable)
            widget = QComboBox()
            _fit_combo_width(widget)
            self._add_differs_item(widget, indeterminate)
            for label, choice in spec.choices:
                widget.addItem(f"{label} ({choice})", choice)
            if not indeterminate and widget.findData(value) < 0 and isinstance(value, int):
                # A value outside the shipped enum. Kept as its own row rather
                # than snapped to the nearest legal one, which would rewrite a
                # field the user never touched.
                widget.addItem(f"unknown ({value})", value)
            widget.setCurrentIndex(0 if indeterminate else max(widget.findData(value), 0))
            widget.setEnabled(editable)
            widget.currentIndexChanged.connect(lambda _, s=spec, r=refs, w=widget: self._combo_changed(s, r, w))
            return widget

        if spec.kind == trigger_fields.REFERENCE:
            catalog_presentation = trigger_fields.CATALOG_PRESENTATIONS.get(spec.presentation)
            if catalog_presentation is not None:
                return self._build_catalog_widget(spec, refs, value, catalog_presentation, editable)
            if spec.presentation in trigger_fields.DOCUMENT_REFERENCES:
                return self._build_document_reference_widget(spec, refs, value, editable)
            # Unit: a placed-unit reference_id, resolved against the open
            # document (describe_unit_reference) and pickable from the map.
            spin = self._trigger_spinbox(value, editable)
            self._connect_spinbox(spin, spec, refs)
            return self._unit_reference_host(spec, refs, spin, editable)

        if spec.kind == trigger_fields.FLOAT:
            spin = self._trigger_float_spinbox(value, editable)
            self._connect_spinbox(spin, spec, refs)
            return spin

        spin = self._trigger_spinbox(value, editable)
        self._connect_spinbox(spin, spec, refs)
        return spin

    def _connect_spinbox(self, spin, spec, refs) -> None:
        """Report a spinbox's edits. A blank (indeterminate) one leaves that
        state on its first real change, and also commits from editingFinished
        when the user typed the parked value, which emits no valueChanged."""
        blank = isinstance(spin, _IndeterminateMixin) and spin.is_indeterminate()
        if not blank:
            spin.valueChanged.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
            return
        spin.valueChanged.connect(lambda new, s=spec, r=refs, w=spin: self._spinbox_changed(s, r, w, new))
        spin.editingFinished.connect(lambda s=spec, r=refs, w=spin: self._spinbox_typed(s, r, w))

    # -- Unit / Unit[] references -------------------------------------------

    # A Unit[] label shows at most this many ids, then "... and N more" (the corpus max is 8).
    UNIT_LIST_LINES = 8

    def _unit_list_text(self, ids) -> str:
        lines = [self.describe_unit_reference(ref_id) for ref_id in ids[: self.UNIT_LIST_LINES]]
        if len(ids) > self.UNIT_LIST_LINES:
            lines.append(f"... and {len(ids) - self.UNIT_LIST_LINES} more")
        return "\n".join(lines)

    def _unit_reference_host(self, spec, refs, editor, editable: bool) -> QWidget:
        """The editor and a Pick button on one line, what the id(s) name below.
        The label follows the spinbox live, a list line edit on commit."""
        host = QWidget()
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(editor, stretch=1)
        column.addLayout(row)
        label = QLabel()
        label.setWordWrap(True)
        column.addWidget(label)
        is_list = isinstance(editor, QLineEdit)

        def refresh(label=label, editor=editor) -> None:
            if is_list:
                try:
                    ids = trigger_fields.parse_int_list(editor.text())
                except ValueError:
                    return
                label.setText(self._unit_list_text(ids))
            elif isinstance(editor, _IndeterminateMixin) and editor.is_indeterminate():
                label.setText("")
            else:
                label.setText(self.describe_unit_reference(editor.value()))
            label.setVisible(bool(label.text()))

        refresh()
        if is_list:
            # After _list_changed's own connection, so a rejected text is already put back.
            editor.editingFinished.connect(refresh)
        else:
            editor.valueChanged.connect(lambda _new: refresh())
        self._unit_ref_rows[spec.name] = (spec, refs, editor, refresh)
        self._add_pick_button(row, spec, refs, editable)
        return host

    def _add_pick_button(self, row, spec, refs, editable: bool) -> None:
        """Pick from map, for one editable entry's field only: the window
        arms a map picker whose clicks come back through apply_picked_reference()."""
        trigger_index = self.current_trigger_index()
        if not editable or len(refs) != 1 or refs[0][0] == "trigger" or trigger_index is None:
            return
        kind, entry_index = refs[0]
        target = (trigger_index, kind, entry_index, spec.name, spec.presentation == unit_references.LIST_PRESENTATION)
        button = QToolButton()
        button.setText("Pick")
        button.setCheckable(True)
        button.setChecked(target == self._armed_pick)
        button.setToolTip(
            "Click units on the map to add or remove them; Esc to finish"
            if target[4]
            else "Click a unit on the map to set this field; Esc to cancel"
        )
        button.toggled.connect(lambda checked, t=target: self.on_pick_unit(t if checked else None))
        self._pick_buttons[target] = button
        row.addWidget(button)

    def set_pick_armed(self, target: tuple | None) -> None:
        """The window's arm state, shown on the Pick buttons without re-firing on_pick_unit."""
        self._armed_pick = target
        for button_target, button in self._pick_buttons.items():
            button.blockSignals(True)
            button.setChecked(button_target == target)
            button.blockSignals(False)

    def unit_reference_ids(self, target: tuple) -> tuple[int, ...]:
        """The ids `target`'s field holds now, or () when the form shows something else."""
        row = self._unit_ref_rows.get(target[3])
        if row is None or not self._pick_target_is_current(target):
            return ()
        spec, refs, _editor, _refresh = row
        entries = self._entries_for(refs)
        if not entries:
            return ()
        value = self._read(entries[0][2], spec.attribute)
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(v for v in values if isinstance(v, int) and v != trigger_fields.UNSET)

    def _pick_target_is_current(self, target: tuple) -> bool:
        trigger_index, kind, entry_index, _field, _is_list = target
        return (
            self.current_trigger_index() == trigger_index
            and len(self.selected_trigger_indices()) <= 1
            and self._form_refs_now() == ((kind, entry_index),)
        )

    def apply_picked_reference(self, target: tuple, ref_id: int) -> bool:
        """Write a map pick through the form's own funnel, so it is one undo
        record like a typed edit. A scalar field takes the id; a list toggles
        it. Returns whether the picker stays armed: only a list pick on the
        still-current target does. A stale target writes nothing."""
        row = self._unit_ref_rows.get(target[3])
        if row is None or not self._editable or not self._pick_target_is_current(target):
            return False
        spec, refs, editor, refresh = row
        if not target[4]:
            editor.setValue(ref_id)
            return False
        current = list(self.unit_reference_ids(target))
        new = [i for i in current if i != ref_id] if ref_id in current else [*current, ref_id]
        editor.setText(trigger_fields.format_int_list(new))
        self._list_changed(spec, refs, editor)
        refresh()
        return True

    def refresh_unit_reference_labels(self) -> None:
        """Re-describe every Unit/Unit[] label, after units moved or changed hands."""
        for _spec, _refs, _editor, refresh in self._unit_ref_rows.values():
            refresh()

    def _spinbox_changed(self, spec, refs, spin, value) -> None:
        spin.clear_indeterminate()
        self._changed(spec, refs, value)

    def _spinbox_typed(self, spec, refs, spin) -> None:
        if not spin.typed_parked_value():
            return
        spin.clear_indeterminate()
        self._changed(spec, refs, spin.value())

    def _checkbox_changed(self, spec, refs, widget, state) -> None:
        """A tristate box writes once it leaves PartiallyChecked, and never
        goes back: "the selection disagrees" is not a value to pick."""
        if state == Qt.PartiallyChecked:
            return
        widget.setTristate(False)
        self._changed(spec, refs, int(state == Qt.Checked))

    def _combo_changed(self, spec, refs, widget) -> None:
        """The "(differs)" row carries None, which no field holds, so leaving
        it selected (or going back to it) writes nothing."""
        data = widget.currentData()
        if data is not None:
            self._changed(spec, refs, data)

    def _text_changed(self, spec, refs, widget, multi: bool) -> None:
        """A STR commit. With N entries it must have been typed in: a blank
        "(differs)" box would otherwise clear every entry on a focus-out.
        N = 1 keeps the plain rule, where _changed()'s equality check suffices."""
        if multi and not widget.isModified():
            return
        self._changed(spec, refs, widget.text())

    def _add_differs_item(self, combo, indeterminate: bool) -> None:
        """The pre-selected "(differs)" first row, whose data is None."""
        if indeterminate:
            combo.addItem(self._DIFFERS, None)

    def _build_catalog_widget(self, spec, refs, value, presentation, editable: bool) -> QWidget:
        """UnitInfo/BuildingInfo/TechInfo: a CatalogLineEdit over
        object_catalog.py's id-dataset catalog, in place of the raw spinbox.
        An empty one commits nothing, which is already the indeterminate rule."""
        indeterminate = value is trigger_fields.MIXED
        widget = CatalogLineEdit(
            presentation.catalog(),
            presentation.default_category,
            placeholder=self._DIFFERS if indeterminate else "(unset)",
        )
        initial = value if isinstance(value, int) and not isinstance(value, bool) else None
        widget.set_value(None if initial == trigger_fields.UNSET else initial)
        widget.setEnabled(editable)
        # Connected after set_value(), matching every other widget here: a
        # populate must not look like a user edit (trigger_fields.py's own
        # note on why _changed()'s _populating guard is otherwise unreachable).
        widget.committed.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
        return widget

    def _build_enum_picker_widget(self, spec, refs, value, editable: bool) -> QWidget:
        """A large ENUM (trigger_fields.wants_picker()): a ValueLineEdit over
        the enum's (label, value) pairs, in place of a combo too long to scroll.

        Unlike _build_catalog_widget(), UNSET (-1) is not mapped to None: an
        enum holding -1 renders "unknown (-1)", exactly as its combo did. Only
        None, an attribute _read() could not reach, shows the placeholder.
        """
        indeterminate = value is trigger_fields.MIXED
        widget = ValueLineEdit(
            [PickerItem(label, choice) for label, choice in spec.choices],
            show_values=True,
            browse_title=spec.label.capitalize(),
            placeholder=self._DIFFERS if indeterminate else "(unset)",
        )
        widget.set_value(value if isinstance(value, int) and not isinstance(value, bool) else None)
        widget.setEnabled(editable)
        # Connected after set_value(), matching every other widget here.
        widget.committed.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
        return widget

    def _build_document_reference_widget(self, spec, refs, value, editable: bool) -> QComboBox:
        """TriggerId/VariableId: a combo box over the open document's own
        trigger or variable list, resolved fresh every populate -- trigger_id
        is renumbered by remove_triggers()/reorder_triggers(), so a stale
        combo would silently point at the wrong trigger after a structural
        edit.

        No _fit_combo_width() here: that caps a combo at 12 characters, sized
        for enum labels, where a trigger or variable name runs far longer and
        would clip -- WrapLongRows already puts a wide editor on its own line
        instead of forcing the form into horizontal overflow.
        """
        indeterminate = value is trigger_fields.MIXED
        widget = QComboBox()
        self._add_differs_item(widget, indeterminate)
        widget.addItem("(unset)", trigger_fields.UNSET)
        manager = self._manager()
        if manager is not None:
            choices = (
                object_catalog.trigger_choices(manager)
                if spec.presentation == "TriggerId"
                else object_catalog.variable_choices(manager)
            )
            for label, choice in choices:
                widget.addItem(f"{label} ({choice})", choice)
        known = widget.findData(value) >= 0
        if not indeterminate and not known and isinstance(value, int) and not isinstance(value, bool):
            # An id outside the document's current list -- kept as its own
            # row rather than snapped to "(unset)", which would rewrite a
            # field the user never touched (same reasoning as the ENUM branch).
            widget.addItem(f"unknown ({value})", value)
        widget.setCurrentIndex(0 if indeterminate else max(widget.findData(value), 0))
        widget.setEnabled(editable)
        widget.currentIndexChanged.connect(lambda _, s=spec, r=refs, w=widget: self._combo_changed(s, r, w))
        return widget

    @staticmethod
    def _trigger_spinbox(value, editable: bool) -> QSpinBox:
        """_make_spinbox() with this panel's own range and unset sentinel.
        -1 is the library's unset marker, shown as text rather than as a
        number the user would have to know the meaning of. MIXED gets the
        blank indeterminate box (GH #60)."""
        indeterminate = value is trigger_fields.MIXED
        return _make_spinbox(
            None if indeterminate else value,
            editable,
            minimum=trigger_fields.UNSET,
            maximum=2**31 - 1,
            special_value_text="(unset)",
            indeterminate=indeterminate,
        )

    @staticmethod
    def _trigger_float_spinbox(value, editable: bool) -> QDoubleSpinBox:
        """_trigger_spinbox()'s float counterpart, for quantity_float: same
        unset sentinel, same keyboard tracking off."""
        indeterminate = value is trigger_fields.MIXED
        spin = IndeterminateDoubleSpinBox() if indeterminate else QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(trigger_fields.UNSET, 2**31 - 1)
        spin.setSpecialValueText("(unset)")
        number = not indeterminate and isinstance(value, (int, float)) and not isinstance(value, bool)
        spin.setValue(float(value) if number else trigger_fields.UNSET)
        spin.setEnabled(editable)
        spin.setKeyboardTracking(False)
        if indeterminate:
            spin.set_indeterminate()
        return spin

    # -- reporting an edit ---------------------------------------------------

    def _changed(self, spec, refs, value) -> None:
        """The one place a widget signal becomes a reported edit, for one
        entry or a selection of them (GH #60).

        The equality check is what actually fires in practice: editingFinished
        arrives on every focus-out whether the text changed or not, and
        commit_trigger_edit() pushes unconditionally, so without it tabbing
        through the form would record one phantom undo step per field. Plural,
        it is "at least one selected entry differs": a value the first entry
        already holds must still reach the others.

        A widget built for a different selection than the live one writes
        nothing: its captured refs are stale, so it cannot say what it edits.

        `_populating` is defence in depth and is unreachable today: every widget
        in _build_widget() is connected *after* its value is set, so a populate
        emits nothing. It is kept because that ordering is the only thing making
        it unreachable, and a future edit that connects before setting would
        otherwise report one edit per field with no other guard in the way. A
        spinbox cannot hold None and a combo cannot hold an out-of-enum value,
        so those populates would differ from the live value and sail past the
        equality check.
        """
        if self._populating or not self._editable or spec.read_only:
            return
        if value is trigger_fields.MIXED:
            return
        if len(self.selected_trigger_indices()) > 1:
            return
        live = self._form_refs_now()
        if tuple(refs) != live:
            return
        trigger_index = self.current_trigger_index()
        entries = self._entries_for(live)
        if trigger_index is None or not entries:
            return
        kind = entries[0][0]
        targets = [(k, i) for k, i, entry in entries if self._read(entry, spec.attribute) != value]
        if not targets:
            return
        before = self._read(entries[0][2], spec.attribute)
        slots_before = self._quantity_slots(entries)

        # 4c: a rename that crosses the divider predicate changes the section
        # partition, which a label patch can't reflect -- escalate to a full
        # repopulate instead. Checked before the edit is applied, since after
        # it `entry`'s own name has already become the new one.
        repartitions = (
            kind == "trigger"
            and spec.name == "name"
            and trigger_organize.is_divider(before or "") != trigger_organize.is_divider(value or "")
        )

        if kind == "trigger":
            self._on_trigger_field(trigger_index, spec, value)
        else:
            # Only the entries that differ, so Effect.quantity's setter is not
            # re-run on an entry already holding the value.
            self._on_entry_field(trigger_index, targets, spec, value)

        if repartitions:
            self.show_scenario(self._loaded, self._pending_exec_order)
            self.select_trigger(trigger_index)
        else:
            self._refresh_labels(trigger_index)

        if self._quantity_slots(self._entries_for(live)) != slots_before:
            # Deferred: this runs inside the editing widget's own signal, and
            # repopulating unparents that widget immediately.
            QTimer.singleShot(0, lambda r=live: self._repopulate_form_if_current(r))

    @staticmethod
    def _quantity_slots(entries) -> list[str]:
        """Each effect's live quantity-cluster slot, to spot an edit that
        switched one (the form's locked fields then change)."""
        return [trigger_fields.live_quantity_slot(entry) if kind == "effect" else "" for kind, _i, entry in entries]

    def _repopulate_form_if_current(self, refs) -> None:
        """The cluster switch's deferred form rebuild, skipped if the user has
        selected something else in between."""
        if self._form_refs_now() == tuple(refs):
            self._populate_property_form()

    def _list_changed(self, spec, refs, widget: QLineEdit, multi: bool = False) -> None:
        if self._populating:
            return
        if multi and not widget.isModified():
            # The STR rule: a blank "(differs)" box must not write [] on a focus-out.
            return
        try:
            value = trigger_fields.parse_int_list(widget.text())
        except ValueError:
            # Rejected whole rather than half-written: "4, x" must not become
            # [4] and silently drop the user's second value.
            objects = [entry for _k, _i, entry in self._entries_for(self._form_refs_now())]
            shared = trigger_fields.shared_value(objects, spec.attribute, self._read)
            widget.setText("" if shared is trigger_fields.MIXED else trigger_fields.format_int_list(shared))
            return
        self._changed(spec, refs, value)

    def _xs_changed(self, spec, refs, widget: XsTextEdit) -> None:
        # Untouched text is a guaranteed no-op, whatever separators the stored
        # value holds; _changed()'s equality check alone misses CRLF and LF.
        # A "(differs)" box latches "", so leaving it blank writes nothing.
        text = widget.toPlainText()
        if text == widget.latched_text:
            return
        self._changed(spec, refs, trigger_fields.xs_from_display(text))
        widget.latched_text = text

    def _prose_changed(self, spec, refs, widget: ProseTextEdit) -> None:
        # Same latch as _xs_changed(): on a CR- or CRLF-stored field the
        # display text differs from the stored value, so _changed()'s equality
        # check alone would record a phantom undo step on a bare focus-out.
        text = widget.toPlainText()
        if text == widget.latched_text:
            return
        value = messages_fields.substitute_newlines(text, widget.newline_token)
        self._changed(spec, refs, value)
        widget.latched_text = text

    def _refresh_labels(self, trigger_index: int) -> None:
        """Update only the rows an edit can have changed.

        Rebuilding wholesale on every keystroke-commit would scroll a
        590-trigger list back to the top and drop the user's selection.
        """
        self._populating = True
        try:
            manager = self._manager()
            if manager is None or trigger_index >= len(manager.triggers):
                return
            trigger = manager.triggers[trigger_index]
            item = self._item_for_index.get(trigger_index)
            if item is not None:
                item.setText(self._COL_NAME, self._trigger_label(trigger))
                # The tag facet's stashed role, kept in step with a rename that
                # doesn't cross the divider predicate (repartitions handles
                # that case with a full repopulate instead -- see _changed()).
                item.setData(self._ROLE_COL, self._TAG_ROLE, trigger_organize.parse_tag(self._read(trigger, "name") or ""))
                if self._grouped and item.parent() is None:
                    self._retitle_section(trigger_index, (self._read(trigger, "name") or "").strip())

            # Every selected row plus the current one: a group edit (GH #60)
            # changes N rows' detail text at once.
            items = list(self.entry_tree.selectedItems())
            current = self.entry_tree.currentItem()
            if current is not None and current not in items:
                items.append(current)
            for entry_item in items:
                data = entry_item.data(0, Qt.UserRole)
                if data is None or data[0] == "group":
                    continue
                kind, entry_index = data
                if kind == "trigger":
                    entry_item.setText(1, self._trigger_label(trigger))
                    continue
                entries = list(self._read(trigger, f"{kind}s") or [])
                if 0 <= entry_index < len(entries):
                    name, detail = self._describe(kind, entries[entry_index])
                    entry_item.setText(0, name)
                    entry_item.setText(1, detail)
        finally:
            self._populating = False

    # -- structural editing (4b.6b) ------------------------------------------
    #
    # Same division as the property form: the panel decides what the user asked
    # for and reports it, the window decides what that means to the model. The
    # panel never adds a trigger itself, so the begin/commit contract stays in
    # one place.

    def refresh_entries(self, select: tuple[str, int] | None = None) -> None:
        """Rebuild the detail tree only, optionally selecting one entry.

        Adding or removing a condition changes one trigger's contents, not the
        trigger list, so rebuilding the whole panel would scroll a 590-trigger
        list back to the top for no reason.
        """
        self._populate_entry_tree()
        if select is not None:
            self.select_entry(*select)

    def select_trigger(self, index: int) -> None:
        """Select the trigger at list `index`, clamped. Used after a
        structural edit to land the user on the trigger that edit produced,
        and after a sort-mode change to keep the same trigger selected across
        rows that just moved.

        `index` is a trigger list index, not a row -- translated through
        `_item_for_index`, populated at the most recent show_scenario(). Falls
        back to the default selection if the index is not currently shown
        (out of range, or -- not reachable today, since nothing filters the
        tree by index -- otherwise absent).
        """
        self.select_triggers([index])

    def _show_trigger_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.trigger_context_menu(item).exec_(self.tree.viewport().mapToGlobal(pos))

    def trigger_context_menu(self, item) -> QMenu:
        """The trigger tree's right-click menu for `item`. Built per click, so
        its enabled states describe that row."""
        menu = QMenu(self.tree)
        section = menu.addAction("Select Section")
        section.setEnabled(self._section_for(item) is not None)
        section.triggered.connect(lambda _=False, i=item: self.select_section_of(i))
        tag = menu.addAction("Select Tag")
        index = item.data(self._ROLE_COL, Qt.UserRole)
        tag.setEnabled(index is not None and item.data(self._ROLE_COL, self._TAG_ROLE) is not None)
        tag.triggered.connect(lambda _=False, i=item: self.select_tag_of(i))
        return menu

    def _show_entry_context_menu(self, pos) -> None:
        item = self.entry_tree.itemAt(pos)
        if item is None:
            return
        self.entry_context_menu(item).exec_(self.entry_tree.viewport().mapToGlobal(pos))

    def entry_context_menu(self, item) -> QMenu:
        """The entry tree's right-click menu for `item`, built per click."""
        menu = QMenu(self.entry_tree)
        same = menu.addAction("Select Same Type")
        same.setEnabled(self._same_type_refs(item) != [])
        same.triggered.connect(lambda _=False, i=item: self.select_same_type_as(i))
        return menu

    def _same_type_refs(self, item) -> list[tuple[str, int]]:
        """Every visible entry in this trigger of `item`'s kind and type, or
        [] when `item` is not a condition or effect row."""
        data = None if item is None else item.data(0, Qt.UserRole)
        if data is None or data[0] in ("trigger", "group"):
            return []
        kind, entry_index = data
        trigger = self._entries_for([("trigger", -1)])
        if not trigger:
            return []
        _, _, type_attribute = self._vocab_for(kind)
        siblings = list(self._read(trigger[0][2], f"{kind}s") or [])
        if not 0 <= entry_index < len(siblings):
            return []
        wanted = self._read(siblings[entry_index], type_attribute)
        return [
            (kind, index)
            for index, sibling in enumerate(siblings)
            if self._read(sibling, type_attribute) == wanted
            and (row := self._entry_item_for(kind, index)) is not None
            and not row.isHidden()
        ]

    def select_same_type_as(self, item) -> None:
        """Select every entry of the clicked row's type (GH #60's literal ask),
        the clicked row current. Selection only, never model state."""
        refs = self._same_type_refs(item)
        clicked = tuple(item.data(0, Qt.UserRole)) if refs else None
        if clicked in refs:
            refs.remove(clicked)
            refs.insert(0, clicked)
        self.select_entries(refs)

    def _section_for(self, item):
        """The grouped tree's Section `item` belongs to (a header row is its
        own section's), or None when flat."""
        if not self._grouped or item is None:
            return None
        top = item.parent() or item
        header = top.data(self._ROLE_COL, Qt.UserRole)
        if header is None:
            # The synthetic leading header: its section has no divider.
            return next((s for s in self._sections if s.header_index is None), None)
        return next((s for s in self._sections if s.header_index == header), None)

    def select_section_of(self, item) -> None:
        """Select `item`'s whole section: its divider trigger, if it has one,
        and every member. Hidden rows stay out, as for any selection."""
        section = self._section_for(item)
        if section is None:
            return
        indices = ([] if section.header_index is None else [section.header_index]) + list(section.member_indices)
        self.select_triggers([i for i in indices if not self._item_for_index[i].isHidden()])

    def select_tag_of(self, item) -> None:
        """Select every visible trigger sharing `item`'s [tag], across sections."""
        tag = item.data(self._ROLE_COL, self._TAG_ROLE)
        if tag is None or item.data(self._ROLE_COL, Qt.UserRole) is None:
            return
        indices = [
            index
            for index in self._display_slots
            if (row := self._item_for_index.get(index)) is not None
            and not row.isHidden()
            and row.data(self._ROLE_COL, self._TAG_ROLE) == tag
        ]
        self.select_triggers(indices)

    def _set_current_trigger_item(self, item) -> None:
        """Make `item` current and the whole selection, whatever keys are held.
        Without an explicit command, ExtendedSelection reads the live keyboard
        modifiers, so a Ctrl still down from Ctrl+V would toggle instead."""
        self.tree.setCurrentItem(item, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def select_triggers(self, indices) -> None:
        """Select every trigger at list `indices`, the first becoming the
        current item. Unresolvable indices are skipped; with none left this
        falls back to the default selection, like select_trigger()."""
        items = [item for item in map(self._item_for_index.get, indices) if item is not None]
        if not items:
            self._select_default_trigger()
            return
        # Current first: setCurrentItem() clears the selection, so the rest
        # are added after it.
        self._set_current_trigger_item(items[0])
        for item in items[1:]:
            item.setSelected(True)

    def select_all(self) -> None:
        """Edit > Select All in Triggers mode (GH #3): every trigger a filter
        leaves visible. Not tree.selectAll(), which skips collapsed sections'
        members. One selection change, not one per row."""
        items = [
            row
            for index in self._display_slots
            if (row := self._item_for_index.get(index)) is not None and not row.isHidden()
        ]
        if not items:
            return
        self._set_current_trigger_item(items[0])
        selection = QItemSelection()
        for item in items[1:]:
            model_index = self.tree.indexFromItem(item)
            selection.select(model_index, model_index)
        self.tree.selectionModel().select(selection, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def clear_trigger_selection(self) -> None:
        self.tree.clearSelection()

    def move_row(self, trigger_index: int, delta: int) -> None:
        """Patch the tree for a display-order-only move when flat: only the
        moved trigger's row changes, so this is a take/insert of one item
        rather than show_scenario()'s full repopulate -- the difference
        between one row moving and 590 rows rebuilding on
        `old-allies-final-v2` for a single Move Up click.

        Grouped, a move can cross a section boundary -- a parentage change,
        not a same-parent row swap -- so this falls back to a full repopulate
        there. Per the plan's "measure before optimising": no grouped fast
        path is built speculatively, only if repopulate cost is measured and
        shown to matter.

        `delta` must be +-1: Move Up/Down only ever swap adjacent display
        slots (`moved_display_order()`'s own contract).
        """
        if self._grouped:
            self.show_scenario(self._loaded, self._pending_exec_order)
            self.select_trigger(trigger_index)
            return

        item = self._item_for_index.get(trigger_index)
        if item is None:
            return
        row = self.tree.indexOfTopLevelItem(item)
        target = row + delta
        if not 0 <= target < self.tree.topLevelItemCount():
            return
        self.tree.takeTopLevelItem(row)
        self.tree.insertTopLevelItem(target, item)

        # _item_for_index needs no correction: both items kept their identity,
        # only their row changed, and the dict maps by identity, not row.
        position = self._display_position.get(trigger_index)
        if position is not None:
            neighbor_position = position + delta
            if 0 <= neighbor_position < len(self._display_slots):
                neighbor_index = self._display_slots[neighbor_position]
                self._display_slots[position] = neighbor_index
                self._display_slots[neighbor_position] = trigger_index
                self._display_position[trigger_index] = neighbor_position
                self._display_position[neighbor_index] = position
                # Both rows' position text is now stale; nothing else on this
                # fast path would rewrite it.
                item.setText(self._COL_POS, self._position_text(trigger_index))
                neighbor_item = self._item_for_index.get(neighbor_index)
                if neighbor_item is not None:
                    neighbor_item.setText(self._COL_POS, self._position_text(neighbor_index))

        self._set_current_trigger_item(item)
        self._update_buttons()

    def select_entry(self, kind: str, entry_index: int) -> None:
        self.select_entries([(kind, entry_index)])

    def select_entries(self, refs) -> None:
        """Select exactly these (kind, index) entry rows, the first becoming
        the current one. Selection only: no model state is touched."""
        items = [item for item in (self._entry_item_for(*ref) for ref in refs) if item is not None]
        if not items:
            return
        # Current first: setCurrentItem() clears the selection.
        self._set_current_entry_item(items[0])
        for item in items[1:]:
            item.setSelected(True)

    def _set_current_entry_item(self, item) -> None:
        """_set_current_trigger_item() for the entry tree: an explicit
        ClearAndSelect, so a held Ctrl cannot turn it into a toggle."""
        if item is None:
            self.entry_tree.setCurrentItem(None)
            return
        self.entry_tree.setCurrentItem(item, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def _entry_item_for(self, kind: str, entry_index: int):
        """The entry-tree row for one (kind, index) ref, or None."""
        for i in range(self.entry_tree.topLevelItemCount()):
            top = self.entry_tree.topLevelItem(i)
            if top.data(0, Qt.UserRole) != ("group", kind):
                continue
            return top.child(entry_index) if 0 <= entry_index < top.childCount() else None
        return None

    def current_entry_ref(self) -> tuple[str, int] | None:
        """Public (kind, index) of the current condition or effect, so the
        viewer never reads the live entry object off the panel (GH #59)."""
        return self._selected_entry_ref()

    def picker_showing(self) -> bool:
        """Whether the vocabulary picker page is up in place of the form."""
        return self.detail_stack.currentIndex() == self._DETAIL_PICKER

    def _selected_entry_ref(self) -> tuple[str, int] | None:
        """(kind, index) for the selected condition or effect, or None when the
        selection is the trigger row, a group heading, or nothing."""
        current = self._current_entry()
        if current is None:
            return None
        kind, entry_index, _entry = current
        if kind == "trigger":
            return None
        return (kind, entry_index)

    def set_clipboard_state(self, ready: bool, reason: str = "") -> None:
        """Whether Paste has something to paste (GH #28's button). `reason`
        says why a held clipboard cannot be pasted here (GH #3's version gate)."""
        self._clipboard_ready = ready
        self.trigger_paste_button.setToolTip(reason or self._PASTE_TOOLTIP)
        self._update_buttons()

    def _request_trigger_op(self, op: str) -> None:
        if not self._editable:
            return
        if op == "new":
            self._on_trigger_structural(op, ())
            return
        if op == "paste":
            # The anchor, not a selection: the block lands below it.
            current = self.current_trigger_index()
            self._on_trigger_structural(op, [] if current is None else [current])
            return
        indices = self.selected_trigger_indices()
        if not indices:
            return
        self._on_trigger_structural(op, indices)

    # -- variables -----------------------------------------------------------

    def variable_rows(self) -> list[tuple[int, str]]:
        """(variable_id, name) for the open document, or [] when there is none.

        Read straight back out of the manager rather than cached, for the same
        reason show_scenario() re-parses: the library's field gating is
        process-global, so a manager held across another file's open is only
        safe to read after a re-parse.
        """
        if self._loaded is None:
            return []
        manager = parse_triggers(self._loaded)
        if manager is None:
            return []
        return [(v.variable_id, v.name or "") for v in manager.variables]

    def _open_variables(self) -> None:
        if self._loaded is None:
            return
        if self._variables_dialog is None:
            self._variables_dialog = VariablesDialog(self, on_structural=self._on_variable_structural)
        self.refresh_variables()
        self._variables_dialog.show()
        self._variables_dialog.raise_()
        self._variables_dialog.activateWindow()

    def refresh_variables(self) -> None:
        """Repopulate the variables dialog if one is open. No-op otherwise, so
        the window's edit funnel can call it unconditionally."""
        if self._variables_dialog is None:
            return
        self._variables_dialog.set_variables(self.variable_rows(), editable=self._editable)

    def _request_entry_op(self, op: str) -> None:
        if not self._editable:
            return
        trigger_index = self.current_trigger_index()
        if trigger_index is None:
            return
        if op == "new":
            self._open_picker(trigger_index)
            return
        entries = self._entries_for(self.selected_entry_refs())
        if op == "retype":
            # One entry only: the picker offers that entry's own kind.
            if len(entries) != 1:
                return
            kind, entry_index, entry = entries[0]
            _, _, type_attribute = self._vocab_for(kind)
            self._open_picker(
                trigger_index,
                entry_ref=(kind, entry_index),
                entry_type=self._read(entry, type_attribute),
            )
            return
        if not entries:
            return
        # The whole set, as one undo record (GH #60).
        refs = [(kind, entry_index) for kind, entry_index, _entry in entries]
        self._on_entry_structural(op, trigger_index, refs[0][0], refs, -1)

    # -- the vocabulary picker -----------------------------------------------

    def _open_picker(
        self,
        trigger_index: int,
        entry_ref: tuple[str, int] | None = None,
        entry_type: int = -1,
    ) -> None:
        """Show the vocabulary picker over the lower pane.

        `entry_ref` switches it from Add to Change: the picker then offers only
        that entry's own kind and opens on the type it already has.
        """
        if self._vocabulary is None:
            return
        self._picker_trigger_index = trigger_index
        self._picker_entry_ref = entry_ref
        self._picker_entry_type = entry_type if isinstance(entry_type, int) else -1
        self._populate_picker()
        retyping = self._picker_entry_ref is not None
        self.picker_add_button.setText("Change" if retyping else "Add")
        self.picker_tree.setHeaderLabels(
            [f"Change this {self._picker_entry_ref[0]} to"]
            if retyping
            else ["Add to this trigger"]
        )
        self.picker_filter.setPlaceholderText(
            f"Filter {self._picker_entry_ref[0]}s…"
            if retyping
            else "Filter conditions and effects…"
        )
        self.detail_stack.setCurrentIndex(self._DETAIL_PICKER)
        # The picker gets the whole lower pane while it is up. Sharing it with
        # the detail tree left 144 types in five visible rows, and the tree has
        # nothing to offer here anyway: the trigger being added to was latched
        # at open and cannot be changed from it.
        self._set_entry_list_visible(False)
        self.picker_filter.setFocus()
        self._update_buttons()

    def _close_picker(self) -> None:
        """Leave the picker without adding anything.

        Opening and cancelling must be indistinguishable from never having
        opened it: no callback fires from here, so no model is built and no
        record is pushed. commit_trigger_edit() pushes unconditionally, which
        is why the window is only ever told about an accepted pick.
        """
        self._picker_trigger_index = None
        self._picker_entry_ref = None
        self._picker_entry_type = -1
        self.picker_tree.clear()
        self.picker_filter.clear()
        self.detail_stack.setCurrentIndex(self._DETAIL_FORM)
        self._set_entry_list_visible(True)
        self._update_buttons()

    def _set_entry_list_visible(self, visible: bool) -> None:
        self.entry_tree.setVisible(visible)
        for i in range(self.entry_buttons.count()):
            widget = self.entry_buttons.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(visible)

    def _populate_picker(self) -> None:
        """Both kinds in Add mode; only the latched entry's own kind in Change
        mode, where it also opens on the type the entry already has. Retyping a
        condition into an effect is not offered: they are separate lists."""
        self.picker_tree.clear()
        if self._vocabulary is None:
            return
        groups = (("condition", "Conditions"), ("effect", "Effects"))
        if self._picker_entry_ref is not None:
            groups = tuple(g for g in groups if g[0] == self._picker_entry_ref[0])
        current_item = None
        for kind, label in groups:
            entries, _presentation, _type_attribute = self._vocab_for(kind)
            if entries is None:
                continue
            # Type 0 is "none", the library's placeholder. The in-game editor
            # offers neither of them -- docs/INGAME_EDITOR_REFERENCE.md's own
            # traversal lists 40 conditions and 100 effects, and "None" is in
            # neither -- and it sits first in an alphabetical list, one click
            # from the top, so offering it is all downside.
            offered = [entry for entry in entries.values() if entry.id != 0]
            group = QTreeWidgetItem([f"{label} ({len(offered)})"])
            self.picker_tree.addTopLevelItem(group)
            # Alphabetical rather than by type id: the in-game editor's own New
            # Condition and New Effect lists are alphabetical, and an id order
            # is meaningless to a scenario designer.
            for entry in sorted(offered, key=lambda e: e.name):
                child = QTreeWidgetItem([entry.name.replace("_", " ")])
                child.setData(0, Qt.UserRole, (kind, entry.id))
                group.addChild(child)
                if (
                    self._picker_entry_ref is not None
                    and entry.id == self._picker_entry_type
                ):
                    current_item = child
            group.setExpanded(True)
        # After the tree is built: setCurrentItem on a child whose group has not
        # been added yet does nothing.
        if current_item is not None:
            self.picker_tree.setCurrentItem(current_item)
            self.picker_tree.scrollToItem(current_item)

    def _apply_picker_filter(self, text: str) -> None:
        """Hide non-matching rows, and any group left with nothing under it."""
        needle = text.strip().lower()
        for i in range(self.picker_tree.topLevelItemCount()):
            group = self.picker_tree.topLevelItem(i)
            shown = 0
            for c in range(group.childCount()):
                child = group.child(c)
                hidden = bool(needle) and needle not in child.text(0).lower()
                child.setHidden(hidden)
                shown += not hidden
            group.setHidden(bool(needle) and shown == 0)

    def _picked(self) -> tuple[str, int] | None:
        item = self.picker_tree.currentItem()
        if item is None or item.isHidden():
            return None
        data = item.data(0, Qt.UserRole)
        if data is None:  # a group heading
            return None
        return data

    def _accept_pick(self) -> None:
        picked = self._picked()
        trigger_index = self._picker_trigger_index
        entry_ref = self._picker_entry_ref
        entry_type = self._picker_entry_type
        if picked is None or trigger_index is None:
            return
        kind, type_id = picked
        self._close_picker()
        if entry_ref is None:
            self._on_entry_structural("new", trigger_index, kind, -1, type_id)
            return
        if type_id == entry_type:
            # The only place a no-op retype can be stopped: commit_trigger_edit()
            # pushes unconditionally, so calling through would record an undo
            # step that changes nothing.
            return
        self._on_entry_structural("retype", trigger_index, kind, entry_ref[1], type_id)

    # -- button enablement ---------------------------------------------------

    def _update_buttons(self) -> None:
        """Every button's enabled state, in one place.

        Gated on _editable, which is the panel-populate-time read of
        trigger_write_supported. A read-only file browses exactly as before.
        """
        picking = self.detail_stack.currentIndex() == self._DETAIL_PICKER
        has_document = self._loaded is not None
        selected = len(self.selected_trigger_indices())
        # The trigger verbs act on the selection set. The entry verbs act on
        # the current trigger, so only while it is the one selected.
        has_trigger = self.current_trigger_index() is not None and selected <= 1
        can_edit = self._editable and has_document and not picking

        self.trigger_new_button.setEnabled(can_edit)
        self.trigger_copy_button.setEnabled(can_edit and selected >= 1)
        self.trigger_paste_button.setEnabled(can_edit and self._clipboard_ready)
        self.trigger_delete_button.setEnabled(can_edit and selected >= 1)
        self._update_reorder_buttons(can_edit and selected >= 1)
        self._update_section_buttons(can_edit)

        self.entry_new_button.setEnabled(can_edit and has_trigger and self._vocabulary is not None)
        # Copy and Delete act on the selection set (GH #60); Type… stays single.
        entries = len(self.selected_entry_refs()) if has_trigger else 0
        self.entry_copy_button.setEnabled(can_edit and entries >= 1)
        self.entry_delete_button.setEnabled(can_edit and entries >= 1)
        self.entry_retype_button.setEnabled(
            can_edit and entries == 1 and self._vocabulary is not None
        )

        self.picker_add_button.setEnabled(picking and self._picked() is not None)

        has_tag = self.tag_combo.currentData() is not None
        self.tag_rename_button.setEnabled(can_edit and has_tag)
        self.tag_remove_action.setEnabled(can_edit and has_tag)
        self.tag_rename_button.setToolTip(
            self._TAG_RENAME_TIP if has_tag else self._TAG_PICK_TIP
        )

        # On has_document, not on can_edit: the variable list is worth reading
        # on a read-only file, and the dialog gates its own Add/Remove on the
        # editable flag it is handed.
        self.variables_button.setEnabled(has_document)

    _MOVE_UP_TIP = "Move the selected trigger up in display order"
    _MOVE_DOWN_TIP = "Move the selected trigger down in display order"

    def _update_reorder_buttons(self, base_ok: bool) -> None:
        """Move Up/Down are display-order-only edits (moved_display_order(),
        never reorder_triggers()), so they only make sense -- and are only
        offered -- while the panel actually shows that order, unobstructed:

        - sorted by File order, moving would relocate the trigger somewhere
          the user cannot currently see, since the tree isn't showing display
          position at all;
        - filtered by text or by the 4c tag facet, both hide-only, so the
          adjacent display slot may be a hidden row and a swap into it would
          look like nothing happened.

        Bounds come from `_display_position`/`_display_slots` (adjacency in
        display order), not tree geometry: grouped, the adjacent row in
        `self.tree` may belong to a different section's top-level item, where
        `indexOfTopLevelItem` would misread it as -1 or as an unrelated row.

        Both disable with a tooltip saying why, rather than silently clearing
        the sort/filter for the user.
        """
        sorted_by_display = self._sort_mode == "display"
        unfiltered = not self.filter_edit.text().strip() and self.tag_combo.currentData() is None
        can_reorder = base_ok and sorted_by_display and unfiltered

        # Block bounds: the leading member must have a slot above it, the
        # trailing one a slot below.
        positions = [
            self._display_position[index]
            for index in self.selected_trigger_indices()
            if index in self._display_position
        ]
        count = len(self._display_slots)

        self.trigger_move_up_button.setEnabled(can_reorder and bool(positions) and min(positions) > 0)
        self.trigger_move_down_button.setEnabled(
            can_reorder and bool(positions) and max(positions) < count - 1
        )

        if sorted_by_display and unfiltered:
            reason = ""
        elif not sorted_by_display:
            reason = " (switch to Display order to reorder)"
        else:
            reason = " (clear the filter or tag to reorder)"
        self.trigger_move_up_button.setToolTip(self._MOVE_UP_TIP + reason)
        self.trigger_move_down_button.setToolTip(self._MOVE_DOWN_TIP + reason)

    # -- shared helpers ------------------------------------------------------

    @staticmethod
    def _read(entry, attribute: str):
        """getattr that tolerates the library's version-gated properties.

        Those raise UnsupportedAttributeError rather than being absent. A
        depoison() should have cleared the gating before we get here, so this
        is belt-and-braces: a browser must never fail to render a row over one
        unreadable field.
        """
        try:
            return getattr(entry, attribute, None)
        except Exception:  # noqa: BLE001 -- see docstring: a row must always render
            return None

    def _apply_filter(self, *_args) -> None:
        """Hide-only, on two composed predicates: the free-text filter (the
        rendered label, unchanged from pre-4c behaviour) and the tag facet
        (the stashed _TAG_ROLE, never the label -- see that constant's
        docstring). `*_args` swallows whichever signal fired this (str from
        filter_edit.textChanged, int from tag_combo.currentIndexChanged).
        """
        needle = self.filter_edit.text().strip().lower()
        wanted_tag = self.tag_combo.currentData()
        if self._grouped:
            self._apply_filter_grouped(needle, wanted_tag)
        else:
            self._apply_filter_flat(needle, wanted_tag)

    def _row_matches(self, item: QTreeWidgetItem, needle: str, wanted_tag) -> bool:
        if needle and needle not in item.text(self._COL_NAME).lower():
            return False
        # Kept as parallel guard clauses; negating only the second one into the
        # return would read as if it were the sole test.
        if wanted_tag is not None and item.data(self._ROLE_COL, self._TAG_ROLE) != wanted_tag:  # noqa: SIM103
            return False
        return True

    def _apply_filter_flat(self, needle: str, wanted_tag) -> None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setHidden(not self._row_matches(item, needle, wanted_tag))

    def _apply_filter_grouped(self, needle: str, wanted_tag) -> None:
        """A section stays visible if its own header trigger matches, or any
        member does -- same "hide the group only if nothing under it matches"
        rule as the vocabulary picker's _apply_picker_filter()."""
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            is_real_header = top.data(self._ROLE_COL, Qt.UserRole) is not None
            header_matches = is_real_header and self._row_matches(top, needle, wanted_tag)
            shown = 0
            for c in range(top.childCount()):
                child = top.child(c)
                matches = self._row_matches(child, needle, wanted_tag)
                child.setHidden(not matches)
                shown += matches
            top.setHidden(not (header_matches or shown))

        # A collapsed section would otherwise swallow its own matches. The flag
        # goes up before the first expand and down after the last collapse, so
        # _on_section_toggled() never records either pass.
        if needle or wanted_tag is not None:
            self._filter_expanded = True
            for i in range(self.tree.topLevelItemCount()):
                top = self.tree.topLevelItem(i)
                if not top.isHidden():
                    top.setExpanded(True)
        elif self._filter_expanded:
            for position in range(min(len(self._sections), self.tree.topLevelItemCount())):
                key = trigger_organize.section_key(self._sections, position)
                self.tree.topLevelItem(position).setExpanded(key not in self._collapsed_keys)
            self._filter_expanded = False
