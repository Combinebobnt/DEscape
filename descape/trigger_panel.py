"""TriggerPanel: the trigger browser and property editor.

TriggerPanel is this module's class plus eight mixins, one module per area:

- trigger_panel.py: the class constants, __init__, the widget tree's
  construction, document state (clear_document, show_scenario, the flat and
  grouped populates), changeEvent and resizeEvent, and the shared filter
  helpers; the trigger tree and its inline-rename delegate.
- trigger_panel_organize.py: the tag combo and tag requests, section
  navigation and collapse state, and section management.
- trigger_panel_list.py: selection preservation and the trigger list.
- trigger_panel_status.py: StatusStyle, status icons and tooltips, and the
  status colours (GH #166).
- trigger_panel_form.py: the property form and reporting an edit.
- trigger_panel_fields.py: Unit and Unit[] references, map picks, the
  coordinate groups and the catalog, enum and list-type pickers.
- trigger_panel_structure.py: structural editing, context menus and the
  vocabulary picker.
- trigger_panel_variables.py: VariablesDialog and the panel's side of it.
- trigger_panel_buttons.py: button enablement and the move tips.

Rules of the split (tests/test_trigger_panel_split.py checks them):

- A mixin defines no __init__, no Qt event override and no signal: all
  instance state is created in TriggerPanel.__init__.
- No name is defined by two of the nine classes.
- The mixins precede QWidget in TriggerPanel's bases, and no mixin module
  imports this one.

A branch cut before the split is ported with tools/split_class.py's
`split --like`."""

from __future__ import annotations

from typing import ClassVar

from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QAction,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from descape import (
    library_compat,
    player_labels,
    settings,
    trigger_organize,
    trigger_status,
)
from descape.button_grid import ButtonGrid
from descape.facet_preview import FacetPreview
from descape.instruction_preview import InstructionPreview
from descape.scenario_io import (
    LoadedScenario,
    parse_triggers,
    repo_version_has_triggers,
    unsupported_version_sentence,
)
from descape.trigger_model import (
    EXEC_MODE_DISPLAY,
    EXEC_MODE_LEGACY,
    EXEC_MODE_UNKNOWN,
    exec_order_value,
    resolve_exec_mode,
)
from descape.trigger_panel_buttons import _TriggerPanelButtonsMixin
from descape.trigger_panel_fields import _GroupRow, _TriggerPanelFieldsMixin
from descape.trigger_panel_form import _TriggerPanelFormMixin
from descape.trigger_panel_list import _TriggerPanelListMixin
from descape.trigger_panel_organize import _TriggerPanelOrganizeMixin
from descape.trigger_panel_status import StatusStyle, _StatusMemo, _TriggerPanelStatusMixin
from descape.trigger_panel_structure import _MoveButton, _TriggerPanelStructureMixin
from descape.trigger_panel_variables import VariablesDialog, _TriggerPanelVariablesMixin
from descape.trigger_status import Status
from descape.value_picker import ValueLineEdit, _HScrollStableTreeWidget
from descape.viewer_common import (
    FontScaledWidth,
    LabelFitButton,
)


class _TriggerTree(_HScrollStableTreeWidget):
    """The trigger list. With the tree focused, Qt's own Ctrl+A calls
    selectAll() before the window's Select All shortcut sees the key, and
    Qt's version skips collapsed sections, so it routes to the panel's."""

    editor_closed = pyqtSignal()

    def __init__(self, select_all) -> None:
        super().__init__()
        self._select_all = select_all

    def selectAll(self) -> None:
        self._select_all()

    def closeEditor(self, editor, hint) -> None:
        # A current-row change lands here directly, past the delegate's closeEditor signal.
        super().closeEditor(editor, hint)
        self.editor_closed.emit()


class _TriggerNameDelegate(QStyledItemDelegate):
    """The trigger list's inline rename (GH #279). A row shows a label, not the
    name, so the editor seeds from `name_of(trigger index)`. Nothing writes the
    model: setModelData() stashes `pending` and the panel applies it on
    closeEditor, once a repartition's tree.clear() can't reach a live editor."""

    def __init__(self, name_of, role_column: int, parent) -> None:
        super().__init__(parent)
        self._name_of = name_of
        self._role_column = role_column
        self.pending: tuple[int, str] | None = None

    def _trigger_index(self, index) -> int:
        return index.sibling(index.row(), self._role_column).data(Qt.UserRole)

    def setEditorData(self, editor, index) -> None:
        editor.setText(self._name_of(self._trigger_index(index)))

    def setModelData(self, editor, model, index) -> None:
        self.pending = (self._trigger_index(index), editor.text())

    def eventFilter(self, editor, event) -> bool:
        # Commit and stay: unhandled, these keys move the current row, and Tab opens an editor there.
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Up, Qt.Key_Down):
            self.commitData.emit(editor)
            self.closeEditor.emit(editor, QAbstractItemDelegate.NoHint)
            return True
        return super().eventFilter(editor, event)


class TriggerPanel(
    _TriggerPanelOrganizeMixin, _TriggerPanelListMixin,
    _TriggerPanelStatusMixin, _TriggerPanelFormMixin, _TriggerPanelFieldsMixin,
    _TriggerPanelStructureMixin, _TriggerPanelVariablesMixin,
    _TriggerPanelButtonsMixin, QWidget,
):
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
    form for whichever of them is selected, with a second bar between the
    two (detail_splitter, GH #274).

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

    # The detail column never gets less than this much of the entry tree's
    # viewport, however long the widest condition name is. Content-sizing the
    # Item column alone would hand a long name the whole pane and push Detail
    # off the right edge, which is what its old fixed width was working around.
    _MIN_DETAIL_WIDTH = 120

    # Each pane's starting weight in both vertical splits. Only the ratio counts, but
    # QSplitter.setSizes() raises a value below a pane's minimum to it, which would skew it.
    _SPLIT_WEIGHT = 10_000

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
    _PASTE_TOOLTIP = (
        "Paste the copied triggers below the current one, or below a collapsed section's end "
        "(Copy or Edit > Copy Triggers copies them)"
    )

    # The only synthetic row in the tree (4c): triggers before the first
    # divider. Never a real trigger -- see current_trigger_index()'s Qt.UserRole
    # sentinel below.
    _BEFORE_FIRST_SECTION = "(before the first section)"

    # A row's tag_chain() as a tuple (() when untagged), stashed at populate
    # time. The tag combo matches membership in this, never the name column's
    # text: the rendered label substitutes "(unnamed)" and appends "  (disabled)", and
    # inheriting that false-positive class into a facet would be far more
    # visible than it already is in the free-text filter.
    _TAG_ROLE = Qt.UserRole + 1
    # A vocabulary picker group heading's kind ("condition"/"effect").
    _PICKER_KIND_ROLE = Qt.UserRole + 2
    # GH #166: a row's trigger_status.EntryStatus, on the column it styles, so
    # apply_status_style() can restyle without a recompute.
    _STATUS_ROLE = Qt.UserRole + 3
    # TASK-317: beside _STATUS_ROLE, whether the row is a section divider's, for the same restyle.
    _DIVIDER_ROLE = Qt.UserRole + 4
    _STATUS_HEADER_TIP = (
        "Trigger status: a tick when complete, a cross when something is missing, "
        "a dash for a section with no conditions or effects"
    )
    _EMPTY_SECTION_TIP = "Section with no conditions or effects"

    # The trigger tree's columns. Detail, picker and variables trees keep
    # their own indices.
    _COL_POS = 0
    _COL_ID = 1
    _COL_STATUS = 2  # TASK-317: the status icon, hidden while the marker shows no icons
    _COL_NAME = 3
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
        on_copy_triggers=None,
        on_tag_add=None,
        on_tag_remove_selected=None,
        on_cut_triggers=None,
        on_copy_entries=None,
        on_entry_field_group=None,
        on_entry_move=None,
    ):
        super().__init__()
        # Callbacks, exactly as MapView takes them. All are no-ops by default
        # so the panel stays constructible on its own for screenshot tests.
        self._on_trigger_field = on_trigger_field or (lambda *args: None)
        self._on_entry_field = on_entry_field or (lambda *args: None)
        # GH #138: (trigger_index, ref, {field: value}, label, defer_refresh), one record per group.
        self._on_entry_field_group = on_entry_field_group or (lambda *args: None)
        self._on_trigger_structural = on_trigger_structural or (lambda *args: None)
        self._on_entry_structural = on_entry_structural or (lambda *args, **kwargs: None)
        # GH #213: (trigger_index, refs, delta[, to_edge]), a display-order move of the selected entries.
        self._on_entry_move = on_entry_move or (lambda *args: None)
        self._on_variable_structural = on_variable_structural or (lambda *args: None)
        # Fired when the trigger selection set changes, so the window can
        # re-gate actions that follow it (Edit > Copy Triggers).
        self._on_selection_changed = on_selection_changed or (lambda *args: None)
        self._on_tag_rename = on_tag_rename or (lambda *args: None)
        self._on_tag_remove = on_tag_remove or (lambda *args: None)
        # GH #27/#28's Copy button: Edit > Copy Triggers, not Duplicate.
        self._on_copy_triggers = on_copy_triggers or (lambda *args: None)
        # GH #137's Cut: the window reads the selection, like Copy.
        self._on_cut_triggers = on_cut_triggers or (lambda *args: None)
        # GH #137's entry clipboard Copy: read-only, the window reads the selection.
        self._on_copy_entries = on_copy_entries or (lambda *args: None)
        # GH #101: (indices, tag) for the Tags menu's Add and Remove on the selection.
        self._on_tag_add = on_tag_add or (lambda *args: None)
        self._on_tag_remove_selected = on_tag_remove_selected or (lambda *args: None)
        # GH #41: fired after each entry-form sync, which a trigger change reaches too. Assigned by the window.
        self.on_focus_changed = lambda: None
        # TASK-550: the selected Display Instructions effect's InstructionView, or None. Assigned by the window.
        self.on_instruction_view = lambda view: None
        # Unit/Unit[] fields: placed-unit id -> label text, and Pick from map's
        # arm (a target tuple) or disarm (None) request. Assigned by the window.
        self.describe_unit_reference = lambda ref_id: ""
        self.on_pick_unit = lambda target: None
        # GH #138: a placed unit's own tile or None (Go to Objects' gate), and
        # Go to's request, (trigger_index, kind, entry_index, group). Assigned by the window.
        self.unit_reference_tile = lambda ref_id: None
        self.on_go_to_group = lambda target: None
        # GH #139: a text box's stored height in lines by key, and its store (None resets). Assigned by the window.
        self.text_box_lines = lambda key: None
        self.on_text_box_lines = lambda key, lines: None
        # GH #166: (vocabulary, trigger_ids) -> trigger_status.StatusContext, called on every
        # recompute (the window can drop and rebuild its reference index), and the marker style.
        # Assigned by the window; these defaults treat every unit id as placed.
        self.status_context = trigger_status.StatusContext
        self.status_style = StatusStyle
        self._status_style = StatusStyle()
        self._status_icons: dict[Status, QIcon] = {}
        self._status_restyle_pending = False
        # Per populate: trigger index -> its statuses, and the triggers whose entries hold placed-unit ids.
        self._status: dict[int, trigger_status.TriggerStatuses] = {}
        self._unit_ref_triggers: set[int] = set()
        # A unit edit outside Triggers mode skipped its recompute; the next show_scenario() does it.
        self._status_stale = False
        # The trigger indices that may differ from the parsed file, or None for "unknown, recompute
        # every trigger". Assigned by the window (TriggerEditModel.dirty_indices()).
        self.edited_trigger_indices = lambda: None
        # id(trigger) -> its memo, and the vocabulary and trigger count it was computed under.
        self._status_memo: dict[int, _StatusMemo] = {}
        self._status_memo_vocabulary = None
        self._status_memo_count = -1
        self._armed_pick: tuple | None = None
        # Per populate: field -> (spec, refs, editor, refresh) and target -> Pick button.
        self._unit_ref_rows: dict[str, tuple] = {}
        self._pick_buttons: dict[tuple, QToolButton] = {}
        # GH #276, per populate: field -> the object row an eyedropper writes through.
        self._type_pick_rows: dict[str, ValueLineEdit] = {}
        # GH #138, per populate: group -> its Set / Go to / Reset row. Never in _rows, whose shape others read.
        self._group_rows: dict[str, _GroupRow] = {}
        # Whether the window holds a pasteable trigger clipboard. Pushed in
        # through set_clipboard_state(); the panel never reads the window.
        self._clipboard_ready = False
        # Why a held block cannot be pasted here, or "" (the context menu's Paste tooltip).
        self._clipboard_reason = ""
        # The same pair for the entry clipboard, a separate slot (GH #137).
        self._entry_clipboard_ready = False
        self._entry_clipboard_reason = ""

        # Built on first open and then kept, so a variables edit can refresh a
        # dialog that is still showing. Closed and dropped by clear_document().
        self._variables_dialog: VariablesDialog | None = None
        self.variable_actions: dict[str, QAction] = {}
        for op in ("add", "rename", "remove"):
            action_id = f"variable_{op}"
            action = QAction(settings.get_action_label(action_id), self)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.setAutoRepeat(False)
            action.setEnabled(False)
            action.triggered.connect(lambda _=False, o=op: self._dispatch_variable_action(o))
            self.variable_actions[action_id] = action

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
        # Add mode's kind, from the detail tree's highlighted Conditions or
        # Effects row: the other group opens collapsed (GH #136). None: both open.
        self._picker_focus_kind: str | None = None

        # True while widgets are being populated programmatically. Qt fires
        # valueChanged/currentIndexChanged on a programmatic set exactly as it
        # does on a user edit, so without this, selecting a row would record one
        # undo step per field and dirty the trigger. See _changed().
        self._populating = False
        self._loaded: LoadedScenario | None = None
        self._vocabulary = None
        self._editable = False
        # GH #130: PlayerId combos' labels and swatches, pushed by refresh_player_labels().
        self._player_labels = player_labels.DEFAULT_LABELS
        self._player_colors = None
        # GH #140: the form's Display Instructions preview, or None.
        self._instruction_preview: InstructionPreview | None = None
        # GH #167: the facet-direction preview and the field it sits under, or None and "".
        self._facet_preview: FacetPreview | None = None
        self._facet_preview_field = ""
        self._button_grid: ButtonGrid | None = None
        self._rows: list[tuple] = []
        # GH #41: field name -> object list type last chosen, kept across repopulates of one selection.
        self._list_types: dict[str, str] = {}
        self._list_types_refs: tuple | None = None
        # The (kind, index) refs the property form was last built for. None
        # means "holds nothing", so the next sync always repopulates.
        self._form_refs: tuple[tuple[str, int], ...] | None = None
        # (kind, list index) -> entry-tree row, rebuilt per populate: rows sit in display order.
        self._entry_items: dict[tuple[str, int], QTreeWidgetItem] = {}
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
        # The trigger count at the last populate: File order view's move bounds (TASK-569).
        self._trigger_count = 0
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
        # Equal halves to start with. Left alone, Qt splits by the two size hints (about
        # 40:60), so a button row taken out of the tree pane gave the tree half its height.
        self.splitter.setSizes([self._SPLIT_WEIGHT, self._SPLIT_WEIGHT])

        # Below the splitter rather than in either pane's button rows: a
        # variable belongs to the document, not to the selected trigger or
        # entry, and both of those rows are already full at 340 px.
        self.variables_button = QPushButton("Variables…")
        self.variables_button.setToolTip("Add or remove this scenario's trigger variables")
        self.variables_button.clicked.connect(lambda checked=False: self._open_variables())
        self.variables_action = QAction(settings.get_action_label("trigger_variables"), self)
        self.variables_action.triggered.connect(self.variables_button.click)
        self.addAction(self.variables_action)
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
        # One menu holds the four tag verbs (GH #101's two act on the selection).
        # The other two act on the facet's tag whatever the text filter hides.
        tag_row = QHBoxLayout()
        tag_row.setContentsMargins(0, 0, 0, 0)
        tag_row.addWidget(self.tag_combo, stretch=1)
        self.tags_button = QToolButton()
        self.tags_button.setText("Tags")
        self.tags_button.setToolTip(
            "Tag or untag the selected triggers; rename or remove the filter's tag on every trigger"
        )
        self.tags_button.setPopupMode(QToolButton.InstantPopup)
        self.tags_menu = QMenu(self.tags_button)
        # The disabled reasons live in the action tooltips.
        self.tags_menu.setToolTipsVisible(True)
        self.tag_add_action = self.tags_menu.addAction("Add Tag to Selection…")
        self.tag_add_action.triggered.connect(lambda checked=False: self.request_tag_add())
        self.tag_remove_selected_action = self.tags_menu.addAction("Remove Tag from Selection…")
        self.tag_remove_selected_action.triggered.connect(lambda checked=False: self.request_tag_remove_selected())
        self.tags_menu.addSeparator()
        self.tag_rename_action = self.tags_menu.addAction("Rename Tag…")
        self.tag_rename_action.triggered.connect(lambda checked=False: self.request_tag_rename())
        self.tag_remove_action = self.tags_menu.addAction("Remove Tag Everywhere")
        self.tag_remove_action.triggered.connect(lambda checked=False: self.request_tag_remove())
        self.tags_button.setMenu(self.tags_menu)
        tag_row.addWidget(self.tags_button)

        # The section verbs, Collapse All / Expand All, then one jump per
        # section. The jump actions are rebuilt per show_scenario().
        self.sections_button = QToolButton()
        self.sections_button.setText("Sections")
        self.sections_button.setToolTip("Add, rename or move to a section; collapse, expand or jump to one")
        self.sections_button.setPopupMode(QToolButton.InstantPopup)
        self.sections_menu = QMenu(self.sections_button)
        # The disabled-reason suffixes live in the action tooltips.
        self.sections_menu.setToolTipsVisible(True)
        self.section_new_action = self.sections_menu.addAction("New Section…")
        self.section_new_action.triggered.connect(lambda checked=False: self.request_new_section())
        self.section_rename_action = self.sections_menu.addAction("Rename Section…")
        self.section_rename_action.triggered.connect(lambda checked=False: self.request_section_rename())
        self.section_move_menu = self.sections_menu.addMenu("Move to Section")
        self.section_move_menu.aboutToShow.connect(self._populate_move_menu)
        self.section_move_action = self.section_move_menu.menuAction()
        self.sections_menu.addSeparator()
        self.collapse_all_action = self.sections_menu.addAction("Collapse All")
        self.collapse_all_action.triggered.connect(lambda checked=False: self.collapse_all_sections())
        self.expand_all_action = self.sections_menu.addAction("Expand All")
        self.expand_all_action.triggered.connect(lambda checked=False: self.expand_all_sections())
        self.sections_menu.addSeparator()
        self.sections_menu.aboutToShow.connect(self._update_section_actions)
        self._section_actions: list = []
        self.sections_button.setMenu(self.sections_menu)
        # On the tag row, always shown: beside the filter at 340 px it left the text box 45 px.
        tag_row.addWidget(self.sections_button)

        self.sort_combo = QComboBox()
        for mode_id, label in self._SORT_MODES:
            self.sort_combo.addItem(label, mode_id)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_mode_changed)
        filter_row.addWidget(self.sort_combo)
        pane_layout.addLayout(filter_row)
        pane_layout.addLayout(tag_row)

        self.tree = _TriggerTree(self.select_all)
        self.tree.setColumnCount(4)
        self._set_position_header(EXEC_MODE_UNKNOWN)
        self.tree.setColumnHidden(self._COL_STATUS, not self._status_style.icons)
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(False)
        self._configure_scrolling(self.tree)
        # All content-sized, unlike the detail tree's capped Item/Detail
        # split: position and ID are a few digits wide, so giving Trigger the
        # same ResizeToContents treatment still leaves it effectively the whole
        # pane. No _fit_entry_columns()-style cap and no resizeEvent hook --
        # the sizing here has nothing viewport-dependent to react to.
        for column in (self._COL_POS, self._COL_ID, self._COL_STATUS, self._COL_NAME):
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
        # GH #279. NoEditTriggers, or the rows' editable flag makes any typed key open an
        # editor and loses keyboard search; only request_trigger_rename() opens one.
        self.tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._name_delegate = _TriggerNameDelegate(self._name_editor_text, self._ROLE_COL, self.tree)
        self.tree.setItemDelegateForColumn(self._COL_NAME, self._name_delegate)
        # After the line above, so the view's own closeEditor slot has dropped the editor first.
        self._name_delegate.closeEditor.connect(self._apply_trigger_rename)
        # Queued: a repartition's tree.clear() can't run inside the view's currentChanged().
        self.tree.editor_closed.connect(self._apply_trigger_rename, Qt.QueuedConnection)
        self.tree.itemDoubleClicked.connect(self.request_trigger_rename)
        pane_layout.addWidget(self.tree, stretch=1)

        self.trigger_buttons = self._build_button_row(
            [
                ("trigger_new_button", "New", "Add a new trigger at the end of the list"),
                (
                    "trigger_clipboard_copy_button",
                    "Copy",
                    "Copy the selected triggers to the clipboard (Edit > Copy Triggers; a collapsed section copies whole)",
                ),
                ("trigger_paste_button", "Paste", self._PASTE_TOOLTIP),
                ("trigger_delete_button", "Delete", "Remove the selected triggers (a collapsed section deletes whole)"),
            ],
            lambda op: (lambda checked=False: self._request_trigger_op(op)),
            ("new", "clipboard_copy", "paste", "delete"),
        )
        pane_layout.addLayout(self.trigger_buttons)

        # Glyph tool buttons beside Duplicate: with full labels this row needs ~560 px.
        self.reorder_buttons = QHBoxLayout()
        self.reorder_buttons.setContentsMargins(0, 0, 0, 0)
        self.trigger_copy_button = QPushButton("Duplicate")
        self.trigger_copy_button.setToolTip(
            "Duplicate the selected triggers in place (a collapsed section duplicates whole). "
            "Their trigger links still point at the originals; Copy + Paste re-links a block"
        )
        self.trigger_copy_button.clicked.connect(lambda checked=False: self._request_trigger_op("copy"))
        self.reorder_buttons.addWidget(self.trigger_copy_button, stretch=1)
        for attribute, glyph, name, tip, op in (
            ("trigger_move_up_button", "▲", "Move Up", self._MOVE_UP_TIP, "move_up"),
            ("trigger_move_down_button", "▼", "Move Down", self._MOVE_DOWN_TIP, "move_down"),
            ("section_up_button", "⇈", "Section Up", self._SECTION_UP_TIP, "section_up"),
            ("section_down_button", "⇊", "Section Down", self._SECTION_DOWN_TIP, "section_down"),
        ):
            if op == "section_up":
                self.reorder_buttons.addSpacing(6)
            # GH #211: ▲▼ read Shift and Ctrl off their own click; ⇈⇊ have one distance.
            is_move = op in ("move_up", "move_down")
            button = _MoveButton() if is_move else QToolButton()
            button.setText(glyph)
            # The glyph alone is no name for AT-SPI locators or screen readers.
            button.setAccessibleName(name)
            button.setToolTip(tip)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            if is_move:
                button.clicked.connect(
                    lambda checked=False, button=button, way=-1 if op == "move_up" else 1: self._move_clicked(
                        button, way, entries=False
                    )
                )
            else:
                button.clicked.connect(lambda checked=False, op=op: self._request_trigger_op(op))
            setattr(self, attribute, button)
            self.reorder_buttons.addWidget(button)
        pane_layout.addLayout(self.reorder_buttons)
        return pane

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

        # GH #274: nested, so self.splitter stays a two-pane split and its drag keeps this ratio.
        self.detail_splitter = QSplitter(Qt.Vertical)
        # A pane collapsed to zero height still passes every isVisibleTo() check (units_panel.py).
        self.detail_splitter.setChildrenCollapsible(False)
        self.detail_splitter.addWidget(self.entry_tree)

        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)

        self.entry_buttons = self._build_button_row(
            [
                ("entry_new_button", "New", "Add a condition or effect to this trigger"),
                ("entry_copy_button", "Duplicate", "Duplicate the selected conditions and effects"),
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
            # GH #213: the four give up Fusion's 80 px floor, which is the room ▲▼ take.
            button_class=LabelFitButton,
        )
        moves = QHBoxLayout()
        moves.setContentsMargins(0, 0, 0, 0)
        moves.setSpacing(2)
        for attribute, glyph, name, tip, op in (
            ("entry_move_up_button", "▲", "Move Entry Up", self._ENTRY_MOVE_UP_TIP, "move_up"),
            ("entry_move_down_button", "▼", "Move Entry Down", self._ENTRY_MOVE_DOWN_TIP, "move_down"),
        ):
            button = _MoveButton()
            button.setText(glyph)
            # Named apart from the trigger row's pair, for AT-SPI locators.
            button.setAccessibleName(name)
            button.setToolTip(tip)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            button.clicked.connect(
                lambda checked=False, button=button, way=-1 if op == "move_up" else 1: self._move_clicked(
                    button, way, entries=True
                )
            )
            setattr(self, attribute, button)
            moves.addWidget(button)
        self.entry_buttons.addLayout(moves)
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

        self.detail_splitter.addWidget(pane)
        self.detail_splitter.setStretchFactor(0, 1)
        self.detail_splitter.setStretchFactor(1, 1)
        self.detail_splitter.setSizes([self._SPLIT_WEIGHT, self._SPLIT_WEIGHT])
        return self.detail_splitter

    def _build_button_row(self, buttons, slot_for, ops, button_class=QPushButton) -> QHBoxLayout:
        """One row of equally-weighted push buttons, stored on self by name."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for (attribute, label, tip), op in zip(buttons, ops, strict=True):
            button = button_class(label)
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
        Item was once pinned at 240 px out of a 340 px pane, so
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
        self._clear_entry_tree()
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
            self._reset_statuses()
            self._item_for_index = {}
            self._display_slots = []
            self._trigger_count = 0
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
            self.close_variables_dialog()
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
            self._list_types_refs = None

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
                self._reset_statuses()
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
            self._trigger_count = len(triggers)
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

            # Before the rows: _make_trigger_item() styles each from the cache.
            self._refresh_status_style()
            self._compute_all_statuses(manager, same_document)
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

    def _set_position_header(self, mode: str) -> None:
        text, tooltip = self._POSITION_HEADERS[mode]
        labels = ["", "ID", "", "Trigger"]
        labels[self._COL_POS] = text
        self.tree.setHeaderLabels(labels)
        self.tree.headerItem().setToolTip(self._COL_POS, tooltip)
        self.tree.headerItem().setToolTip(self._COL_STATUS, self._STATUS_HEADER_TIP)

    def _position_text(self, index: int) -> str:
        position = self._display_position.get(index)
        return "" if position is None else str(position)

    def _make_trigger_item(self, index: int, trigger) -> QTreeWidgetItem:
        columns = ["", "", "", ""]
        columns[self._COL_POS] = self._position_text(index)
        columns[self._COL_ID] = str(index)
        columns[self._COL_NAME] = self._trigger_label(trigger)
        item = QTreeWidgetItem(columns)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setData(self._ROLE_COL, Qt.UserRole, index)
        self._stash_tags(item, self._read(trigger, "name") or "")
        self._style_trigger_item(item, index)
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
                columns = ["", "", "", ""]
                columns[self._COL_NAME] = self._BEFORE_FIRST_SECTION
                top = QTreeWidgetItem(columns)
                top.setData(self._ROLE_COL, Qt.UserRole, None)
                top.setData(self._ROLE_COL, self._TAG_ROLE, ())
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

    def changeEvent(self, event) -> None:
        # A theme change moves the trees' Base, so the row brushes are re-shaded. Deferred and
        # coalesced like ViewerWindow's icon re-ink: apply_theme() sends several.
        if event.type() == QEvent.PaletteChange and not self._status_restyle_pending:
            self._status_restyle_pending = True
            QTimer.singleShot(0, self._restyle_for_palette)
        super().changeEvent(event)

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
        if wanted_tag is not None and wanted_tag not in self._row_tags(item):  # noqa: SIM103
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
