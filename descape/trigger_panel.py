"""TriggerPanel: the trigger browser and property editor, and the
VariablesDialog that hangs off it."""

from __future__ import annotations



from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


from descape import (
    library_compat,
    object_catalog,
    trigger_fields,
    trigger_organize,
)
from descape.constant_picker import CatalogLineEdit
from descape.scenario_io import (
    LoadedScenario,
    parse_triggers,
)
from descape.trigger_model import (
    exec_order_value,
)
from descape.viewer_common import _fit_combo_width, _make_spinbox


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


class _HScrollStableTreeWidget(QTreeWidget):
    """QTreeWidget that ignores Qt's horizontal auto-scroll on selection.

    Qt's default scrollTo(index, EnsureVisible) chases a ResizeToContents
    column's full width, so selecting a long row yanks the horizontal
    scrollbar to that row's far-right edge. Restoring the horizontal
    position after the base implementation runs keeps its vertical
    auto-scroll (needed for keyboard navigation) while dropping the
    horizontal jump, regardless of what triggered the scroll.
    """

    def scrollTo(self, index, hint=QTreeWidget.EnsureVisible) -> None:
        hbar = self.horizontalScrollBar()
        pos = hbar.value()
        super().scrollTo(index, hint)
        hbar.setValue(pos)


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
    # when a name column and a detail column shared a single tree.
    MIN_USEFUL_WIDTH = 340
    _PROPERTY_MIN_WIDTH = 240

    # The detail column never gets less than this much of the entry tree's
    # viewport, however long the widest condition name is. Content-sizing the
    # Item column alone would hand a long name the whole pane and push Detail
    # off the right edge, which is what its old fixed width was working around.
    _MIN_DETAIL_WIDTH = 120

    _NO_DOCUMENT = "No map open."
    _UNSUPPORTED = (
        "This file's Triggers section can't be read.\n\n"
        "Scenario version 1.54 with trigger version 3.9 uses an older trigger "
        "format AoE2ScenarioParser can't parse. Open it in the in-game scenario "
        "editor and re-save it to upgrade the format, then reopen it here.\n\n"
        "Terrain and elevation editing work normally on this file."
    )
    _READ_ONLY_NOTE = "This file's triggers can be read but not written, so editing is off."

    # The only synthetic row in the tree (4c): triggers before the first
    # divider. Never a real trigger -- see current_trigger_index()'s Qt.UserRole
    # sentinel below.
    _BEFORE_FIRST_SECTION = "(before the first section)"

    # A row's parsed [tag] (str | None), stashed at populate time. The tag
    # combo matches against this, never against text(1): the rendered label
    # substitutes "(unnamed)" and appends "  (disabled)", and inheriting that
    # false-positive class into a facet would be far more visible than it
    # already is in the free-text filter.
    _TAG_ROLE = Qt.UserRole + 1

    def __init__(
        self,
        on_trigger_field=None,
        on_entry_field=None,
        on_trigger_structural=None,
        on_entry_structural=None,
        on_variable_structural=None,
    ):
        super().__init__()
        # Callbacks, exactly as MapView takes them. All are no-ops by default
        # so the panel stays constructible on its own for screenshot tests.
        self._on_trigger_field = on_trigger_field or (lambda *args: None)
        self._on_entry_field = on_entry_field or (lambda *args: None)
        self._on_trigger_structural = on_trigger_structural or (lambda *args: None)
        self._on_entry_structural = on_entry_structural or (lambda *args: None)
        self._on_variable_structural = on_variable_structural or (lambda *args: None)

        # Built on first open and then kept, so a variables edit can refresh a
        # dialog that is still showing. Closed and dropped by clear_document().
        self._variables_dialog: VariablesDialog | None = None

        # The trigger the picker was opened against, latched at open rather
        # than read back at accept: the trigger tree stays live while the
        # picker is showing, so a selection change in between would otherwise
        # add the condition to a different trigger than the user asked.
        self._picker_trigger_index: int | None = None

        # True while widgets are being populated programmatically. Qt fires
        # valueChanged/currentIndexChanged on a programmatic set exactly as it
        # does on a user edit, so without this, selecting a row would record one
        # undo step per field and dirty the trigger. See _changed().
        self._populating = False
        self._loaded: LoadedScenario | None = None
        self._vocabulary = None
        self._editable = False
        self._rows: list[tuple] = []
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
        filter_row.addWidget(self.tag_combo)

        self.sort_combo = QComboBox()
        for mode_id, label in self._SORT_MODES:
            self.sort_combo.addItem(label, mode_id)
        self.sort_combo.currentIndexChanged.connect(self._on_sort_mode_changed)
        filter_row.addWidget(self.sort_combo)
        pane_layout.addLayout(filter_row)

        self.tree = _HScrollStableTreeWidget()
        self.tree.setHeaderLabels(["ID", "Trigger"])
        self.tree.setColumnCount(2)
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(False)
        self._configure_scrolling(self.tree)
        # Both content-sized, unlike the detail tree's capped Item/Detail
        # split: ID is a few digits wide, so giving Trigger the same
        # ResizeToContents treatment still leaves it effectively the whole
        # pane. No _fit_entry_columns()-style cap and no resizeEvent hook --
        # the sizing here has nothing viewport-dependent to react to.
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.currentItemChanged.connect(lambda *_: self._on_trigger_selected())
        pane_layout.addWidget(self.tree, stretch=1)

        self.trigger_buttons = self._build_button_row(
            [
                ("trigger_new_button", "New", "Add a new trigger at the end of the list"),
                ("trigger_copy_button", "Copy", "Duplicate the selected trigger"),
                ("trigger_delete_button", "Delete", "Remove the selected trigger"),
            ],
            lambda op: (lambda checked=False: self._request_trigger_op(op)),
            ("new", "copy", "delete"),
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
        return pane

    def _on_sort_mode_changed(self, combo_index: int) -> None:
        if self._populating:
            return
        mode = self.sort_combo.itemData(combo_index)
        if mode == self._sort_mode:
            return
        selected = self.current_trigger_index()
        self._sort_mode = mode
        self.show_scenario(self._loaded)
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
        self.entry_tree.currentItemChanged.connect(lambda *_: self._on_entry_selected())
        pane_layout.addWidget(self.entry_tree, stretch=1)

        self.entry_buttons = self._build_button_row(
            [
                ("entry_new_button", "New", "Add a condition or effect to this trigger"),
                ("entry_copy_button", "Copy", "Duplicate the selected condition or effect"),
                ("entry_delete_button", "Delete", "Remove the selected condition or effect"),
            ],
            lambda op: (lambda checked=False: self._request_entry_op(op)),
            ("new", "copy", "delete"),
        )
        pane_layout.addLayout(self.entry_buttons)

        # The property form and the vocabulary picker are two pages of one
        # stack rather than two stacked widgets: the picker is a 140-row tree
        # and a permanent home for it would halve the form's height on every
        # document, including the ones nobody ever adds a condition to.
        self.detail_stack = QStackedWidget()
        self.detail_stack.addWidget(self._build_property_page())
        self.detail_stack.addWidget(self._build_picker_page())
        pane_layout.addWidget(self.detail_stack, stretch=1)
        return pane

    def _build_button_row(self, buttons, slot_for, ops) -> QHBoxLayout:
        """One row of equally-weighted push buttons, stored on self by name."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for (attribute, label, tip), op in zip(buttons, ops):
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
                self.status.setText(self._UNSUPPORTED)
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
            # Display order by default -- the in-game order, and what raw
            # list order already disagrees with on 6 of 14 parseable corpus
            # files. "index" (raw list order) is the alternate sort a user can
            # pick, e.g. to match a file's trigger ids for scripting purposes.
            order = (
                list(manager.trigger_display_order)
                if self._sort_mode == "display"
                else list(range(len(triggers)))
            )
            self._display_slots = order
            self._display_position = {index: position for position, index in enumerate(order)}

            names = [self._read(t, "name") or "" for t in triggers]
            self._populate_tag_combo(names)

            # Grouping (4c) exists only under Display order -- section
            # membership is order-derived, so under File order there are no
            # sections and the tree stays flat, whatever the names look like.
            if self._sort_mode == "display":
                parts = trigger_organize.sections(names, order)
            else:
                parts = [trigger_organize.Section("", None, tuple(order))]
            self._grouped = not (len(parts) == 1 and parts[0].header_index is None)
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

    def _make_trigger_item(self, index: int, trigger) -> QTreeWidgetItem:
        item = QTreeWidgetItem([str(index), self._trigger_label(trigger)])
        item.setData(0, Qt.UserRole, index)
        item.setData(0, self._TAG_ROLE, trigger_organize.parse_tag(self._read(trigger, "name") or ""))
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
        for section in parts:
            if section.header_index is None:
                top = QTreeWidgetItem(["", self._BEFORE_FIRST_SECTION])
                top.setData(0, Qt.UserRole, None)
                top.setData(0, self._TAG_ROLE, None)
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
            top.setExpanded(True)

    def _select_default_trigger(self) -> None:
        """topLevelItem(0), or its first child when that row is the synthetic
        leading header -- selecting the header itself would show an empty
        entry pane on most real files (a before-first run is 1..6 triggers
        and never zero once dividers exist, per the corpus census)."""
        if not self.tree.topLevelItemCount():
            return
        first = self.tree.topLevelItem(0)
        if first.data(0, Qt.UserRole) is None and first.childCount():
            first = first.child(0)
        self.tree.setCurrentItem(first)

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
        """(trigger tree path, entry row path, scroll) so a rebuild keeps the
        user's place. By list index / tree path, never by trigger_id: ids are
        positional and a reorder or removal renumbers the whole list."""
        entry = self.entry_tree.currentItem()
        entry_path = None
        if entry is not None:
            parent = entry.parent()
            if parent is None:
                entry_path = (self.entry_tree.indexOfTopLevelItem(entry), None)
            else:
                entry_path = (
                    self.entry_tree.indexOfTopLevelItem(parent),
                    parent.indexOfChild(entry),
                )
        return (
            self._trigger_tree_path(self.tree.currentItem()),
            entry_path,
            self.tree.verticalScrollBar().value(),
        )

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
        trigger_path, entry_path, scroll = state
        self._restore_trigger_path(trigger_path)
        self.tree.verticalScrollBar().setValue(scroll)
        if entry_path is None:
            return
        top, child = entry_path
        if not 0 <= top < self.entry_tree.topLevelItemCount():
            return
        item = self.entry_tree.topLevelItem(top)
        if child is not None:
            if not 0 <= child < item.childCount():
                return
            item = item.child(child)
        self.entry_tree.setCurrentItem(item)

    def _restore_trigger_path(self, path: tuple[int, int] | None) -> None:
        item = None
        if path is not None:
            top, child = path
            if 0 <= top < self.tree.topLevelItemCount():
                item = self.tree.topLevelItem(top)
                if child is not None and 0 <= child < item.childCount():
                    item = item.child(child)
        # Falls back to the default selection rather than the raw item on any
        # miss -- an out-of-range path, or a resolved item that turned out to
        # be the synthetic header (grouping can restructure which row a given
        # path now points at, e.g. after a rename crosses the divider
        # predicate -- see _changed()).
        if item is not None and item.data(0, Qt.UserRole) is not None:
            self.tree.setCurrentItem(item)
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
        value = exec_order_value(loaded)
        if value is None:
            return "Execution order is not stored in this file."
        shown = value if pending_exec_order is None else pending_exec_order
        unsaved = " - unsaved change." if shown != value else "."
        if shown:
            return f"Executes in trigger-ID order (legacy){unsaved}"
        return f"Executes in display order{unsaved}"

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
        return item.data(0, Qt.UserRole)

    def _on_trigger_selected(self) -> None:
        if self._populating:
            return
        self._populate_entry_tree()
        self._update_buttons()

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
        self.entry_tree.setCurrentItem(self.entry_tree.topLevelItem(0))

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

    def _on_entry_selected(self) -> None:
        if self._populating:
            return
        self._populate_property_form()
        self._update_buttons()

    def _specs_for(self, kind: str, entry) -> tuple:
        if kind == "trigger":
            return trigger_fields.TRIGGER_FIELDS
        definition = self._definition_for(kind, entry)
        if definition is None:
            return ()
        _, presentation, type_attribute = self._vocab_for(kind)
        specs = trigger_fields.field_specs(definition, presentation, type_attribute)
        if kind == "effect":
            # Which of quantity / armour_attack_* is authoritative is a runtime
            # decision, and writing the inert one can corrupt the other.
            specs = trigger_fields.apply_armour_attack_rule(specs, entry)
        return specs

    def _populate_property_form(self) -> None:
        self._populating = True
        try:
            self._clear_property_form()
            current = self._current_entry()
            if current is None:
                return
            kind, entry_index, entry = current
            specs = self._specs_for(kind, entry)
            if not specs:
                self.property_form.addRow(QLabel("This type has no editable fields."))
                return
            for spec in specs:
                widget = self._build_widget(spec, kind, entry_index, entry)
                self.property_form.addRow(spec.label, widget)
                self._rows.append((spec, kind, entry_index, widget))
        finally:
            self._populating = False
        self._fit_property_height()

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
        if value is None or value == trigger_fields.UNSET:
            return "(unset)"
        if spec.kind == trigger_fields.INT_LIST:
            return trigger_fields.format_int_list(value) or "(unset)"
        if spec.kind == trigger_fields.BOOL:
            return "yes" if value else "no"
        if spec.kind == trigger_fields.ENUM:
            for label, choice in spec.choices:
                if choice == value:
                    return f"{label} ({choice})"
        if spec.kind == trigger_fields.REFERENCE:
            resolved = trigger_fields.resolve_reference(spec.presentation, value)
            return f"{resolved} ({value})" if resolved else str(value)
        return str(value)

    def _build_widget(self, spec, kind: str, entry_index: int, entry) -> QWidget:
        value = self._read(entry, spec.name)
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
            widget.setChecked(bool(value))
            widget.setEnabled(editable)
            widget.toggled.connect(
                lambda checked, s=spec, k=kind, i=entry_index: self._changed(s, k, i, int(checked))
            )
            return widget

        if spec.kind == trigger_fields.STR:
            widget = QLineEdit("" if value is None else str(value))
            widget.setReadOnly(not editable)
            # Show the start of the value, not its end. A freshly-set QLineEdit
            # leaves the cursor past the last character, so a name too long for
            # the field rendered as "ure: armour split".
            widget.setCursorPosition(0)
            widget.editingFinished.connect(
                lambda s=spec, k=kind, i=entry_index, w=widget: self._changed(s, k, i, w.text())
            )
            return widget

        if spec.kind == trigger_fields.INT_LIST:
            widget = QLineEdit(trigger_fields.format_int_list(value))
            widget.setReadOnly(not editable)
            widget.setPlaceholderText("comma separated")
            widget.editingFinished.connect(
                lambda s=spec, k=kind, i=entry_index, w=widget: self._list_changed(s, k, i, w)
            )
            return widget

        if spec.kind == trigger_fields.ENUM:
            widget = QComboBox()
            _fit_combo_width(widget)
            for label, choice in spec.choices:
                widget.addItem(f"{label} ({choice})", choice)
            if widget.findData(value) < 0 and isinstance(value, int):
                # A value outside the shipped enum. Kept as its own row rather
                # than snapped to the nearest legal one, which would rewrite a
                # field the user never touched.
                widget.addItem(f"unknown ({value})", value)
            widget.setCurrentIndex(max(widget.findData(value), 0))
            widget.setEnabled(editable)
            widget.currentIndexChanged.connect(
                lambda _, s=spec, k=kind, i=entry_index, w=widget: self._changed(
                    s, k, i, w.currentData()
                )
            )
            return widget

        if spec.kind == trigger_fields.REFERENCE:
            catalog_presentation = trigger_fields.CATALOG_PRESENTATIONS.get(spec.presentation)
            if catalog_presentation is not None:
                return self._build_catalog_widget(spec, kind, entry_index, value, catalog_presentation, editable)
            if spec.presentation in trigger_fields.DOCUMENT_REFERENCES:
                return self._build_document_reference_widget(spec, kind, entry_index, value, editable)
            # Unit / Unit[] -- placed-unit reference_ids, not type constants.
            # A picker for them is map-selection integration, deferred to
            # phase 3.5b, so they stay the plain spinbox 4b.6 shipped.
            host = QWidget()
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            spin = self._trigger_spinbox(value, editable)
            resolved = QLabel(trigger_fields.resolve_reference(spec.presentation, value))
            spin.valueChanged.connect(
                lambda new, s=spec, k=kind, i=entry_index, label=resolved: (
                    label.setText(trigger_fields.resolve_reference(s.presentation, new)),
                    self._changed(s, k, i, new),
                )
            )
            row.addWidget(spin, stretch=1)
            row.addWidget(resolved)
            return host

        spin = self._trigger_spinbox(value, editable)
        spin.valueChanged.connect(
            lambda new, s=spec, k=kind, i=entry_index: self._changed(s, k, i, new)
        )
        return spin

    def _build_catalog_widget(
        self, spec, kind: str, entry_index: int, value, presentation, editable: bool
    ) -> QWidget:
        """UnitInfo/BuildingInfo/TechInfo: a CatalogLineEdit over
        object_catalog.py's id-dataset catalog, in place of the raw spinbox."""
        widget = CatalogLineEdit(presentation.catalog(), presentation.default_category)
        initial = value if isinstance(value, int) and not isinstance(value, bool) else None
        widget.set_value(None if initial == trigger_fields.UNSET else initial)
        widget.setEnabled(editable)
        # Connected after set_value(), matching every other widget here: a
        # populate must not look like a user edit (trigger_fields.py's own
        # note on why _changed()'s _populating guard is otherwise unreachable).
        widget.committed.connect(
            lambda new, s=spec, k=kind, i=entry_index: self._changed(s, k, i, new)
        )
        return widget

    def _build_document_reference_widget(
        self, spec, kind: str, entry_index: int, value, editable: bool
    ) -> QComboBox:
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
        widget = QComboBox()
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
        if widget.findData(value) < 0 and isinstance(value, int) and not isinstance(value, bool):
            # An id outside the document's current list -- kept as its own
            # row rather than snapped to "(unset)", which would rewrite a
            # field the user never touched (same reasoning as the ENUM branch).
            widget.addItem(f"unknown ({value})", value)
        widget.setCurrentIndex(max(widget.findData(value), 0))
        widget.setEnabled(editable)
        widget.currentIndexChanged.connect(
            lambda _, s=spec, k=kind, i=entry_index, w=widget: self._changed(s, k, i, w.currentData())
        )
        return widget

    @staticmethod
    def _trigger_spinbox(value, editable: bool) -> QSpinBox:
        """_make_spinbox() with this panel's own range and unset sentinel.
        -1 is the library's unset marker, shown as text rather than as a
        number the user would have to know the meaning of."""
        return _make_spinbox(
            value,
            editable,
            minimum=trigger_fields.UNSET,
            maximum=2**31 - 1,
            special_value_text="(unset)",
        )

    # -- reporting an edit ---------------------------------------------------

    def _changed(self, spec, kind: str, entry_index: int, value) -> None:
        """The one place a widget signal becomes a reported edit.

        The equality check is what actually fires in practice: editingFinished
        arrives on every focus-out whether the text changed or not, and
        commit_trigger_edit() pushes unconditionally, so without it tabbing
        through the form would record one phantom undo step per field.

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
        current = self._current_entry()
        trigger_index = self.current_trigger_index()
        if current is None or trigger_index is None:
            return
        entry_kind, _, entry = current
        if entry_kind != kind:
            return
        before = self._read(entry, spec.name)
        if before == value:
            return

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
            self._on_entry_field(trigger_index, kind, entry_index, spec, value)

        if repartitions:
            self.show_scenario(self._loaded)
            self.select_trigger(trigger_index)
        else:
            self._refresh_labels(trigger_index)

    def _list_changed(self, spec, kind: str, entry_index: int, widget: QLineEdit) -> None:
        if self._populating:
            return
        try:
            value = trigger_fields.parse_int_list(widget.text())
        except ValueError:
            # Rejected whole rather than half-written: "4, x" must not become
            # [4] and silently drop the user's second value.
            current = self._current_entry()
            if current is not None:
                widget.setText(trigger_fields.format_int_list(self._read(current[2], spec.name)))
            return
        self._changed(spec, kind, entry_index, value)

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
                item.setText(1, self._trigger_label(trigger))
                # The tag facet's stashed role, kept in step with a rename that
                # doesn't cross the divider predicate (repartitions handles
                # that case with a full repopulate instead -- see _changed()).
                item.setData(0, self._TAG_ROLE, trigger_organize.parse_tag(self._read(trigger, "name") or ""))

            entry_item = self.entry_tree.currentItem()
            if entry_item is None:
                return
            data = entry_item.data(0, Qt.UserRole)
            if data is None:
                return
            kind, entry_index = data
            if kind == "trigger":
                entry_item.setText(1, self._trigger_label(trigger))
                return
            if kind == "group":
                return
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
        item = self._item_for_index.get(index)
        if item is not None:
            self.tree.setCurrentItem(item)
        else:
            self._select_default_trigger()

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
            self.show_scenario(self._loaded)
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

        self.tree.setCurrentItem(item)
        self._update_buttons()

    def select_entry(self, kind: str, entry_index: int) -> None:
        for i in range(self.entry_tree.topLevelItemCount()):
            top = self.entry_tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole)
            if data != ("group", kind):
                continue
            if 0 <= entry_index < top.childCount():
                self.entry_tree.setCurrentItem(top.child(entry_index))
            return

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

    def _request_trigger_op(self, op: str) -> None:
        if not self._editable:
            return
        index = self.current_trigger_index()
        if op == "new":
            self._on_trigger_structural(op, -1)
            return
        if index is None:
            return
        self._on_trigger_structural(op, index)

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
        selected = self._selected_entry_ref()
        if selected is None:
            return
        kind, entry_index = selected
        self._on_entry_structural(op, trigger_index, kind, entry_index, -1)

    # -- the vocabulary picker -----------------------------------------------

    def _open_picker(self, trigger_index: int) -> None:
        if self._vocabulary is None:
            return
        self._picker_trigger_index = trigger_index
        self._populate_picker()
        self.detail_stack.setCurrentIndex(1)
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
        self.picker_tree.clear()
        self.picker_filter.clear()
        self.detail_stack.setCurrentIndex(0)
        self._set_entry_list_visible(True)
        self._update_buttons()

    def _set_entry_list_visible(self, visible: bool) -> None:
        self.entry_tree.setVisible(visible)
        for i in range(self.entry_buttons.count()):
            widget = self.entry_buttons.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(visible)

    def _populate_picker(self) -> None:
        self.picker_tree.clear()
        if self._vocabulary is None:
            return
        for kind, label in (("condition", "Conditions"), ("effect", "Effects")):
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
            group.setExpanded(True)

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
        if picked is None or trigger_index is None:
            return
        kind, type_id = picked
        self._close_picker()
        self._on_entry_structural("new", trigger_index, kind, -1, type_id)

    # -- button enablement ---------------------------------------------------

    def _update_buttons(self) -> None:
        """Every button's enabled state, in one place.

        Gated on _editable, which is the panel-populate-time read of
        trigger_write_supported. A read-only file browses exactly as before.
        """
        picking = self.detail_stack.currentIndex() == 1
        has_document = self._loaded is not None
        has_trigger = self.current_trigger_index() is not None
        can_edit = self._editable and has_document and not picking

        self.trigger_new_button.setEnabled(can_edit)
        self.trigger_copy_button.setEnabled(can_edit and has_trigger)
        self.trigger_delete_button.setEnabled(can_edit and has_trigger)
        self._update_reorder_buttons(can_edit and has_trigger)

        self.entry_new_button.setEnabled(can_edit and has_trigger and self._vocabulary is not None)
        has_entry = has_trigger and self._selected_entry_ref() is not None
        self.entry_copy_button.setEnabled(can_edit and has_entry)
        self.entry_delete_button.setEnabled(can_edit and has_entry)

        self.picker_add_button.setEnabled(picking and self._picked() is not None)

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

        index = self.current_trigger_index()
        position = self._display_position.get(index) if index is not None else None
        count = len(self._display_slots)

        self.trigger_move_up_button.setEnabled(can_reorder and position is not None and position > 0)
        self.trigger_move_down_button.setEnabled(
            can_reorder and position is not None and position < count - 1
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
        except Exception:
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
        if needle and needle not in item.text(1).lower():
            return False
        if wanted_tag is not None and item.data(0, self._TAG_ROLE) != wanted_tag:
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
            is_real_header = top.data(0, Qt.UserRole) is not None
            header_matches = is_real_header and self._row_matches(top, needle, wanted_tag)
            shown = 0
            for c in range(top.childCount()):
                child = top.child(c)
                matches = self._row_matches(child, needle, wanted_tag)
                child.setHidden(not matches)
                shown += matches
            top.setHidden(not (header_matches or shown))
