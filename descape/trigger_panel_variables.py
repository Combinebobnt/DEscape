"""TriggerPanel's variables: VariablesDialog and the panel's side of opening, refreshing and closing it."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from descape.scenario_io import (
    parse_triggers,
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

    def __init__(self, parent=None, on_structural=None, keybind_actions=None):
        super().__init__(parent)
        self.setWindowTitle("Trigger Variables")
        self.resize(420, 360)
        self._on_structural = on_structural or (lambda *args: None)
        self._editable = False
        self._keybind_actions = dict(keybind_actions or {})

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

        for action in self._keybind_actions.values():
            self.addAction(action)
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
        for action_id, action in self._keybind_actions.items():
            button = getattr(self, f"{action_id.removeprefix('variable_')}_button")
            action.setEnabled(button.isEnabled())

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


class _TriggerPanelVariablesMixin:
    """TriggerPanel's variables dialog hooks (trigger_panel.py's docstring has the split's rules)."""

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
            self._variables_dialog = VariablesDialog(
                self, on_structural=self._on_variable_structural, keybind_actions=self.variable_actions
            )
        self.refresh_variables()
        self._variables_dialog.show()
        self._variables_dialog.raise_()
        self._variables_dialog.activateWindow()

    def _dispatch_variable_action(self, op: str) -> None:
        dialog = self._variables_dialog
        if dialog is not None and dialog.isVisible():
            button = getattr(dialog, f"{op}_button")
            if button.isEnabled():
                button.click()

    def close_variables_dialog(self) -> None:
        if self._variables_dialog is not None:
            dialog = self._variables_dialog
            self._variables_dialog = None
            for action in dialog._keybind_actions.values():
                dialog.removeAction(action)
                action.setEnabled(False)
            # Old deferred selection/text signals must not re-enable a replacement's actions.
            dialog._keybind_actions.clear()
            dialog.close()
            dialog.deleteLater()

    def refresh_variables(self) -> None:
        """Repopulate the variables dialog if one is open. No-op otherwise, so
        the window's edit funnel can call it unconditionally."""
        if self._variables_dialog is None:
            return
        self._variables_dialog.set_variables(self.variable_rows(), editable=self._editable)
