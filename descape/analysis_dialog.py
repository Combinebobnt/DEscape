"""Tools > Map Analysis results: a modeless dialog over a map_analysis.AnalysisReport.

Dumb the same way VariablesDialog is: it never touches the map. Double-
clicking a finding with a tile or unit calls the on_navigate callback it was
constructed with, and the window decides what "jump there" means.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from descape.map_analysis import AnalysisReport, CheckResult, Finding
from descape.value_picker import _configure_scrolling

_SEVERITY_TEXT = {"error": "Error", "warning": "Warning", "info": "Info"}
# A check row's CheckResult.label: its text carries a count, so it can't name the check.
_LABEL_ROLE = Qt.UserRole + 1


def _group_text(result: CheckResult) -> str:
    if result.unavailable:
        return f"{result.label} (unavailable: {result.unavailable})"
    if not result.findings:
        return f"{result.label} (clean)"
    return f"{result.label} ({len(result.findings)})"


class AnalysisDialog(QDialog):
    """on_select gets the current row's Finding, or None for a group row;
    on_closed fires however the dialog goes away (Close, Escape, title-bar X);
    on_refresh is the Refresh button (GH #209): the window re-runs the
    analysis and calls set_report() again."""

    def __init__(self, parent=None, on_navigate=None, on_select=None, on_closed=None, on_refresh=None):
        super().__init__(parent)
        self.setWindowTitle("Map Analysis")
        self.resize(640, 420)
        self._on_navigate = on_navigate or (lambda finding: None)
        self._on_select = on_select or (lambda finding: None)
        self._on_closed = on_closed or (lambda: None)
        self._on_refresh = on_refresh or (lambda: None)
        # finished, not closeEvent: Escape reaches reject() -> done() without a closeEvent.
        self.finished.connect(lambda _result: self._on_closed())

        self.headline = QLabel("")
        self.headline.setWordWrap(True)

        self.tree = QTreeWidget()
        # Severity first: messages are long, and a trailing column scrolls out of view.
        self.tree.setHeaderLabels(["Severity", "Finding"])
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        _configure_scrolling(self.tree)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)

        hint = QLabel("Double-click a finding to jump to its tile or unit.")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        self.refresh_button = buttons.addButton("Refresh", QDialogButtonBox.ActionRole)
        self.refresh_button.setToolTip("Run the analysis again on the map as it is now")
        # Close stays the dialog's default button, so Enter never starts a re-run.
        self.refresh_button.setAutoDefault(False)
        self.refresh_button.clicked.connect(self._refresh)

        layout = QVBoxLayout(self)
        layout.addWidget(self.headline)
        layout.addWidget(self.tree, stretch=1)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def set_report(self, report: AnalysisReport, source_name: str = "") -> None:
        """Refills the tree. A re-run keeps the current row (matched by check
        and finding, never by row number), the scroll position and which
        checks were collapsed."""
        kept = self._row_key(self.tree.currentItem())
        collapsed = {
            group.data(0, _LABEL_ROLE)
            for group in map(self.tree.topLevelItem, range(self.tree.topLevelItemCount()))
            if group.childCount() and not group.isExpanded()
        }
        scroll = self.tree.verticalScrollBar().value()
        self.headline.setText(f"{source_name}: {report.headline}" if source_name else report.headline)
        self.tree.clear()
        for result in report.results:
            group = QTreeWidgetItem([_group_text(result), ""])
            group.setData(0, _LABEL_ROLE, result.label)
            self.tree.addTopLevelItem(group)
            group.setFirstColumnSpanned(True)
            for finding in result.findings:
                item = QTreeWidgetItem([_SEVERITY_TEXT.get(finding.severity, finding.severity), finding.message])
                item.setData(0, Qt.UserRole, finding)
                if finding.tile is None and finding.unit_key is None:
                    item.setToolTip(1, "No map location for this finding")
                group.addChild(item)
            group.setExpanded(bool(result.findings) and result.label not in collapsed)
        target = self._item_for_key(kept)
        if target is not None:
            self.tree.setCurrentItem(target)
        self.tree.verticalScrollBar().setValue(scroll)
        if target is not None:
            # Rows above it may be gone; a no-op when the old position still shows it.
            self.tree.scrollToItem(target)

    def _row_key(self, item: QTreeWidgetItem | None) -> tuple[str | None, Finding | None]:
        """(check label, finding) for a finding row, (label, None) for a check's own row."""
        if item is None:
            return None, None
        finding = item.data(0, Qt.UserRole)
        group = item if item.parent() is None else item.parent()
        return group.data(0, _LABEL_ROLE), finding if isinstance(finding, Finding) else None

    def _item_for_key(self, key: tuple[str | None, Finding | None]) -> QTreeWidgetItem | None:
        label, finding = key
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            if group.data(0, _LABEL_ROLE) != label:
                continue
            if finding is None:
                return group
            rows = [group.child(j) for j in range(group.childCount())]
            same = [row for row in rows if row.data(0, Qt.UserRole) == finding]
            if not same and finding.unit_key is not None:
                # A moved unit's message changes; it is still this check's finding about that unit.
                same = [row for row in rows if row.data(0, Qt.UserRole).unit_key == finding.unit_key]
            return same[0] if same else None
        return None

    def _refresh(self) -> None:
        self._on_refresh()
        # The window disables itself while it re-runs, and this dialog with it, which drops the focus.
        self.refresh_button.setFocus()

    def _on_current_item_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        finding = current.data(0, Qt.UserRole) if current is not None else None
        self._on_select(finding if isinstance(finding, Finding) else None)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        finding = item.data(0, Qt.UserRole)
        if isinstance(finding, Finding) and (finding.tile is not None or finding.unit_key is not None):
            self._on_navigate(finding)
