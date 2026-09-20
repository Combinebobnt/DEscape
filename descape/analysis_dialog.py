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


def _group_text(result: CheckResult) -> str:
    if result.unavailable:
        return f"{result.label} (unavailable: {result.unavailable})"
    if not result.findings:
        return f"{result.label} (clean)"
    return f"{result.label} ({len(result.findings)})"


class AnalysisDialog(QDialog):
    def __init__(self, parent=None, on_navigate=None):
        super().__init__(parent)
        self.setWindowTitle("Map Analysis")
        self.resize(640, 420)
        self._on_navigate = on_navigate or (lambda finding: None)

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

        hint = QLabel("Double-click a finding to jump to its tile or unit.")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.headline)
        layout.addWidget(self.tree, stretch=1)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def set_report(self, report: AnalysisReport, source_name: str = "") -> None:
        self.headline.setText(f"{source_name}: {report.headline}" if source_name else report.headline)
        self.tree.clear()
        for result in report.results:
            group = QTreeWidgetItem([_group_text(result), ""])
            self.tree.addTopLevelItem(group)
            group.setFirstColumnSpanned(True)
            for finding in result.findings:
                item = QTreeWidgetItem([_SEVERITY_TEXT.get(finding.severity, finding.severity), finding.message])
                item.setData(0, Qt.UserRole, finding)
                if finding.tile is None and finding.unit_key is None:
                    item.setToolTip(1, "No map location for this finding")
                group.addChild(item)
            group.setExpanded(bool(result.findings))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        finding = item.data(0, Qt.UserRole)
        if isinstance(finding, Finding) and (finding.tile is not None or finding.unit_key is not None):
            self._on_navigate(finding)
