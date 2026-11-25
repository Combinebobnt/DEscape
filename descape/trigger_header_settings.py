"""Settings > General > Trigger section headers (TASK-486): which trigger names head a section.

A QWidget beside SettingsDialog, the way tool_defaults_page.ToolDefaultsPage is.
One checkbox per built-in shape DEscape never writes, and a field of extra
starting tokens. Every change goes through ViewerWindow.set_divider_rule(),
which saves it and re-groups the open scenario at once; Settings' Cancel and
Reset put it back through the dialog's applier like any other key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QCheckBox, QLabel, QLineEdit, QVBoxLayout, QWidget

from descape import settings, trigger_organize
from descape.viewer_common import status_color

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow

_TOGGLES = (
    ("bare_rule", "Bare rules (------)", "e.g. a trigger named ------"),
    ("leading_rule", "Leading rule only (----- Name)", "e.g. ----- ambush-setup"),
    ("tilde_wrap", "Single ~ wrap (~Name~)", "e.g. ~Part I~"),
)
_AFFECTS = "Affects grouping, Select Section, Move to Section and the trigger status. The file is not changed."


class TriggerHeaderSettings(QWidget):
    def __init__(self, window: ViewerWindow, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._window = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Trigger section headers:"))
        note = QLabel('"--- Name ---" always counts: it is the shape New Section writes.')
        note.setWordWrap(True)
        layout.addWidget(note)

        rule = settings.get_trigger_section_headers()
        self.checkboxes: dict[str, QCheckBox] = {}
        for field, text, example in _TOGGLES:
            box = QCheckBox(text)
            box.setObjectName(f"trigger_header_{field}")
            box.setChecked(getattr(rule, field))
            box.setToolTip(f"Counts {example} as a section header. {_AFFECTS}")
            box.toggled.connect(self._apply)
            layout.addWidget(box)
            self.checkboxes[field] = box

        prefix_label = QLabel("Also a header when the name starts with:")
        prefix_label.setWordWrap(True)
        layout.addWidget(prefix_label)
        self.prefix_edit = QLineEdit(" ".join(rule.prefixes))
        self.prefix_edit.setObjectName("trigger_header_prefixes")
        self.prefix_edit.setPlaceholderText("e.g. >> //")
        self.prefix_edit.setToolTip(
            "Space-separated, case-sensitive, matched against the name as stored: "
            f"with >> a trigger named \">> Setup\" heads a section titled Setup. "
            f"At most {trigger_organize.MAX_PREFIXES}. {_AFFECTS}"
        )
        self.prefix_edit.editingFinished.connect(self._apply)
        layout.addWidget(self.prefix_edit)
        self.error_label = QLabel()
        self.error_label.setObjectName("trigger_header_error")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {status_color(False, on_tab_page=True)};")
        self.error_label.hide()
        layout.addWidget(self.error_label)

    def detach(self, apply: bool = False) -> None:
        """No further applies: the dialog is closing, or this page is being replaced.
        `apply` first takes a typed but unconfirmed prefix field (OK pressed from the keyboard)."""
        if apply and not self.prefix_edit.signalsBlocked():
            self._apply()
        self.prefix_edit.blockSignals(True)
        for box in self.checkboxes.values():
            box.blockSignals(True)

    def _apply(self) -> None:
        toggles = {field: box.isChecked() for field, box in self.checkboxes.items()}
        tokens = list(dict.fromkeys(self.prefix_edit.text().split()))
        rule = trigger_organize.DividerRule.with_prefixes(tokens, **toggles)
        error = rule.error()
        self.error_label.setText(error[:1].upper() + error[1:] if error else "")
        self.error_label.setVisible(bool(error))
        if error:
            # A toggle still applies, beside the saved prefixes; the field keeps the bad text to fix.
            saved = settings.get_trigger_section_headers().prefixes
            rule = trigger_organize.DividerRule.with_prefixes(saved, **toggles)
        else:
            self.prefix_edit.setText(" ".join(tokens))
        self._window.set_divider_rule(rule)
