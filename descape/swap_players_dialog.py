"""SwapPlayersDialog: Swap Players (GH #185), opened from the Players panel's
button and Map > Swap Players….

Dumb like DisablesDialog: it never touches a model. It is handed the defined
players, the player labels and colours, the starting player and two
callbacks the window answers from player_swap: `partners(a)`, the players A
can swap with (swap_partners()), and `preview(a, b, opts) -> (text,
can_swap)` (plan_swap()); the line shows what would move, or the reason the
file refuses the swap, which also disables Swap. choice() is what the window
applies once exec_() accepts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from descape import player_labels
from descape.player_swap import SwapOptions
from descape.viewer_common import _add_player_item


class SwapPlayersDialog(QDialog):
    """Modal. Construct per open, exec_(), read choice(), discard."""

    _TITLE = "Swap Players"
    _INTRO = (
        "The two players exchange slots: each one's units, Players settings, "
        "Personality and disabled objects move to the other."
    )
    _PLAYER_LABEL = "Player"
    _PARTNER_LABEL = "Swap with"
    _TRIGGERS_LABEL = "Swap players in triggers"
    _TRIGGERS_TOOLTIP = (
        "Conditions and effects that name one of the two as their source or "
        "target player name the other instead. Trigger names, text and scripts "
        "are not changed."
    )
    _DIPLOMACY_LABEL = "Swap diplomacy"
    _DIPLOMACY_TOOLTIP = (
        "Each player's stances toward the others, the others' stances toward "
        "them, and allied victory move with them."
    )
    _SWAP_LABEL = "Swap"

    def __init__(
        self,
        players: Sequence[int],
        initial: int,
        partners: Callable[[int], Sequence[int]],
        preview: Callable[[int, int, SwapOptions], tuple[str, bool]],
        parent=None,
        *,
        labels=None,
        colors=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(self._TITLE)
        self._players = list(players)
        self._partners = partners
        self._preview = preview
        self._labels = labels or player_labels.DEFAULT_LABELS
        self._colors = colors
        self._populating = False

        layout = QVBoxLayout(self)
        intro = QLabel(self._INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.player_combo = QComboBox()
        for pid in self._players:
            _add_player_item(self.player_combo, pid, self._labels, self._colors, pid)
        self.partner_combo = QComboBox()
        form.addRow(self._PLAYER_LABEL, self.player_combo)
        form.addRow(self._PARTNER_LABEL, self.partner_combo)
        layout.addLayout(form)

        defaults = SwapOptions()
        self.triggers_check = QCheckBox(self._TRIGGERS_LABEL)
        self.triggers_check.setChecked(defaults.triggers)
        self.triggers_check.setToolTip(self._TRIGGERS_TOOLTIP)
        self.diplomacy_check = QCheckBox(self._DIPLOMACY_LABEL)
        self.diplomacy_check.setChecked(defaults.diplomacy)
        self.diplomacy_check.setToolTip(self._DIPLOMACY_TOOLTIP)
        layout.addWidget(self.triggers_check)
        layout.addWidget(self.diplomacy_check)

        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.swap_button = self.buttons.addButton(self._SWAP_LABEL, QDialogButtonBox.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        index = self.player_combo.findData(initial)
        self.player_combo.setCurrentIndex(max(index, 0))
        self._refresh_partners()
        self.player_combo.currentIndexChanged.connect(lambda *_: self._refresh_partners())
        self.partner_combo.currentIndexChanged.connect(lambda *_: self._update_preview())
        self.triggers_check.toggled.connect(lambda *_: self._update_preview())
        self.diplomacy_check.toggled.connect(lambda *_: self._update_preview())

    def _refresh_partners(self) -> None:
        """B's list is `partners(A)`, keeping B when it still can be."""
        a = self.player_combo.currentData()
        keep = self.partner_combo.currentData()
        self._populating = True
        try:
            self.partner_combo.clear()
            for pid in self._partners(a):
                _add_player_item(self.partner_combo, pid, self._labels, self._colors, pid)
            index = self.partner_combo.findData(keep)
            self.partner_combo.setCurrentIndex(max(index, 0))
        finally:
            self._populating = False
        self._update_preview()

    def _update_preview(self) -> None:
        if self._populating:
            return
        a, b, opts = self.choice()
        if a is None or b is None:
            text, ok = "", False
        else:
            text, ok = self._preview(a, b, opts)
        self.preview_label.setText(text)
        self.swap_button.setEnabled(ok)
        # A longer line (a refusal) wraps to more rows: grow to fit it, never shrink under the user.
        needed = self.heightForWidth(self.width())
        if self.isVisible() and needed > self.height():
            self.resize(self.width(), needed)

    def choice(self) -> tuple[int | None, int | None, SwapOptions]:
        """(A, B, the checkboxes) as the dialog shows them."""
        opts = SwapOptions(triggers=self.triggers_check.isChecked(), diplomacy=self.diplomacy_check.isChecked())
        return self.player_combo.currentData(), self.partner_combo.currentData(), opts
