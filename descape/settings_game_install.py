"""Settings > Advanced > Game data's install rows (TASK-430 D5, TASK-431 H6): one per game with its own install.

Each Game.derived_root game whose install kind has a recipe
(legacy_content.recipe(): the 1999 game's, HD Edition's) gets one row. Set,
the game's files show with its own data under Auto (its derived root is built
on the next activation, descape/viewer_game_root.py); unset, they show with
its stand-in's data, as before. Like the AoE2:DE install path each is a
machine-local key (config.yaml's recipe INSTALL_KEY), applied at once and
left alone by Cancel and Reset. No auto-detection: nothing is set without the
user's pick, though Browse opens where the recipe expects the game.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from descape import asset_source, debug_log, game_dataset, games, legacy_content


def install_games() -> list[games.Game]:
    """The registered games that get a row, in registry order (one per install kind)."""
    found, kinds = [], set()
    for game in games.GAMES:
        if game.derived_root and game.install not in kinds:
            kinds.add(game.install)
            found.append(game)
    return found


class InstallRow:
    """One game's row: its widgets and handlers, owned by the Settings dialog (`dialog._window` is the viewer)."""

    def __init__(self, dialog, layout: QVBoxLayout, game: games.Game) -> None:
        self._dialog = dialog
        self.game = game
        self._recipe = legacy_content.recipe(game.install)
        self._stand_in = games.get(game.stand_in).display_name
        label = QLabel(f"{game.display_name} install (optional, shows its files with its own data):")
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(self._recipe.PLACEHOLDER)
        row.addWidget(self.edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)
        row.addWidget(self.load_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip(f"Forget this install: its files show with {self._stand_in} data again.")
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.clear_button)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._show_current()

    def _show_current(self) -> None:
        current = self._recipe.install_path()
        self.edit.setText("" if current is None else str(current))
        self.clear_button.setEnabled(current is not None)
        if current is None:
            self.status.setText(f"Not set: these files show with {self._stand_in} data.")
        else:
            self.status.setText(self._recipe.validate_install(current)[1])

    def browse_start(self) -> str:
        """The typed path, else where the recipe expects the game, else the home folder."""
        typed = self.edit.text().strip()
        if typed:
            return typed
        expected = self._recipe.browse_start()
        return str(expected if expected is not None else Path.home())

    def browse(self) -> None:
        title = f"Select the {self.game.display_name} install folder"
        path = QFileDialog.getExistingDirectory(self._dialog, title, self.browse_start())
        if path:
            self.edit.setText(path)

    def load(self) -> None:
        name = self.game.display_name
        text = self.edit.text().strip()
        if not text:
            self.status.setText(f"Enter or browse to the {name} install first.")
            return
        ok, message = self._recipe.validate_install(Path(text))
        self.status.setText(message)
        window = self._dialog._window
        if not ok:
            window._log_status(f"{name} install rejected ({text}): {message}", level=debug_log.WARNING)
            return
        self._recipe.save_install_path(Path(text))
        window._log_status(f"{name} install set to {text}")
        self._changed()
        if os.environ.get(self._recipe.INSTALL_ENV):
            self.status.setText(f"Saved, but {self._recipe.INSTALL_ENV} still sets the install in use.")

    def clear(self) -> None:
        self._recipe.save_install_path(None)
        self._dialog._window._log_status(f"{self.game.display_name} install cleared")
        self._changed()
        if os.environ.get(self._recipe.INSTALL_ENV):
            self.status.setText(f"Saved path cleared; {self._recipe.INSTALL_ENV} still sets one.")

    def _changed(self) -> None:
        """D5's install-change clear: forget the old root and any failed build, re-pick and redraw."""
        self._recipe.forget_failures()
        asset_source.clear_install_caches(None)
        window = self._dialog._window
        was = game_dataset.active()
        window.apply_game_data()  # builds the root first when the shown game needs one
        if game_dataset.active() == was and game_dataset.game().derived_root:
            asset_source.clear_install_caches(None)
            window._refresh_game_assets()  # the same game, a new root: apply_game_data() saw no switch
        self._show_current()
