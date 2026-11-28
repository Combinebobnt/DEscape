"""The viewer's side of a Game.derived_root game (TASK-430, D4 and D5).

It builds the root before activating the game, and says why a file shows with other data.

game_dataset.set_active() stays Qt-free and never builds, so both of the
viewer's activation sites ask ready_shown_game() for the game to activate:
when that game needs a derived root that is not current, the root is built
first on a worker thread, the status bar's progress line showing it while the
window waits (no processEvents(), for load_scenario()'s reentrancy rule). A
build that fails is recorded (legacy_content.build_failed()), which makes the
game unconfigured for the rest of the session's install setting, so Auto
shows the file with its stand-in's data and the status log says why.
"""

from __future__ import annotations

import threading

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from descape import asset_source, debug_log, game_dataset, games, legacy_content, settings

PROGRESS_OWNER = "game_root"
_POLL_S = 0.05


def ready_shown_game(window) -> str:
    """The game Settings > Advanced > Game data shows the open document in
    (window._shown_game_dataset()), its derived root built first when it needs one."""
    shown = games.get(window._shown_game_dataset())
    if shown.derived_root:
        build_root(window, shown)
    return window._shown_game_dataset()  # again: a failed build turns Auto to the stand-in


def build_root(window, game: games.Game) -> bool:
    """Builds `game`'s derived root unless it is current; True when it is usable afterwards."""
    folder = game_dataset.install_folder(game.install)
    builder = legacy_content.recipe(game.install)
    if folder is None or not builder.validate_install(folder)[0]:
        return False
    if builder.built_root(folder) is not None:
        return True
    if builder.build_failed(folder) is not None:
        return False
    state: dict[str, object] = {"progress": (0, "starting")}

    def run() -> None:
        try:
            state["root"] = builder.ensure_root(folder, lambda p, c: state.__setitem__("progress", (p, c)))
        except legacy_content.RootBuildError as exc:
            state["error"] = str(exc)

    caption = f"Preparing {game.display_name} data"
    window.load_progress.start(PROGRESS_OWNER, caption)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    worker = threading.Thread(target=run, name="descape-game-root", daemon=True)
    try:
        worker.start()
        shown = None
        while worker.is_alive():
            worker.join(_POLL_S)
            if state["progress"] != shown:
                shown = state["progress"]
                window.load_progress.step(PROGRESS_OWNER, shown[0], f"{caption}: {shown[1]}")
    finally:
        QApplication.restoreOverrideCursor()
        window.load_progress.finish(PROGRESS_OWNER)
    # The root (or the failure) changes what get_install_path() and configured() answer.
    asset_source.clear_install_caches(None)
    if "error" in state or state.get("root") is None:
        state.setdefault("error", f"Could not unpack the {game.display_name} install")
        window._log_status(f"{state['error']}. Showing the file with other game data.", level=debug_log.ERROR)
        return False
    window._log_status(f"{game.display_name} data unpacked to {state['root']}")
    return True


def status_lines(window) -> list[str]:
    """What the status log says about the shown game data, after a change:
    a 1999 or HD Edition file shown with its stand-in's data under Auto (D4),
    a file a configured game's data shows only by hand (TASK-431 H4), and a
    shown game whose own art is missing."""
    lines = []
    scenario = window.scenario
    own = None if scenario is None else games.for_scenario(scenario)
    if own is not None and settings.get_game_data() == settings.GAME_DATA_DEFAULT and not game_dataset.configured(own):
        stand_in = game_dataset.effective_game(own)
        lines.append(
            f"Shown with {stand_in.display_name} data. Set the {own.display_name} install in "
            "Settings > Advanced > Game data to show it with its own."
        )
    if own is not None and settings.get_game_data() == settings.GAME_DATA_DEFAULT:
        lines += _hand_pick_lines(scenario)
    shown = game_dataset.game()
    name = shown.display_name
    marks = f"names and stats are {name}'s, units draw as coloured marks."
    if shown.derived_root and game_dataset.install_root(shown) is None:
        lines.append(f"{name} art is not available (its game data {game_dataset.missing_install_reason(shown)}): {marks}")
    else:
        install = asset_source.get_install_path()
        if install is not None and shown.root_subpath is not None and not shown.root(install).is_dir():
            lines.append(f"{name} art is not installed: {marks}")
    return lines


def _hand_pick_lines(scenario) -> list[str]:
    """TASK-431 H4: a file no game claims that a configured game's data can show when picked by hand."""
    fmt = getattr(scenario, "source_format", None)
    label = getattr(fmt, "game", None)
    if label is None:
        return []
    return [
        f"An {fmt.display_name} file: Settings > Advanced > Game data > {game.display_name} shows it "
        "with that game's own data, view-only."
        for game in games.GAMES
        if label in game.hand_pick_formats and game_dataset.configured(game)
    ]
