"""ViewerWindow's Tools > Launch in AoE2:DE (GH #4): writes the open document,
unsaved edits included, to game_launch's fixed launch copy, then starts the
game through Steam. Not a Save: the document's saved state, title and
autosave are left alone."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QMessageBox

from descape import debug_log, game_launch, games, legacy_write, scenario_format, yaml_file_io
from descape.scenario_write import WriteBlockedError, check_destination

ACTION_LABEL = "&Launch in AoE2:DE"
ACTION_TIP = (
    f"Write this scenario to {game_launch.LAUNCH_NAME} in the game's scenario folder "
    "and start AoE2:DE"
)
LEGACY_REFUSAL = "Convert it to .aoe2scenario first (File > Convert to .aoe2scenario…)"
NO_CONVERT_REFUSAL = "AoE2:DE opens only .aoe2scenario files, and {} files do not convert"
IS_TARGET_REFUSAL = "This is the launch copy: Save As another name first"
RUNNING_LINE = "AoE2:DE is running: in its editor, Load DEscape_launch"


def _same_path(a: Path, b: Path) -> bool:
    try:
        return yaml_file_io._same_file(Path(a).resolve(), Path(b).resolve())
    except OSError:
        return False


class _ViewerGameLaunchMixin:
    def _launch_refusal(self) -> str | None:
        """Why Tools > Launch in AoE2:DE is off, None when it is on."""
        platform = game_launch.refusal_for_platform()
        if platform is not None:
            return platform
        if self.scenario is None:
            return "Open a scenario to launch it in AoE2:DE"
        if self._file_is_legacy():
            fmt = self.scenario.source_format
            return LEGACY_REFUSAL if scenario_format.converts(fmt) else NO_CONVERT_REFUSAL.format(fmt.display_name)
        reason = self.terrain_lock_reason(legacy_write.ANY)
        if not self.scenario.terrain_write_supported or reason is not None:
            return reason or "This scenario can't be written"
        # Name first: this runs on every tool-state update, and launch_target() reads config and disk.
        # launch_in_game() repeats the check without the name filter, so another link to it is refused too.
        if not self._untitled and Path(self.scenario.path).name.lower() == game_launch.LAUNCH_NAME.lower():
            target, _ = game_launch.launch_target()
            if target is not None and _same_path(self.scenario.path, target):
                return IS_TARGET_REFUSAL
        return None

    def _confirm_overwrite_launch_copy(self, target: Path) -> bool:
        kept = game_launch.keep_foreign_name(target)
        box = QMessageBox(QMessageBox.Warning, "Launch in AoE2:DE", "", QMessageBox.NoButton, self)
        box.setText(f"{target.name} at {target} was changed outside DEscape.")
        box.setInformativeText(f"Overwrite it? The changed file is kept beside it as {kept.name}.")
        overwrite = box.addButton("Overwrite", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        box.exec_()
        return box.clickedButton() is overwrite

    def launch_in_game(self) -> None:
        # Busy and mid-stroke are checked here, not in the gate: a load updates the gate while _busy.
        refusal = self._launch_refusal()
        if refusal is None and (self._busy or self._stroke_in_progress()):
            refusal = "Finish the current operation first"
        if refusal is not None:
            self._log_status(f"Launch in AoE2:DE refused. {refusal}", level=debug_log.WARNING)
            return
        self._commit_focused_field()
        target, missing = game_launch.launch_target()
        if target is None:
            self._log_status(f"Launch in AoE2:DE: {missing}", level=debug_log.WARNING)
            QMessageBox.information(self, "Launch in AoE2:DE", missing)
            return
        if not self._untitled and _same_path(self.scenario.path, target):
            self._log_status(f"Launch in AoE2:DE refused. {IS_TARGET_REFUSAL}", level=debug_log.WARNING)
            return
        try:
            # The writers check this too; here so a refused folder is never asked about first.
            check_destination(self.scenario, target)
            if target.exists() and not game_launch.ledger_matches(target):
                if not self._confirm_overwrite_launch_copy(target):
                    self._log_status("Launch in AoE2:DE cancelled")
                    return
                kept = game_launch.keep_foreign(target)
                self._log_status(f"Kept the changed {target.name} as {kept.name}")
            game_launch.write_launch_copy(target, self._write_document)
        except WriteBlockedError as e:
            self._log_status(f"Launch blocked: {e}", level=debug_log.WARNING)
            QMessageBox.critical(self, "Launch blocked", str(e))
            return
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reported below
            self._log_status(f"Launch failed: {type(e).__name__}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Launch failed", f"{type(e).__name__}: {e}")
            return
        game = games.for_scenario(self.scenario)
        suffix = "" if game is games.default() else f"; switch the game to {game.display_name} first"
        if game_launch.game_running():
            self._log_status(f"Wrote {target}. {RUNNING_LINE}{suffix}")
            return
        # The target sits under the profile, so it tells a Flatpak Steam's profile apart too.
        command = game_launch.launch_command(game_launch.launch_args(), profile=target)
        try:
            game_launch.spawn(command)
        except OSError as e:
            shown = command if isinstance(command, str) else " ".join(command)
            self._log_status(f"Launch failed: {shown}: {e}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Launch failed", f"Could not start AoE2:DE with\n{shown}\n\n{e}")
            return
        self._log_status(f"Wrote {target}; launching AoE2:DE (in its editor, Load DEscape_launch){suffix}")
