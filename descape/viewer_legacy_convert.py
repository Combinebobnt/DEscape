"""ViewerWindow's File > Convert to .aoe2scenario (GH #315): a new untitled DE
document from the open pre-DE one, through legacy_convert. The legacy file
is only read; the result starts unsaved, and Save As starts beside it."""

from __future__ import annotations

import weakref
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from descape import debug_log, legacy_convert, scenario_format, scenario_library
from descape.scenario_io import LoadedScenario
from descape.viewer_file import UNTITLED_PATH, _in_subscribed, _in_template_dir

ACTION_LABEL = "Convert to .aoe2scenario…"
ACTION_TIP = "Make an editable AoE2:DE copy of this older-format scenario, as AoE2:DE's own editor imports it"


class _ViewerLegacyConvertMixin:
    # (converted document, its legacy source's path); weak, so a later load frees the document.
    _converted_from: tuple[weakref.ref[LoadedScenario], Path] | None = None

    def _convert_legacy_refusal(self) -> str | None:
        """Why File > Convert to .aoe2scenario is off, None when it is on."""
        if self.scenario is None or not self.scenario.source_format.legacy:
            return "Open an older-format scenario (.scn, .scx, .scx2) to convert it"
        if not scenario_format.converts(self.scenario.source_format):
            return f"{self.scenario.source_format.display_name} files do not convert to .aoe2scenario"
        return None

    def convert_legacy(self) -> None:
        old = self.scenario
        if self._busy or self._convert_legacy_refusal() is not None or not self._confirm_discard_changes():
            return
        source, failure = old.path, None
        self.statusBar().showMessage(f"Converting {source.name}…")
        # As load_scenario(): no queued Open may run inside processEvents() and be replaced by the result.
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            data, report = legacy_convert.convert(old)
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reported below
            failure = f"{type(e).__name__}: {e}"
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        if failure is not None:
            self.statusBar().clearMessage()
            self._log_status(f"Convert to .aoe2scenario failed: {failure}", level=debug_log.ERROR)
            QMessageBox.critical(self, "Convert to .aoe2scenario failed", failure)
            return
        self.load_scenario(UNTITLED_PATH, untitled=True, data=data)
        if self.scenario is old or self.scenario is None:
            return  # the load itself failed and already reported why
        self._converted_from = (weakref.ref(self.scenario), source)
        self.edit_history.saved_at_cursor = None  # unsaved, so the discard prompt and autosave cover it
        self._update_edit_actions()
        self._update_title()
        self._log_status(f"Converted {source.name} to a new AoE2:DE scenario. Not saved yet: use Save As.")
        lines = report.lines()
        for line in lines:
            self._log_status(f"Convert: {line}", level=debug_log.WARNING if report.left_out else debug_log.INFO)
        if report.left_out:
            box = QMessageBox(QMessageBox.Information, "Convert to .aoe2scenario", "", QMessageBox.Ok, self)
            box.setText(f"{source.name} was converted, but not everything came across.")
            box.setInformativeText("The Status Log lists each item.")
            box.setDetailedText("\n".join(lines))
            box.exec_()

    def _converted_save_path(self) -> str | None:
        """Save As's start for a converted document: <legacy folder>/<stem>.aoe2scenario, or the
        game's scenario folder when that folder is subscribed or the shipped templates. Else None."""
        converted = self._converted_from
        if not (self._untitled and converted is not None and converted[0]() is self.scenario):
            return None
        target = converted[1].with_suffix(".aoe2scenario")
        if _in_subscribed(target) or _in_template_dir(target) or not target.parent.is_dir():
            return str((scenario_library.default_folder() or Path.home()) / target.name)
        return str(target)
