"""GH #359: the status bar's load progress, a caption and a bar on its right.

It holds one job per owner in OWNERS and shows the highest-ranked running one,
so a scenario load started during a data mod parse takes the bar over, and the
parse's comes back once the load finishes. It is hidden while no job runs.

A scenario load runs on the GUI thread, so between its stages no posted layout
or paint event runs: each change lays the status bar out and repaints it on
the spot. No processEvents() here, for the reentrancy load_scenario() documents.
"""

from __future__ import annotations

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QWidget

# Highest rank first: a data mod parse can start inside a load (for_document()), and a 1999 root build inside either.
OWNERS = ("game_root", "load", "data_mod")
# A scenario load's stages: what the caption says, and the bar's percent as it starts.
SCENARIO_STAGES = {
    "read": ("reading the file", 10),
    "prepare": ("preparing the map", 35),
    "panels": ("filling the panels", 60),
    "draw": ("drawing the map", 80),
}
BAR_WIDTH = 160
# What a scenario load's caption opens with, by the kind of load.
CAPTION_KINDS = {"open": "Loading {name}", "new": "Creating new map", "resize": "Resizing {name}"}


def scenario_caption(name: str, kind: str, stage: str) -> str:
    return f"{CAPTION_KINDS[kind].format(name=name)}: {SCENARIO_STAGES[stage][0]}"


class LoadProgress(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.caption = QLabel()
        layout.addWidget(self.caption)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedWidth(BAR_WIDTH)
        # Fixed across, or it splits the status bar's free space with its messages; Ignored down, so its
        # 27 px hint never makes the status bar taller (QStatusBar counts hidden widgets too).
        self.bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        self.bar.setFormat("%p%")
        layout.addWidget(self.bar)
        # owner -> (caption, percent), for every running job.
        self._jobs: dict[str, tuple[str, int]] = {}
        self.setVisible(False)

    def start(self, owner: str, caption: str, percent: int = 0) -> None:
        """Starts `owner`'s job, replacing one it still has running."""
        self._check(owner)
        self._jobs[owner] = (caption, _clamp(percent))
        self._show()

    def step(self, owner: str, percent: int, caption: str | None = None) -> None:
        """Moves `owner`'s running job on; a no-op once it has finished."""
        self._check(owner)
        if owner not in self._jobs:
            return
        self._jobs[owner] = (self._jobs[owner][0] if caption is None else caption, _clamp(percent))
        self._show()

    def finish(self, owner: str) -> None:
        self._check(owner)
        if self._jobs.pop(owner, None) is not None:
            self._show()

    def job(self, owner: str) -> tuple[str, int] | None:
        """`owner`'s running job's (caption, percent), or None."""
        self._check(owner)
        return self._jobs.get(owner)

    def shown_owner(self) -> str | None:
        return next((owner for owner in OWNERS if owner in self._jobs), None)

    def _check(self, owner: str) -> None:
        if owner not in OWNERS:
            raise ValueError(f"unknown load progress owner {owner!r}")

    def _show(self) -> None:
        owner = self.shown_owner()
        if owner is None:
            self.setVisible(False)
        else:
            caption, percent = self._jobs[owner]
            self.caption.setText(caption)
            self.bar.setValue(percent)
            self.setVisible(True)
        parent = self.parentWidget()
        if parent is not None and parent.isVisible():
            # A new caption only posts a layout request; deliver just those, so a blocked load lays it out.
            QCoreApplication.sendPostedEvents(self, QEvent.LayoutRequest)
            QCoreApplication.sendPostedEvents(parent, QEvent.LayoutRequest)
            if parent.layout() is not None:
                parent.layout().activate()
            # The parent, not just this widget: showing or hiding it moves its neighbours.
            parent.repaint()


def _clamp(percent: int) -> int:
    return max(0, min(100, int(percent)))
