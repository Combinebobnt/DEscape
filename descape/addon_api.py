"""What an addon imports: `AddonCommand`, the `ADDON_COMMANDS` registry and
`ScriptError`.

An addon is a `.py` file or a package in the addons folder. DEscape imports
it at startup, once the user agreed to run it, and the addon registers its
commands while it is imported:

    from descape.addon_api import ADDON_COMMANDS, AddonCommand

    def paint_corner(editor):
        for x in range(5):
            editor.set_terrain(x, 0, 2)

    ADDON_COMMANDS.register(AddonCommand("paint_corner", "Paint Corner", "Paints five tiles", paint_corner))

Each command is a row in Tools > Addons. `run(editor)` runs inside the same
transaction as a script: one undo step, everything put back on an error or a
cancel, and `editor` the whole supported API. `refusal(editor)`, when given,
returns why the command cannot run right now (the row is greyed out with that
reason), or None. It may only read through `editor`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from descape.registry import Registry, ShadowedNameError
from descape.scripting import Editor, ScriptError


@dataclass(frozen=True)
class AddonCommand:
    """One Tools > Addons row. `name` is the stable id (unique across every
    addon), `title` the row's text and `status_tip` its tooltip."""

    name: str
    title: str
    status_tip: str
    run: Callable[[Editor], object]
    refusal: Callable[[Editor], str | None] | None = None

    def __post_init__(self) -> None:
        # Raised at registration, inside the addon's import, so a malformed command fails its addon, not the menu.
        for field_name in ("name", "title", "status_tip"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"AddonCommand.{field_name} must be a str, got {getattr(self, field_name)!r}")
        if not self.title:
            raise ValueError("AddonCommand.title must not be empty")
        if not callable(self.run):
            raise TypeError(f"AddonCommand.run must be callable, got {self.run!r}")
        if self.refusal is not None and not callable(self.refusal):
            raise TypeError(f"AddonCommand.refusal must be callable or None, got {self.refusal!r}")


ADDON_COMMANDS = Registry("addon command")


__all__ = ["ADDON_COMMANDS", "AddonCommand", "ScriptError", "ShadowedNameError"]
