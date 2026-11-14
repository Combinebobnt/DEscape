"""The config keys each Settings tab owns, and the snapshot behind its Cancel and Reset (GH #191).

Every Settings control writes straight through to config.yaml, so the dialog
keeps a copy of its keys as they were at open (or at the last Apply). Cancel
puts that copy back and Reset drops one tab's keys; both clear the memo
globals in descape.settings, so every getter re-reads the file. Qt-free, like
settings_file.py, whose PORTABLE_KEYS tab labels the key lists come from.
"""

from __future__ import annotations

import copy

from descape import settings, settings_file

# The Settings tabs, by the labels settings_file.PORTABLE_KEYS files their keys under.
TABS = ("General", "Mods", "Appearance", "Tools", "Saving", "Keybinds", "Advanced")
# Keys a tab sets that stay on this machine, so PORTABLE_KEYS leaves them out.
_LOCAL_TAB_KEYS = {"xwayland_on_wayland": "General", "autosave_dir": "Saving"}

# Tab label -> the config keys it sets, legacy keys included: a setter pops its legacy key.
DIALOG_TAB_KEYS: dict[str, tuple[str, ...]] = {
    tab: (
        *(key for key, (group, _label) in settings_file.PORTABLE_KEYS.items() if group == tab),
        *(key for key, local_tab in _LOCAL_TAB_KEYS.items() if local_tab == tab),
    )
    for tab in TABS
}
ALL_DIALOG_KEYS: tuple[str, ...] = tuple(key for keys in DIALOG_TAB_KEYS.values() for key in keys)

# A key whose memo globals are not just "_<key>" (or its legacy successor's).
_MEMO_EXCEPTIONS: dict[str, tuple[str, ...]] = {
    "tool_options": ("_tool_option_overrides",),
    "autosave_dir": ("_autosave_dir", "_autosave_dir_refused"),
}


class _Absent:
    """A key config.yaml does not hold. Survives deepcopy as itself, so snapshots compare by identity."""

    def __repr__(self) -> str:
        return "ABSENT"

    def __deepcopy__(self, memo: dict) -> _Absent:
        return self


ABSENT = _Absent()


def snapshot_settings(keys, config: dict | None = None) -> dict[str, object]:
    """key -> a deep copy of its config.yaml value, or ABSENT. One read, unless `config` is given."""
    if config is None:
        config = settings._load_config()
    return {key: copy.deepcopy(config.get(key, ABSENT)) for key in keys}


def differs(snapshot: dict[str, object], config: dict | None = None) -> bool:
    """Whether config.yaml (or `config`, a dict just saved) no longer holds `snapshot`'s values."""
    if config is None:
        config = settings._load_config()
    return any(config.get(key, ABSENT) != value for key, value in snapshot.items())


def restore_settings(snapshot: dict[str, object]) -> None:
    """Puts `snapshot` back in one save: each key set, or dropped where it was ABSENT."""
    config = settings._load_config()
    for key, value in snapshot.items():
        if value is ABSENT:
            config.pop(key, None)
        else:
            config[key] = copy.deepcopy(value)
    settings._save_config(config)
    _clear_memos(snapshot)


def reset_settings(keys) -> None:
    """Drops `keys` in one save, so every getter falls back to its default."""
    config = settings._load_config()
    for key in keys:
        config.pop(key, None)
    settings._save_config(config)
    _clear_memos(keys)


def memo_names(key: str) -> tuple[str, ...]:
    """The descape.settings globals that cache `key`'s value."""
    if key in _MEMO_EXCEPTIONS:
        return _MEMO_EXCEPTIONS[key]
    successor = settings_file.LEGACY_SUCCESSORS.get(key)
    if successor is not None:
        return memo_names(successor[0])
    return (f"_{key}",)


def _clear_memos(keys) -> None:
    for key in keys:
        for name in memo_names(key):
            setattr(settings, name, None)
