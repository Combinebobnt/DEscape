"""Tools > Launch in AoE2:DE (GH #4): write the open document to one fixed
file in the game profile's scenario folder, then start the game through
Steam. Qt-free; the viewer side is viewer_game_launch.py.

The copy is always DEscape_launch.aoe2scenario, never the document's own
name, so a user's real file there is never overwritten. An ownership ledger
beside config.yaml records the sha256 DEscape last wrote to each target, so
a launch copy saved over in-game is asked about before it is replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping
from pathlib import Path

from descape import asset_source, content_roots, debug_log, qt_platform, scenario_library, yaml_file_io

LAUNCH_NAME = "DEscape_launch.aoe2scenario"
AOE2DE_APP_ID = content_roots.AOE2DE_STEAM_APP_ID
LAUNCH_ARGS_CONFIG_KEY = "aoe2de_launch_args"
DEFAULT_LAUNCH_ARGS = ("EDITOR",)
GAME_EXE = "AoE2DE_s.exe"
FLATPAK_STEAM_ID = "com.valvesoftware.Steam"
LEDGER_DIRNAME = "launch"
LEDGER_FILENAME = "owned.json"

NO_PROFILE_REFUSAL = (
    "DEscape could not find your AoE2:DE profile folder. Set it in Settings > Mods (profile folder)."
)
NO_SCENARIO_FOLDER_REFUSAL = (
    "Your AoE2:DE profile has no scenario folder yet. Save one scenario from the game's own "
    "editor first: that creates the folder."
)
MACOS_REFUSAL = "AoE2:DE has no macOS build"

_UNSET = object()


def refusal_for_platform(platform: str | None = None) -> str | None:
    """None where the launch works (Linux, Windows), else why not."""
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux") or platform == "win32":
        return None
    if platform == "darwin":
        return MACOS_REFUSAL
    return f"Launching AoE2:DE is not supported on {platform}"


def launch_target(profile=_UNSET) -> tuple[Path | None, str | None]:
    """(the launch copy's path, None), or (None, why there is none).
    `profile` defaults to content_roots.profile_path(); None means no profile."""
    if profile is _UNSET:
        profile = content_roots.profile_path()
    if profile is None:
        return None, NO_PROFILE_REFUSAL
    folder = scenario_library.default_folder(profile)
    if folder is None:
        return None, NO_SCENARIO_FOLDER_REFUSAL
    return folder / LAUNCH_NAME, None


def ledger_path() -> Path:
    # Resolved at call time, like autosave.autosave_dir(): tests redirect asset_source.CONFIG_PATH.
    return asset_source.CONFIG_PATH.parent / LEDGER_DIRNAME / LEDGER_FILENAME


def _key(path: Path) -> str:
    return str(Path(path).resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ledger() -> dict[str, str]:
    path = ledger_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        debug_log.log(f"launch ledger unreadable ({exc!r})", level=debug_log.WARNING)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def ledger_matches(path: Path) -> bool:
    """True when `path` holds exactly what DEscape last wrote there."""
    try:
        actual = _sha256(path)
    except OSError:
        return False
    return _read_ledger().get(_key(path)) == actual


def record_owned(path: Path) -> None:
    """Records `path`'s current bytes as DEscape's own (atomic: never a truncated ledger)."""
    ledger = _read_ledger()
    ledger[_key(path)] = _sha256(path)
    target = ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    yaml_file_io.atomic_write(target, json.dumps(ledger, indent=1, sort_keys=True))


def keep_foreign_name(path: Path, now: float | None = None) -> Path:
    """Where keep_foreign() copies a launch copy changed outside DEscape: beside it, under a
    timestamped name ending in .bak, so the game's Load list doesn't show it."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return path.with_name(f"{path.name}.outside-{stamp}.bak")


def keep_foreign(path: Path) -> Path:
    """Copies `path` to keep_foreign_name() and returns it. Unlike the rolling .bak, a
    later launch never replaces it."""
    kept = keep_foreign_name(path)
    yaml_file_io.atomic_write(kept, path.read_bytes())
    return kept


def write_launch_copy(path: Path, write: Callable[..., object]):
    """`write(str(path), backup=True)`, then the ledger entry. Returns what `write`
    returned. A failed ledger write is logged and ignored: the next launch only asks
    once more. The overwrite question is the caller's (ledger_matches())."""
    result = write(str(path), backup=True)
    try:
        record_owned(path)
    except OSError as exc:
        debug_log.log(f"launch ledger not updated for {path} ({exc!r})", level=debug_log.WARNING)
    return result


def launch_args() -> list[str]:
    """The game's command line after the app id: config key aoe2de_launch_args (one string,
    split like a shell command line), else EDITOR. An empty value is the default too."""
    raw = content_roots._config_value(LAUNCH_ARGS_CONFIG_KEY)
    if raw is None:
        return list(DEFAULT_LAUNCH_ARGS)
    try:
        return shlex.split(raw)
    except ValueError as exc:
        debug_log.log(f"{LAUNCH_ARGS_CONFIG_KEY} unparseable ({exc}), using the default", level=debug_log.WARNING)
        return list(DEFAULT_LAUNCH_ARGS)


def _steam_url(args: list[str]) -> str:
    return f"steam://run/{AOE2DE_APP_ID}//{urllib.parse.quote(shlex.join(args), safe='')}/"


def _winreg_steam_exe() -> str | None:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _kind = winreg.QueryValueEx(key, "SteamExe")
    except OSError:
        return None
    return str(value) if value else None


def _is_flatpak_profile(profile: Path | None) -> bool:
    if profile is None:
        return False
    parts = Path(profile).parts
    return any(parts[i:i + 3] == (".var", "app", FLATPAK_STEAM_ID) for i in range(len(parts)))


def launch_command(
    args: list[str],
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    profile: Path | None = None,
    winreg_lookup: Callable[[], str | None] = _winreg_steam_exe,
) -> list[str] | str:
    """The command that starts AoE2:DE: an argv list, or a steam:// URL string for
    os.startfile() on Windows. Raises ValueError on an unsupported platform."""
    platform = sys.platform if platform is None else platform
    refusal = refusal_for_platform(platform)
    if refusal is not None:
        raise ValueError(refusal)
    if platform == "win32":
        steam = winreg_lookup()
        if steam:
            return [steam, "-applaunch", AOE2DE_APP_ID, *args]
        return _steam_url(args)
    if _is_flatpak_profile(profile):
        flatpak = which("flatpak") or "flatpak"
        return [flatpak, "run", FLATPAK_STEAM_ID, "-applaunch", AOE2DE_APP_ID, *args]
    steam = which("steam")
    if steam:
        return [steam, "-applaunch", AOE2DE_APP_ID, *args]
    return [which("xdg-open") or "xdg-open", _steam_url(args)]


def game_running(proc_root: Path = Path("/proc"), *, platform: str | None = None) -> bool | None:
    """Whether AoE2:DE's exe is running: Linux scans /proc/*/cmdline, elsewhere None (unknown)."""
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        return None
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    needle = GAME_EXE.lower().encode()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in cmdline.lower():
            return True
    return False


def spawn(
    command: list[str] | str,
    environ: Mapping[str, str] | None = None,
    frozen: bool | None = None,
) -> None:
    """Starts `command` detached, all streams on DEVNULL. A str is a URL for os.startfile()."""
    if isinstance(command, str):
        os.startfile(command)  # type: ignore[attr-defined]  # Windows only
        return
    environ = os.environ if environ is None else environ
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    subprocess.Popen(
        command,
        env=qt_platform.system_tool_env(environ, frozen),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
