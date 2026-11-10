"""Windows taskbar identity, set before any application window is created."""

from __future__ import annotations

import ctypes
import sys

# Keep stable across releases: Windows pins and jump lists key off this ID.
APP_USER_MODEL_ID = "DEscape.ScenarioEditor"


def set_app_user_model_id(platform: str | None = None, shell32=None) -> str | None:
    """Return a diagnostic on Windows, without letting a shell failure stop startup."""
    if (sys.platform if platform is None else platform) != "win32":
        return None
    try:
        if shell32 is None:
            shell32 = ctypes.windll.shell32
        result = shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        if result != 0:
            return f"Windows taskbar AppUserModelID failed: HRESULT 0x{result & 0xFFFFFFFF:08X}"
    except Exception as exc:  # noqa: BLE001 -- optional shell integration must never prevent startup
        return f"Windows taskbar AppUserModelID failed: {exc}"
    return f"Windows taskbar AppUserModelID set: {APP_USER_MODEL_ID}"
