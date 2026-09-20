"""PyInstaller entry point. Deliberately not descape/viewer.py directly --
PyInstaller would load it as __main__, and every `from descape import ...`
elsewhere would then import a second copy of the module. Not descape/, so
this file itself isn't bundled as application code (see descape.spec).

`--self-check` runs before any QApplication exists, so it works headless and
with no display: it verifies the frozen build actually bundled the data
files the app needs at runtime (the dominant freeze bug class, and one that
otherwise fails silently).

`--smoke` goes one step further: it constructs a real `QApplication` +
`ViewerWindow` under `QT_QPA_PLATFORM=offscreen` (so it still works headless)
and exits 0, to catch a missing/unloadable Qt platform plugin that
`--self-check` can't see since it never touches Qt at all.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path

# Frozen build's platform-plugin filename per OS -- what a real user's Qt
# actually loads, as opposed to the offscreen plugin --smoke itself uses.
_PLATFORM_PLUGIN_NAMES = {
    "linux": "libqxcb.so",
    "win32": "qwindows.dll",
    "darwin": "libqcocoa.dylib",
}

# Every version AoE2ScenarioParser==0.8.3 ships under versions/DE/ -- fixed,
# not derived from examples/, since examples/ is gitignored and would exist
# on this machine but not on a CI runner, making the check pass vacuously
# with the whole versions/DE tree missing. Bump this list if the pin moves.
# No "v" prefix -- vocabulary_is_available() prepends it itself.
_EXPECTED_VERSIONS = [
    "1.36", "1.37",
    "1.40", "1.41", "1.42", "1.43", "1.44", "1.45", "1.46", "1.47", "1.48", "1.49",
    "1.51",
    "1.53", "1.54", "1.55", "1.56", "1.57", "1.58",
]


def _self_check() -> int:
    import descape

    failures = []

    for _, name, _ in pkgutil.walk_packages(descape.__path__, prefix="descape."):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            # A module that reads a data file at import time fails here rather
            # than in the path scan below, and FileNotFoundError's repr drops
            # the name -- append it so the culprit file is still identifiable.
            filename = getattr(exc, "filename", None)
            detail = f"{exc!r} [{filename}]" if filename else f"{exc!r}"
            failures.append(f"import {name}: {detail}")

    from descape import asset_source, library_compat

    # Derived from the imported code, never from a hand-maintained list: GH #72
    # shipped because a file absent from the spec's list was also absent from
    # this check, so the check could not possibly catch it. Every module-level
    # Path pointing inside descape/ is a data path the app resolves at runtime.
    package_dir = Path(descape.__path__[0]).resolve()
    checked_paths: dict[Path, str] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "descape" and not module_name.startswith("descape."):
            continue
        for attr_name, value in sorted(vars(module).items()):
            if not isinstance(value, Path):
                continue
            resolved = value.resolve()
            if resolved.is_relative_to(package_dir):
                checked_paths.setdefault(resolved, f"{module_name}.{attr_name}")

    if not checked_paths:
        failures.append("discovered no bundled data paths -- this check itself is broken")

    for path, label in sorted(checked_paths.items()):
        if not path.exists():
            failures.append(f"missing bundled path for {label}: {path}")
        else:
            print(f"OK  {label}: {path}")

    # No exported path constant for this one (asset_source._terrain_texture_map()
    # builds it inline) -- call the real function so the check exercises the
    # actual lookup rather than reconstructing the path by hand.
    try:
        texture_map = asset_source._terrain_texture_map()
        if not texture_map:
            failures.append("asset_source._terrain_texture_map() returned empty")
        else:
            print(f"OK  asset_source._terrain_texture_map() ({len(texture_map)} entries)")
    except Exception as exc:  # noqa: BLE001 -- report, don't stop the rest of the checks
        failures.append(f"asset_source._terrain_texture_map(): {exc!r}")

    for version in _EXPECTED_VERSIONS:
        if not library_compat.vocabulary_is_available(version):
            failures.append(f"vocabulary_is_available({version!r}) is False")
        else:
            print(f"OK  vocabulary_is_available({version!r})")

    if failures:
        print("\nSELF-CHECK FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nSELF-CHECK PASSED")
    return 0


def _smoke() -> int:
    # Before importing PyQt5 or descape.viewer -- must be set before Qt picks
    # a platform plugin. setdefault, not assignment: a real Linux CI leg
    # overrides this with QT_QPA_PLATFORM=xcb to exercise the real plugin.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    failures = []

    if getattr(sys, "frozen", False):
        plugin_name = _PLATFORM_PLUGIN_NAMES.get(sys.platform)
        if plugin_name is None:
            failures.append(f"no known platform-plugin filename for sys.platform={sys.platform!r}")
        else:
            platforms_dirs = list(Path(sys._MEIPASS).rglob("platforms"))
            if not any((d / plugin_name).exists() for d in platforms_dirs):
                failures.append(f"missing bundled platform plugin: {plugin_name}")
            else:
                print(f"OK  platform plugin present: {plugin_name}")
    else:
        print("OK  platform-plugin presence check skipped (running from source tree)")

    from PyQt5.QtWidgets import QApplication

    from descape import settings
    from descape.viewer import ViewerWindow
    from descape.viewer_dialogs import apply_theme

    app = QApplication(sys.argv)
    apply_theme(app, settings.get_dark_mode())
    window = ViewerWindow()
    window.show()
    app.processEvents()
    app.processEvents()

    if failures:
        print("\nSMOKE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.stdout.flush()
        return 1

    print("\nSMOKE PASSED")
    sys.stdout.flush()
    return 0


def main() -> None:
    if "--self-check" in sys.argv:
        sys.exit(_self_check())
    elif "--smoke" in sys.argv:
        sys.exit(_smoke())

    from descape.viewer import main as viewer_main

    viewer_main()


if __name__ == "__main__":
    main()
