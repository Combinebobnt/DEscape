#!/usr/bin/env python3
"""Structural verification of a staged AppImage AppDir -- the part that
needs no exec permission, so it runs in a session that can't chmod. Run
after packaging/build_appimage.sh stages build/AppDir/, before appimagetool
is invoked:

    .venv/bin/python3 tools/verify_appdir.py build/AppDir

Checks: every path in the layout table exists; AppRun's exec target
resolves to a real file inside the AppDir; DEscape.desktop parses and
carries the required keys; both icon copies are byte-identical 256x256
RGBA PNGs; usr/share/doc/DEscape/ is non-empty and contains LICENSE; and no
config.yaml leaked in anywhere (the repo-root config.yaml is gitignored but
present on real dev machines and holds a personal path -- Stage 1's
explicit `datas` list should never pick it up, but this is one cheap
assertion instead of a code review every time).
"""

from __future__ import annotations

import configparser
import re
import sys
from pathlib import Path

from PIL import Image

REQUIRED_PATHS = [
    "AppRun",
    "DEscape.desktop",
    "DEscape.png",
    ".DirIcon",
    "usr/bin/DEscape/DEscape",
    "usr/share/applications/DEscape.desktop",
    "usr/share/icons/hicolor/256x256/apps/DEscape.png",
    "usr/share/doc/DEscape/LICENSE",
]

REQUIRED_DESKTOP_KEYS = ["Type", "Name", "Exec", "Icon", "Categories"]

ICON_PATHS = [".DirIcon", "DEscape.png", "usr/share/icons/hicolor/256x256/apps/DEscape.png"]


def _check_paths(appdir: Path) -> list[str]:
    return [p for p in REQUIRED_PATHS if not (appdir / p).is_file()]


def _check_apprun_target(appdir: Path) -> str | None:
    apprun = (appdir / "AppRun").read_text()
    match = re.search(r'exec\s+"\$HERE/([^"]+)"', apprun)
    if match is None:
        return "AppRun does not exec a $HERE-relative path"
    target = appdir / match.group(1)
    if not target.is_file():
        return f"AppRun execs {match.group(1)}, which does not exist under {appdir}"
    return None


def _check_desktop_file(appdir: Path) -> list[str]:
    errors = []
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read(appdir / "DEscape.desktop")
    except configparser.Error as exc:
        return [f"DEscape.desktop failed to parse: {exc}"]
    if "Desktop Entry" not in parser:
        return ["DEscape.desktop has no [Desktop Entry] section"]
    section = parser["Desktop Entry"]
    for key in REQUIRED_DESKTOP_KEYS:
        if key not in section or not section[key].strip():
            errors.append(f"DEscape.desktop missing key {key!r}")
    return errors


def _check_icons(appdir: Path) -> list[str]:
    errors = []
    images = {}
    for rel in ICON_PATHS:
        path = appdir / rel
        if not path.is_file():
            continue
        with Image.open(path) as img:
            img.load()
            if img.mode != "RGBA":
                errors.append(f"{rel}: expected RGBA, got {img.mode}")
            if img.size != (256, 256):
                errors.append(f"{rel}: expected 256x256, got {img.size}")
        images[rel] = path.read_bytes()
    distinct = set(images.values())
    if len(distinct) > 1:
        errors.append(f"icon copies are not byte-identical: {sorted(images)}")
    return errors


def _check_no_config_leak(appdir: Path) -> list[str]:
    return [str(p) for p in appdir.rglob("config.yaml")]


def verify(appdir: Path) -> list[str]:
    errors = []
    missing = _check_paths(appdir)
    if missing:
        errors += [f"missing required path: {m}" for m in missing]
        # Everything below assumes these paths exist; skip rather than
        # cascade unrelated failures from a bad staging run.
        return errors

    apprun_error = _check_apprun_target(appdir)
    if apprun_error:
        errors.append(apprun_error)

    errors += _check_desktop_file(appdir)
    errors += _check_icons(appdir)

    doc_dir = appdir / "usr" / "share" / "doc" / "DEscape"
    if not any(doc_dir.iterdir()):
        errors.append(f"{doc_dir} is empty")

    leaked = _check_no_config_leak(appdir)
    errors += [f"leaked config.yaml found at {p}" for p in leaked]

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_appdir.py <appdir>", file=sys.stderr)
        return 2
    appdir = Path(sys.argv[1])
    if not appdir.is_dir():
        print(f"no such directory: {appdir}", file=sys.stderr)
        return 2

    errors = verify(appdir)
    if errors:
        for err in errors:
            print(f"verify_appdir: {err}", file=sys.stderr)
        return 1
    print(f"verify_appdir: {appdir} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
