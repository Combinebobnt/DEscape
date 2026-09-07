#!/usr/bin/env bash
# Wraps the PyInstaller onedir bundle (dist/DEscape/) into a single portable
# DEscape-<version>-x86_64.AppImage with desktop integration. Build the onedir first:
#   .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
#
# Needs a real chmod (AppRun, the downloaded appimagetool) and network
# access to fetch appimagetool, so this script is meant to be run by a
# human, not from an agent session. An in-session pass can still check the
# built AppDir with tools/verify_appdir.py and run ./run_all_tests.sh
# (which covers tests/test_packaging_assets.py); only the download, chmod,
# and appimagetool call itself need a human.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-.venv/bin/python3}"
DIST_EXE="dist/DEscape/DEscape"
APPDIR="build/AppDir"
APPIMAGETOOL="build/appimagetool-x86_64.AppImage"
# Pinned to a specific release (not the moving `continuous` tag) so a build
# run today and one run in a year produce the same appimagetool.
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
VERSION="$("$PY" -c 'import descape; print(descape.__version__)')"
OUT="dist/DEscape-${VERSION}-x86_64.AppImage"

if [ ! -e "$DIST_EXE" ]; then
    echo "error: $DIST_EXE not found -- build the onedir bundle first:" >&2
    echo "  .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm" >&2
    exit 1
fi

echo "==> staging $APPDIR"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/share/doc/DEscape"

cp -a "dist/DEscape" "$APPDIR/usr/bin/DEscape"
cp "packaging/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"
cp "packaging/DEscape.desktop" "$APPDIR/DEscape.desktop"
cp "packaging/DEscape.desktop" "$APPDIR/usr/share/applications/DEscape.desktop"
cp "descape/app_icon.png" "$APPDIR/.DirIcon"
cp "descape/app_icon.png" "$APPDIR/DEscape.png"
cp "descape/app_icon.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/DEscape.png"

echo "==> collecting third-party license texts"
"$PY" tools/collect_licenses.py --strict --out "$APPDIR/usr/share/doc/DEscape/third-party"
cp "LICENSE" "$APPDIR/usr/share/doc/DEscape/LICENSE"

if command -v objdump >/dev/null 2>&1; then
    echo "==> glibc floor (highest symbol version linked by any bundled binary)"
    find "$APPDIR/usr/bin/DEscape" -type f \( -name '*.so' -o -name '*.so.*' -o -perm -u+x \) -print0 \
        | xargs -0 -r objdump -T 2>/dev/null \
        | grep -o 'GLIBC_[0-9.]*' | sort -V | uniq -c | tail -5

    echo "==> libqxcb.so NEEDED (cross-reference against \$APPDIR/usr/bin/DEscape/_internal to find the host-supplied portability contract)"
    QXCB="$APPDIR/usr/bin/DEscape/_internal/PyQt5/Qt5/plugins/platforms/libqxcb.so"
    if [ -f "$QXCB" ]; then
        objdump -p "$QXCB" | grep NEEDED
    else
        echo "warning: $QXCB not found, skipping" >&2
    fi
else
    echo "warning: objdump not found, skipping glibc-floor and libqxcb NEEDED scans" >&2
fi

echo "==> verifying AppDir structure"
"$PY" tools/verify_appdir.py "$APPDIR"

if [ ! -x "$APPIMAGETOOL" ]; then
    echo "==> downloading appimagetool"
    mkdir -p build
    curl -fL -o "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
    chmod +x "$APPIMAGETOOL"
fi

echo "==> building $OUT"
ARCH=x86_64 "$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$OUT"

echo "==> done: $OUT"
