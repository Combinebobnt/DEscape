#!/usr/bin/env bash
# Wraps the PyInstaller onedir bundle (dist/DEscape/) into a single portable
# DEscape-<version>-x86_64.AppImage with desktop integration. The normal route
# is packaging/build_linux_container.sh, which builds the onedir inside an
# AlmaLinux 8 image and then runs this script there. Run directly, build the
# onedir first:
#   .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
#
# tools/check_portability.py fails the build if any bundled binary needs a
# glibc above MAX_GLIBC (default 2.28). A native build on a newer distro
# fails that by design; raise MAX_GLIBC or set SKIP_PORTABILITY_CHECK=1 to
# get a local-only AppImage anyway.
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
# glibc 2.28 is RHEL 8 / Debian 10 / Ubuntu 20.04; GLIBCXX_3.4.25 is RHEL 8's
# libstdc++, which the host supplies (see packaging/descape.spec).
MAX_GLIBC="${MAX_GLIBC:-2.28}"
MAX_GLIBCXX="${MAX_GLIBCXX:-3.4.25}"
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

echo "==> portability check (glibc <= $MAX_GLIBC, GLIBCXX <= $MAX_GLIBCXX, host-supplied libs)"
"$PY" tools/check_portability.py "$APPDIR/usr/bin/DEscape" \
    --max-glibc "$MAX_GLIBC" --max-glibcxx "$MAX_GLIBCXX" --expect-host-libs

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

# The type-2 runtime appimagetool prepends has its own floor (1.9.1's is static).
echo "==> portability check on the finished AppImage's runtime"
"$PY" tools/check_portability.py "$OUT" --max-glibc "$MAX_GLIBC"

echo "==> done: $OUT"
