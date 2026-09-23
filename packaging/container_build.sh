#!/usr/bin/env bash
# Runs INSIDE the pinned AlmaLinux 8 image (glibc 2.28); started by
# packaging/build_linux_container.sh, not meant to be run on a host. Installs
# the toolchain, builds the onedir bundle, then hands off to build_appimage.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Runtime libs so PyInstaller finds and bundles the same set it does on
# Debian; tools/check_portability.py --expect-host-libs fails if one is missing.
dnf -y install \
    python3.11 python3.11-libs python3.11-pip \
    mesa-libGL mesa-libEGL libxkbcommon libxkbcommon-x11 \
    libxcb xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil \
    libX11 libX11-xcb libXext libXrender fontconfig freetype zlib \
    binutils curl file desktop-file-utils

python3.11 -m pip install --upgrade pip
# --only-binary: an sdist fallback would either fail (no compiler) or
# compile against this image and could silently raise the floor.
python3.11 -m pip install --only-binary :all: -r requirements.txt -r packaging/requirements-build.txt

python3.11 -m PyInstaller packaging/descape.spec --noconfirm
PYTHON=python3.11 bash packaging/build_appimage.sh

# Docker writes root-owned files into the bind mount; rootless podman maps to
# the invoking user already, so this is a no-op there.
if [ -n "${HOST_UID:-}" ]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" dist build
fi
