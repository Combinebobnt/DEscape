#!/usr/bin/env bash
# Runs INSIDE the pinned AlmaLinux 8 image (glibc 2.28); started by
# packaging/build_linux_container.sh, not meant to be run on a host. Installs
# the toolchain, builds the onedir bundle, then hands off to build_appimage.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The checkout is bind-mounted, and the CI host builds the native kernel in
# place first for its own tests. That build links against glibc 2.35 and has
# the same file name, so build_ext would call it current and PyInstaller
# would bundle it. Drop it (and its stamps) so the kernel below is built here.
# A local run leaves the container's build in the checkout; it runs on the host too.
rm -f descape/_composite_native*.c descape/_composite_native*.so \
    descape/_composite_native.stamp descape/_composite_native.failed

# Runtime libs so PyInstaller finds and bundles the same set it does on
# Debian; tools/check_portability.py --expect-host-libs fails if one is missing.
dnf -y install \
    python3.11 python3.11-libs python3.11-pip \
    mesa-libGL mesa-libEGL libxkbcommon libxkbcommon-x11 \
    libxcb xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil \
    libX11 libX11-xcb libXext libXrender libXcomposite fontconfig freetype zlib \
    binutils curl file desktop-file-utils \
    gcc python3.11-devel

python3.11 -m pip install --upgrade pip
# --only-binary: an sdist fallback would compile against this image and could
# silently raise the floor.
python3.11 -m pip install --only-binary :all: -r requirements.txt -r packaging/requirements-build.txt

# setuptools and cython come from requirements-build.txt's pins, installed above.
python3.11 tools/build_native.py --force
python3.11 -m PyInstaller packaging/descape.spec --noconfirm
PYTHON=python3.11 bash packaging/build_appimage.sh

# Docker writes root-owned files into the bind mount. Rootless podman maps root
# to the invoking user already, so build_linux_container.sh doesn't set HOST_UID.
if [ -n "${HOST_UID:-}" ]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" dist build descape/_composite_native*
fi
