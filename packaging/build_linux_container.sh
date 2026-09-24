#!/usr/bin/env bash
# Builds the Linux onedir bundle and AppImage inside a pinned AlmaLinux 8
# image, so the artifact's glibc floor is 2.28 rather than the build host's.
# Same script locally (podman) and in CI (CONTAINER_RUNTIME=docker):
#   bash packaging/build_linux_container.sh
# Output lands in dist/ as usual: dist/DEscape/ and dist/DEscape-<version>-x86_64.AppImage.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
# Pinned to a minor, not almalinux:8, for the same reason appimagetool is pinned.
IMAGE="${IMAGE:-docker.io/library/almalinux:8.10}"

# Rootless podman already maps container root to the invoking user; a chown
# to HOST_UID there would hand build/ and dist/ to a subuid. Checked via
# --version so a podman-docker shim named "docker" is caught too.
OWNER_ENV=()
if ! "$CONTAINER_RUNTIME" --version 2>/dev/null | grep -qi podman; then
    OWNER_ENV=(-e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)")
fi

"$CONTAINER_RUNTIME" run --rm -v "$ROOT:/src:Z" -w /src \
    "${OWNER_ENV[@]}" \
    "$IMAGE" bash packaging/container_build.sh
