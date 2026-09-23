#!/usr/bin/env bash
# One command for a CI-equivalent local run. Run it before every tag:
#
#     bash tools/ci_parity.sh v0.8     # tag check, then the default tier
#     bash tools/ci_parity.sh          # the default tier only
#
# Hides what CI's runner does not have: the Workshop corpus (under $HOME), the
# AoE2:DE install (AOE2DE_INSTALL_PATH, or config.yaml under XDG_CONFIG_HOME)
# and examples/ (an empty --scenario-dir). QT_QPA_PLATFORM is cleared so the
# test bootstrap picks offscreen, as it does on CI. QT_FONT_DPI needs nothing
# here: testkit/qt_window.ensure_qapp() hard-sets it.
#
# This matches build.yml's test step only until that workflow changes, so it
# does not replace a real CI run on a release branch before tagging.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi
if [ "$#" -gt 1 ]; then
    echo "usage: tools/ci_parity.sh [tag]" >&2
    exit 1
fi

# Same order as build.yml: the cheap tag check fails fast, before the tier.
if [ "$#" -eq 1 ]; then
    .venv/bin/python3 tools/check_release_tag.py "$1" || exit 1
else
    echo "ci_parity: no tag given, skipping check_release_tag.py"
fi

sandbox="$(mktemp -d "${TMPDIR:-/tmp}/descape-ci-parity.XXXXXX")"
trap 'rm -rf "$sandbox"' EXIT
mkdir -p "$sandbox/home" "$sandbox/config" "$sandbox/scenarios"

HOME="$sandbox/home" XDG_CONFIG_HOME="$sandbox/config" \
    env -u AOE2DE_INSTALL_PATH -u QT_QPA_PLATFORM \
    .venv/bin/python3 -m pytest "--scenario-dir=$sandbox/scenarios"
status=$?
if [ "$status" -eq 0 ]; then
    echo "ci_parity: clean"
else
    echo "ci_parity: default tier failed (exit $status)" >&2
fi
exit "$status"
