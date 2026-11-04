#!/usr/bin/env bash
# Runs the quick suite: every test that needs no local data and is not slow,
# the generated test map's `test_map` items included. This is what CI runs.
# For the examples/ corpus and the slow checks too, see run_full_tests.sh.
# See tests/README.md.
#
# Parallel via pytest-xdist: one worker per CPU, capped at 8 so the machine
# stays usable. Arguments come last, so `-n 0` runs serially and
# `--maxprocesses 16` raises the cap.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

.venv/bin/python3 -m pytest -n auto --maxprocesses 8 --dist loadgroup "$@"
