#!/usr/bin/env bash
# Runs the default tier plus the generated test map's `test_map` items and
# the `slow` checks, in parallel like run_all_tests.sh. No examples/ map runs
# here: the real corpus is run_corpus_stress.sh's (~27 min). See
# tests/README.md and tests/conftest.py.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

.venv/bin/python3 -m pytest -n auto --maxprocesses 8 --dist loadgroup -m "not corpus" "$@"
