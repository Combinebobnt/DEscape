#!/usr/bin/env bash
# Runs the full suite: the quick suite plus the `slow` checks and the
# COMPLETE examples/ corpus, in one parallel run. The run before a release or
# after touching rendering or write-path internals; see tests/README.md for
# its time and tests/conftest.py for the corpus.
#
# Parallel like run_quick_tests.sh, capped at 8 workers. Arguments come last,
# so `-n 0` runs serially, `--maxprocesses 4` suits a machine with less RAM,
# `-m corpus` narrows it and `--scenario-dir=` points it at other files.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

# A tautology rather than -m "": selects every item, and the .bat twin needs no empty-argument quoting.
.venv/bin/python3 -m pytest -n auto --maxprocesses 8 --dist loadgroup -m "corpus or not corpus" --corpus-full "$@"
