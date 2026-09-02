#!/usr/bin/env bash
# Runs the corpus/gui pytest tier against the default QUICK_CORPUS_NAMES
# subset (~15 min) -- see tests/README.md and tests/conftest.py. For the
# complete corpus (~27 min), use run_corpus_stress.sh instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

.venv/bin/python3 -m pytest -m "corpus or slow" "$@"
