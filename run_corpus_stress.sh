#!/usr/bin/env bash
# Runs the corpus/gui pytest tier against the COMPLETE examples/ corpus
# (~27 min) -- see tests/README.md and tests/conftest.py. For routine
# after-a-feature verification, use run_corpus_quick.sh instead (~15 min,
# the default scope); reach for this before a release or after touching
# rendering/write-path internals.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DESCAPE_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

.venv/bin/python3 -m pytest -m "corpus or slow" --corpus-full "$@"
