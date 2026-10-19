#!/usr/bin/env bash
# Runs the corpus/gui pytest tier against the COMPLETE examples/ corpus
# (~27 min), plus the generated test map's `test_map` items -- see
# tests/README.md and tests/conftest.py. For routine after-a-feature
# verification, use run_corpus_quick.sh instead (default tier plus the test
# map, no examples/); reach for this before a release or after touching
# rendering/write-path internals.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -x ".venv/bin/python3" ]; then
    echo "No .venv/bin/python3 found -- see LAUNCH_DEscape_LinuxMac.sh or README.md to set one up." >&2
    exit 1
fi

.venv/bin/python3 -m pytest -m "corpus or slow or test_map" --corpus-full "$@"
