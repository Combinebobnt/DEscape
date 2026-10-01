#!/usr/bin/env bash
# Runs the default (fast, no-corpus) pytest tier -- see tests/README.md for
# what that covers. For the corpus/gui tier, see run_corpus_quick.sh
# (~15 min, the default scope) and run_corpus_stress.sh (~27 min, the
# complete corpus).
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
