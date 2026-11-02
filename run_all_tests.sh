#!/usr/bin/env bash
# Alias kept for old instructions: runs run_quick_tests.sh with the same arguments.
echo "run_all_tests is now run_quick_tests" >&2
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_quick_tests.sh" "$@"
