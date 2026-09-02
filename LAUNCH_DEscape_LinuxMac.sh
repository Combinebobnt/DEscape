#!/usr/bin/env bash
# Double-click-friendly launcher (Linux/macOS): finds Python 3 and hands off
# to bootstrap.py, which creates the venv, installs dependencies, and starts
# the editor (with a progress splash so a double-click launch isn't silent
# while that happens) -- no manual `python -m venv` / `pip install` /
# `source .venv/bin/activate` steps needed. See README.md.
#
# Assumes Python 3 itself is already installed (get it from python.org, or
# your distro's package manager, if not) -- everything downstream of that is
# handled automatically by bootstrap.py.
#
# macOS note: double-clicking a .sh file in Finder usually opens it in a text
# editor rather than running it (Finder doesn't execute plain shell scripts
# by default). Run it from Terminal instead (`./LAUNCH_DEscape_LinuxMac.sh`), or
# rename this file's extension to .command, which Finder does execute on
# double-click.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo
    echo "Python 3 wasn't found. Install it from https://www.python.org/downloads/ (or your OS's package manager), then run this script again."
    read -rp "Press Enter to close this window..." _
    exit 1
fi

exec "$PYTHON" bootstrap.py "$@"
