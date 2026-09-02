"""Command-line target expansion shared by the tools/ census scripts.

Lives here rather than in one of the scripts because tools/ has no
__init__.py and its scripts can't import each other; both do put the repo
root on sys.path, which makes this package reachable from either.
"""

from __future__ import annotations

from pathlib import Path


def collect_files(targets: list[Path], on_missing) -> list[Path]:
    """Directories recurse for *.aoe2scenario; explicitly named files are taken as-is
    whatever their extension (examples/play_Test has none)."""
    files: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        # A mistyped path would otherwise be reported as LOADFAIL -- a legitimate
        # result category here -- and exit 0, which reads as "measured, unloadable"
        # rather than "you named a file that isn't there".
        if not target.exists():
            on_missing(f"no such file or directory: {target}")
        found = sorted(target.rglob("*.aoe2scenario")) if target.is_dir() else [target]
        for path in found:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return sorted(files)
