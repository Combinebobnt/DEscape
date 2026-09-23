#!/usr/bin/env python3
"""Pass/fail portability gate for the Linux bundle. Replaces the print-only
objdump scans packaging/build_appimage.sh used to run, which could never fail.

    python3 tools/check_portability.py dist/DEscape --max-glibc 2.28 --max-glibcxx 3.4.25 --expect-host-libs
    python3 tools/check_portability.py dist/DEscape-0.7-x86_64.AppImage --max-glibc 2.28

--max-glibc / --max-glibcxx: every ELF file under each PATH (a directory is
walked, a file is scanned as-is) must reference no GLIBC_ / GLIBCXX_ symbol
version above the given one. Only undefined (required) symbols count. A
static binary, like the AppImage type-2 runtime, has no floor. The GLIBCXX
ceiling exists because libstdc++ is host-supplied (see packaging/descape.spec),
so the oldest target host's libstdc++ is part of the floor too: glibc 2.28's
RHEL 8 ships GLIBCXX_3.4.25.

--expect-host-libs: the directory must be a PyInstaller Qt bundle (it holds
libqxcb.so). Every NEEDED entry of every ELF in it that the bundle does not
provide must be either HOST_SUPPLIED or on PyInstaller's own system-library
exclude list (glibc, libGL/libEGL, libxcb, libwayland, ...). Anything else
means the build image lacked a library PyInstaller would have bundled, which
a CI host that happens to have it installed would never notice. And
HOST_SUPPLIED libraries must not be bundled at all.

Exit status: 0 pass, 1 a check failed, 2 usage or tooling error (objdump
missing). SKIP_PORTABILITY_CHECK=1 skips everything with a warning, for a
machine without binutils.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Mirrors packaging/descape.spec's HOST_SUPPLIED filter.
HOST_SUPPLIED = frozenset({"libstdc++.so.6", "libgcc_s.so.1"})

QXCB_NAME = "libqxcb.so"

_UND_VERSION_RE = re.compile(r"\*UND\*.*?\b(GLIBCXX|GLIBC)_(\d+(?:\.\d+)*)\b")


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def iter_elf_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if is_elf(target) else []
    found = []
    for dirpath, _dirnames, filenames in os.walk(target):
        for name in filenames:
            path = Path(dirpath) / name
            # A symlink's target is scanned under its own name if it lives in the tree.
            if not path.is_symlink() and is_elf(path):
                found.append(path)
    return sorted(found)


def required_versions(path: Path) -> dict[str, tuple[int, ...]]:
    """Highest required GLIBC / GLIBCXX version per prefix; {} for a static binary."""
    proc = subprocess.run(["objdump", "-T", str(path)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if "not a dynamic object" in proc.stderr:
            return {}
        raise RuntimeError(f"objdump -T {path} failed: {proc.stderr.strip()}")
    highest: dict[str, tuple[int, ...]] = {}
    for prefix, text in _UND_VERSION_RE.findall(proc.stdout):
        version = parse_version(text)
        if version > highest.get(prefix, ()):
            highest[prefix] = version
    return highest


def check_floor(targets: list[Path], limits: dict[str, tuple[int, ...]]) -> list[str]:
    per_file: list[tuple[Path, dict[str, tuple[int, ...]]]] = []
    for target in targets:
        per_file.extend((path, required_versions(path)) for path in iter_elf_files(target))
    print(f"check_portability: scanned {len(per_file)} ELF file(s)")

    errors = []
    for prefix, limit in limits.items():
        ranked = sorted(((v[prefix], p) for p, v in per_file if prefix in v), reverse=True)
        if not ranked:
            print(f"  {prefix}: no versioned references")
            continue
        print(f"  {prefix} floor: {format_version(ranked[0][0])} (limit {format_version(limit)}); highest:")
        for version, path in ranked[:5]:
            print(f"    {prefix}_{format_version(version)}  {path}")
        errors.extend(
            f"{path} needs {prefix}_{format_version(version)} > {format_version(limit)}" for version, path in ranked if version > limit
        )
    return errors


def needed_entries(path: Path) -> list[str]:
    proc = subprocess.run(["objdump", "-p", str(path)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"objdump -p {path} failed: {proc.stderr.strip()}")
    return re.findall(r"^\s*NEEDED\s+(\S+)", proc.stdout, re.MULTILINE)


def pyinstaller_excludes(name: str) -> bool:
    """True if PyInstaller deliberately leaves this library to the host system."""
    try:
        from PyInstaller.depend.dylib import include_library
    except ImportError as exc:
        raise RuntimeError("--expect-host-libs needs PyInstaller importable (packaging/requirements-build.txt)") from exc
    return name.startswith("ld-linux") or not include_library(name)


def host_only_needed(root: Path) -> dict[str, Path]:
    """Every NEEDED entry of every ELF under root that nothing in root provides, with one requirer each."""
    present = {p.name for p in root.rglob("*")}
    host_only: dict[str, Path] = {}
    for path in iter_elf_files(root):
        for name in needed_entries(path):
            if name not in present:
                host_only.setdefault(name, path)
    return host_only


def check_host_libs(root: Path) -> list[str]:
    matches = sorted(root.rglob(QXCB_NAME)) if root.is_dir() else []
    if len(matches) != 1:
        return [f"expected exactly one {QXCB_NAME} under {root}, found {len(matches)}"]
    internal = next((p for p in matches[0].parents if p.name == "_internal"), None)
    bundle = internal.parent if internal is not None else root
    host_only = host_only_needed(bundle)
    print(f"check_portability: host-only NEEDED across the bundle: {' '.join(sorted(host_only))}")

    errors = [
        f"{name} (needed by {requirer}) is neither bundled nor a PyInstaller system exclude: missing from the build image?"
        for name, requirer in sorted(host_only.items())
        if name not in HOST_SUPPLIED and not pyinstaller_excludes(name)
    ]
    bundled = {p.name for p in bundle.rglob("*")}
    errors += [f"{name} is bundled but must be host-supplied (packaging/descape.spec)" for name in sorted(HOST_SUPPLIED & bundled)]
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="bundle directory or single ELF/AppImage file")
    parser.add_argument("--max-glibc", help="highest allowed GLIBC_ symbol version, e.g. 2.28")
    parser.add_argument("--max-glibcxx", help="highest allowed GLIBCXX_ symbol version, e.g. 3.4.25")
    parser.add_argument("--expect-host-libs", action="store_true", help="assert the bundle's host-only NEEDED set")
    args = parser.parse_args(argv)

    if not (args.max_glibc or args.max_glibcxx or args.expect_host_libs):
        parser.error("nothing to check: pass --max-glibc, --max-glibcxx and/or --expect-host-libs")
    if os.environ.get("SKIP_PORTABILITY_CHECK") == "1":
        print("check_portability: SKIP_PORTABILITY_CHECK=1, skipping all checks", file=sys.stderr)
        return 0
    if shutil.which("objdump") is None:
        print("check_portability: objdump not found (install binutils, or set SKIP_PORTABILITY_CHECK=1)", file=sys.stderr)
        return 2
    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print(f"check_portability: no such path: {', '.join(missing)}", file=sys.stderr)
        return 2

    limits = {}
    if args.max_glibc:
        limits["GLIBC"] = parse_version(args.max_glibc)
    if args.max_glibcxx:
        limits["GLIBCXX"] = parse_version(args.max_glibcxx)

    try:
        errors = check_floor(args.paths, limits) if limits else []
        if args.expect_host_libs:
            for path in args.paths:
                errors += check_host_libs(path)
    except RuntimeError as exc:
        print(f"check_portability: {exc}", file=sys.stderr)
        return 2

    if errors:
        sys.stdout.flush()
        print(f"check_portability: FAILED ({len(errors)} problem(s))", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("check_portability: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
