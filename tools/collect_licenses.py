#!/usr/bin/env python3
"""Walks a venv's installed-package metadata and copies each package's
license text into an output directory, for bundling into a frozen
distribution's AppDir/dist tree (GPL-3.0 requires carrying the license
texts of bundled dependencies). Reusable across platforms -- not
AppImage-specific -- so it lives in tools/, not packaging/.

    .venv/bin/python3 tools/collect_licenses.py --out build/AppDir/usr/share/doc/DEscape/third-party
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_LICENSE_FILE_RE = re.compile(r"^(LICENSE|COPYING)([._-].*)?$", re.IGNORECASE)


def _dist_name(dist_info: Path) -> str:
    # "PyQt5-5.15.11.dist-info" -> "PyQt5"
    return dist_info.name.removesuffix(".dist-info").rsplit("-", 1)[0]


def _license_files(dist_info: Path) -> list[Path]:
    # PEP 639 wheels put license files under dist-info/licenses/ instead of
    # dist-info's own top level (numpy, Pillow both do this now).
    licenses_dir = dist_info / "licenses"
    search_dirs = [dist_info, licenses_dir] if licenses_dir.is_dir() else [dist_info]
    found = []
    for d in search_dirs:
        found += [p for p in d.rglob("*") if p.is_file() and _LICENSE_FILE_RE.match(p.name)]
    return found


def _license_from_metadata(dist_info: Path) -> str | None:
    metadata = dist_info / "METADATA"
    if not metadata.is_file():
        return None
    text = metadata.read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("License:") and line.split(":", 1)[1].strip():
            return line
        if line.startswith("Classifier: License ::"):
            return line
    return None


def collect(site_packages: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        name = _dist_name(dist_info)
        license_files = _license_files(dist_info)
        if license_files:
            for src in license_files:
                dest = out_dir / f"{name}-{src.name}"
                dest.write_bytes(src.read_bytes())
            continue
        classifier = _license_from_metadata(dist_info)
        if classifier is not None:
            (out_dir / f"{name}-LICENSE.txt").write_text(classifier + "\n")
        else:
            missing.append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-packages", type=Path, default=ROOT / ".venv" / "lib" / "python3.11" / "site-packages")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not args.site_packages.is_dir():
        raise SystemExit(f"no such site-packages dir: {args.site_packages}")

    missing = collect(args.site_packages, args.out)
    if missing:
        print("collect_licenses: no license text or classifier found for:", ", ".join(missing), file=sys.stderr)
    print(f"collect_licenses: wrote license texts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
