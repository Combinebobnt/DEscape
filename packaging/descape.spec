# PyInstaller onedir build spec. Run via:
#   .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
# onedir (not onefile) so the bundle's data files stay individually
# inspectable and startup avoids onefile's self-extraction cost.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent

DESCAPE_DIR = ROOT / "descape"

# Enumerated from the source tree rather than hand-listed: a hand list silently
# went stale three times (GH #72), and a data file missing from the bundle is
# an unconditional crash the moment its reader runs.
# Nothing excluded today. versions/DE/v1.21/structure.json used to be (then
# only a schema reference); scenario_io.REPO_VERSIONS_DIR now loads it.
DESCAPE_DATA_JSON_EXCLUDED: set[str] = set()

_json_paths = sorted(DESCAPE_DIR.rglob("*.json"))
_json_rel = {p.relative_to(DESCAPE_DIR).as_posix(): p for p in _json_paths}
# Fail the build on a stale exclusion rather than let the comment above rot.
_stale = DESCAPE_DATA_JSON_EXCLUDED - _json_rel.keys()
if _stale:
    raise SystemExit(f"descape.spec: DESCAPE_DATA_JSON_EXCLUDED names missing files: {sorted(_stale)}")

datas = [
    (str(path), (Path("descape") / rel).parent.as_posix())
    for rel, path in sorted(_json_rel.items())
    if rel not in DESCAPE_DATA_JSON_EXCLUDED
]
datas.append((str(ROOT / "descape" / "templates"), "descape/templates"))
datas.append((str(ROOT / "descape" / "app_icon.png"), "descape"))
# GH #160: the built-in guide's topic files (help_guide.GUIDE_DIR).
datas.append((str(ROOT / "descape" / "help"), "descape/help"))
# GH #58: the data-mod derive subprocess runs tools/gen_*.py, and every cache key
# hashes them and dat_derive.py's own source, so all of them ship as data too.
sys.path.insert(0, str(ROOT))
from descape.dat_derive import source_files  # noqa: E402 -- needs ROOT on sys.path first

datas += [(str(path), "descape" if path.name == "dat_derive.py" else "tools") for path in source_files(ROOT / "tools")]

# Picks up versions/DE/ (the 5.5MB third-party vocabulary tree) plus
# datasets/sources/*.json. xs-check binaries are excluded: 5.2MB, unused --
# their path is only built on instantiation, which never happens here.
datas += collect_data_files("AoE2ScenarioParser", excludes=["**/dependencies/xs-check/*"])

a = Analysis(
    [str(ROOT / "packaging" / "entry_frozen.py")],
    pathex=[str(ROOT)],
    datas=datas,
    # composite_backend imports the kernel inside a try, which analysis can miss.
    # GH #58: the generators above are data, never analysed; argparse is the one import they add.
    hiddenimports=["descape._composite_native", "argparse", "genieutils.datfile"],
    excludes=["tkinter", "pytest", "pytest-qt", "cython"],
)
# Host-supplied by design: a bundled libstdc++ shadows the host's and breaks a
# newer host Mesa dlopened through libGL. tools/check_portability.py guards both halves.
HOST_SUPPLIED = {"libstdc++.so.6", "libgcc_s.so.1"}
a.binaries = [e for e in a.binaries if Path(e[0]).name not in HOST_SUPPLIED]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DEscape",
    # True (not the GUI-app-conventional False) so --self-check's stdout is
    # reliably captured on every platform, including Windows, where a
    # console=False build detaches stdout -- Stage 2's CI verification
    # depends on this. Revisit once a Windows console-window nuisance
    # actually needs fixing; flipping this is a one-line, no-op-elsewhere
    # change.
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, name="DEscape")
