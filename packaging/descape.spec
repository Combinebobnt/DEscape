# PyInstaller onedir build spec. Run via:
#   .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
# onedir (not onefile) so the bundle's data files stay individually
# inspectable and startup avoids onefile's self-extraction cost.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent

DESCAPE_DIR = ROOT / "descape"

# Enumerated from the source tree rather than hand-listed: a hand list silently
# went stale three times (GH #72), and a data file missing from the bundle is
# an unconditional crash the moment its reader runs.
# Excluded on purpose: a schema reference cited only in comments and a CLI tool
# arg, with nothing in descape/ constructing a path to it.
DESCAPE_DATA_JSON_EXCLUDED = {"versions/DE/v1.21/structure.json"}

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

# Picks up versions/DE/ (the 5.5MB third-party vocabulary tree) plus
# datasets/sources/*.json. xs-check binaries are excluded: 5.2MB, unused --
# their path is only built on instantiation, which never happens here.
datas += collect_data_files("AoE2ScenarioParser", excludes=["**/dependencies/xs-check/*"])

a = Analysis(
    [str(ROOT / "packaging" / "entry_frozen.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=[],
    excludes=["tkinter", "pytest", "pytest-qt", "cython"],
)
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
