# PyInstaller onedir build spec. Run via:
#   .venv/bin/python3 -m PyInstaller packaging/descape.spec --noconfirm
# onedir (not onefile) so the bundle's data files stay individually
# inspectable and startup avoids onefile's self-extraction cost.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent

# The five JSON files descape/ actually reads at runtime (not
# descape/versions/DE/v1.21/structure.json -- nothing constructs a path to
# it; it's a schema reference cited only in comments and a CLI tool arg).
DESCAPE_DATA_JSON = [
    "object_catalog.json",
    "unit_graphic_map.json",
    "unit_render_data.json",
    "terrain_texture_map.json",
    "tree_unit_ids.json",
]

datas = [(str(ROOT / "descape" / name), "descape") for name in DESCAPE_DATA_JSON]
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
