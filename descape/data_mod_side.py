"""A data mod's side files, the JSON beside its .dat (GH #263).

civilizations.json names the mod's civs: the list index is the civ id, and a
1.56+ scenario stores `data_name`. eras.json names the five ages of each era a
civ can name. Both are read at runtime, never derived: DataModTables.side_key
hashes them, so an edit reaches every game_dataset.cache loader on the next
pick or open with no .dat re-parse. Missing, unreadable or wrong-shaped files
count as absent, and the caller keeps its AoE2 behaviour. Read-only and Qt-free.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from descape import asset_source, content_roots, debug_log, game_dataset

CIVILIZATIONS_FILE = "civilizations.json"
ERAS_FILE = "eras.json"
SIDE_FILES = (CIVILIZATIONS_FILE, ERAS_FILE)
AGE_COUNT = 5
# StartingAge's value of the first of an era's ages.
FIRST_AGE_VALUE = 2


@dataclass(frozen=True)
class ModCiv:
    index: int
    token: str
    name_string_id: int | None
    internal_name: str
    era: str


def side_dir(root: Path) -> Path | None:
    """The folder a data mod at `root` keeps its .dat and side files in, or None."""
    return content_roots.child_dir(root, "resources", "_common", "dat")


def side_file(folder: Path, name: str) -> Path | None:
    """folder/`name`, matched case-insensitively with an exact match first, or None."""
    exact = folder / name
    if exact.is_file():
        return exact
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return None
    return next((p for p in entries if p.name.casefold() == name and p.is_file()), None)


def fingerprint(folder: Path | None) -> str:
    """A hash over each readable side file's name and bytes; "" when there is none. Never raises."""
    if folder is None:
        return ""
    digest = hashlib.sha256()
    found = False
    for name in SIDE_FILES:
        path = side_file(folder, name)
        if path is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        found = True
        digest.update(name.encode("utf-8") + b"\0" + data + b"\0")
    return digest.hexdigest() if found else ""


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_civilizations(path: Path) -> tuple[ModCiv, ...] | None:
    """A civilizations.json's civs by list position, or None when it is unreadable or wrong-shaped."""
    try:
        rows = _read_json(path)["civilization_list"]
        civs = []
        for index, row in enumerate(rows):
            token = row["data_name"]
            string_id = row.get("name_string_id")
            internal_name = row.get("internal_name", "")
            era = row.get("era", "")
            if not (isinstance(token, str) and token and isinstance(internal_name, str) and isinstance(era, str)):
                raise ValueError(f"civ {index} has a wrong field type")
            civs.append(ModCiv(index, token, string_id if _is_int(string_id) else None, internal_name, era))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        debug_log.log(f"data mod side file: {path} ignored ({exc!r})")
        return None
    # GAIA alone leaves no civ to list.
    return tuple(civs) if len(civs) >= 2 else None


def parse_eras(path: Path) -> dict[str, tuple[int, ...]] | None:
    """An eras.json's era name -> its NameIds, or None when it is unreadable or
    wrong-shaped. An era without exactly AGE_COUNT int ids is left out."""
    try:
        eras = {}
        for row in _read_json(path):
            ids = tuple(age["NameId"] for age in row["Ages"])
            if len(ids) == AGE_COUNT and all(_is_int(id_) for id_ in ids) and isinstance(row["Name"], str):
                eras[row["Name"]] = ids
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        debug_log.log(f"data mod side file: {path} ignored ({exc!r})")
        return None
    return eras or None


def _active_file(name: str) -> Path | None:
    mod = game_dataset.data_mod()
    folder = None if mod is None else side_dir(mod.root)
    return None if folder is None else side_file(folder, name)


@game_dataset.cache(maxsize=1)
def civilizations() -> tuple[ModCiv, ...] | None:
    """The civs of the data mod in effect, GAIA (index 0) included; None with no mod or no usable file."""
    path = _active_file(CIVILIZATIONS_FILE)
    return None if path is None else parse_civilizations(path)


@game_dataset.cache(maxsize=1)
def _eras() -> dict[str, tuple[int, ...]] | None:
    path = _active_file(ERAS_FILE)
    return None if path is None else parse_eras(path)


def find_civ(value: object) -> ModCiv | None:
    """The active mod's civ a raw civilization value names (a token, or an int below 1.56), or None."""
    civs = civilizations()
    if civs is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return next((civ for civ in civs if civ.token == value), None)
    return civs[value] if isinstance(value, int) and 0 <= value < len(civs) else None


def civ_name(civ: ModCiv, lang: str | None = None) -> str:
    """`civ`'s name through the layered strings, so a mod's own win; then its internal name, then its token."""
    name = None if civ.name_string_id is None else asset_source.resource_string(civ.name_string_id, lang)
    return name or civ.internal_name or civ.token


def playable_civs(lang: str | None = None) -> tuple[tuple[ModCiv, str], ...]:
    """(civ, name) for the active mod's civs without GAIA, sorted by name; empty with none."""
    civs = civilizations()
    if civs is None:
        return ()
    lang = lang or asset_source.get_language()  # a YAML read per call, so once per list
    return tuple(sorted(((civ, civ_name(civ, lang)) for civ in civs[1:]), key=lambda pair: pair[1]))


def _common_era(civs: tuple[ModCiv, ...]) -> str:
    """The era most playable civs share, the earlier-listed on a tie."""
    return Counter(civ.era for civ in civs[1:]).most_common(1)[0][0]


def age_labels(civ_value: object = None, lang: str | None = None) -> tuple[str, ...] | None:
    """The five age names for the era of the civ `civ_value` names, or of the
    era most civs share when it names none (a random or unknown civ). None
    without civs and eras, or when the era or one of its names is missing."""
    civs = civilizations()
    eras = _eras()
    if civs is None or eras is None:
        return None
    civ = find_civ(civ_value)
    ids = eras.get(civ.era if civ is not None else _common_era(civs))
    if ids is None:
        return None
    lang = lang or asset_source.get_language()  # a YAML read per call, so once per list
    labels = tuple(asset_source.resource_string(id_, lang) for id_ in ids)
    return None if not all(labels) else labels
