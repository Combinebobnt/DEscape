"""AI personality choices for Players mode (GH #126): scan the .ai markers
every content root offers, and resolve one choice into the exact bytes a
scenario stores for it. Qt-free and read-only.

A personality is four coupled pieces of data (PlayerDataTwo's ai_names entry,
its embedded script text, ai_type, and for a custom AI the Files library
entries). What the AoE2:DE 1.59 editor writes, measured from three in-game
saves and the examples/ corpus:

- Standard stores the name "PromiDE" with a game-generated stub script and
  ai_type 1; None stores "NoneAi" with a 54-byte stub and ai_type 2. Neither
  stub exists as a file on disk, so both are vendored below.
- A custom AI stores its .ai marker's file name as on disk (ai_type 0), and
  embeds the sibling .per verbatim.
- The Files library holds, per custom AI, its .per plus the transitive closure
  of its `(load "X")` targets, keyed "<X>.per" plus one trailing NUL, X as
  written in the load (backslashes kept). `;` comments are skipped and
  `#load-if` conditionals are not evaluated: every branch's loads get entries.
  `(include ...)` lines (xs files) never do.

A choice's includes resolve in its own root first, then the install root (mod
AIs routinely load the install's Promisory\\ files), matching each path
component case-insensitively. A missing include makes the choice
unresolvable, with the missing target named.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from descape import content_roots, debug_log
from descape.content_roots import ContentRoot

AI_TYPE_CUSTOM, AI_TYPE_STANDARD, AI_TYPE_NONE = 0, 1, 2

STANDARD_NAME = "PromiDE"
NONE_NAME = "NoneAi"
KEY_STANDARD = "builtin:standard"
KEY_NONE = "builtin:none"
SOURCE_BUILTIN = "Built-in"

AI_DIR = "ai"
MARKER_SUFFIX = ".ai"
SCRIPT_SUFFIX = ".per"
LIBRARY_KEY_SUFFIX = b".per"
LIBRARY_KEY_TERMINATOR = b"\x00"

# Game-generated stubs, copied from the AoE2:DE 1.59 editor's own save (GH #126 Step 0).
STANDARD_STUB = (
    b'(load "Promisory\\defaultConstants")\r\n'
    b'(load "Promisory\\finalingConstants")\r\n'
    b'(load "Promisory\\customConstants")\r\n'
    b"#load-if-not-defined BATTLE-ROYALE\r\n"
    b'(load "Promisory\\init")\r\n'
    b'(load "Promisory\\threats")\r\n'
    b'(load "Promisory\\escrow")\r\n'
    b'(load "Promisory\\dawn")\r\n'
    b'(load "Promisory\\gatherers")\r\n'
    b'(load "Promisory\\scoutcontrol")\r\n'
    b'(load "Promisory\\tsa")\r\n'
    b'(load "Promisory\\watercontrol")\r\n'
    b'(load "Promisory\\general")\r\n'
    b'(load "Promisory\\orb")\r\n'
    b"#load-if-not-defined DIFFICULTY-EASIEST\r\n"
    b"#load-if-not-defined DIFFICULTY-EASY\r\n"
    b"#load-if-not-defined DIFFICULTY-MODERATE\r\n"
    b"#load-if-not-defined INFINITE-RESOURCES-START\r\n"
    b'(load "Promisory\\boarhunting")\r\n'
    b"#end-if\r\n"
    b"#end-if\r\n"
    b"#end-if\r\n"
    b"#end-if\r\n"
    b'(load "Promisory\\researches")\r\n'
    b'(load "Promisory\\interaction")\r\n'
    b'(load "Promisory\\buildings")\r\n'
    b'(load "Promisory\\units")\r\n'
    b'(load "Promisory\\trade")\r\n'
    b'(load "Promisory\\resign")\r\n'
    b"#else\r\n"
    b'(load "Promisory\\finaling")\r\n'
    b"#end-if\r\n"
    b'(load "Promisory\\event")\r\n'
    b'(include "ailib/GoalArrayOps.xs")\r\n'
    b'(include "ailib/Geometry.xs")\r\n'
)
NONE_STUB = b"(defconst the-maze 0)\r\n(defrule(true)=>(disable-self))"

_LOAD_RE = re.compile(rb'\(\s*load\s+"([^"]*)"')
_LOAD_RANDOM_RE = re.compile(rb"\(\s*load-random\b([^)]*)\)")
_QUOTED_RE = re.compile(rb'"([^"]*)"')


class AiResolveError(Exception):
    """A choice whose script or one of its includes cannot be read."""


@dataclass(frozen=True)
class AiChoice:
    """One Personality combo entry. `text`/`library` are empty until
    resolve() fills them; `library` is ((key incl. NUL, text), ...), the
    choice's own .per first."""

    key: str  # unique across roots; "builtin:standard", "builtin:none", or a root-qualified marker
    stored_name: str  # the ai_names value written for it
    ai_type: int
    source: str  # "Built-in" or the root's label, shown when names collide
    marker: Path | None = None
    root: ContentRoot | None = None
    text: bytes = b""
    library: tuple[tuple[bytes, bytes], ...] = ()
    resolved: bool = False

    @property
    def is_custom(self) -> bool:
        return self.ai_type == AI_TYPE_CUSTOM


STANDARD = AiChoice(KEY_STANDARD, STANDARD_NAME, AI_TYPE_STANDARD, SOURCE_BUILTIN, text=STANDARD_STUB, resolved=True)
NONE = AiChoice(KEY_NONE, NONE_NAME, AI_TYPE_NONE, SOURCE_BUILTIN, text=NONE_STUB, resolved=True)
BUILTINS = (STANDARD, NONE)


def stub_digest(stub: bytes) -> str:
    """sha256 prefix, as the Step 0 findings recorded each stub."""
    return hashlib.sha256(stub).hexdigest()[:16]


def _choice_key(root: ContentRoot, marker: Path) -> str:
    return f"{root.kind}:{root.profile_id or ''}:{root.mod_id or ''}:{marker.name}"


def scan_ai_choices(roots: list[ContentRoot] | None = None) -> list[AiChoice]:
    """Standard and None, then every *.ai marker under each root's
    resources/_common/ai, in root order. Reads no script: resolve() does."""
    if roots is None:
        roots = content_roots.content_roots()
    choices = list(BUILTINS)
    for root, marker in content_roots.find_resources(AI_DIR, (MARKER_SUFFIX,), roots):
        choices.append(AiChoice(_choice_key(root, marker), marker.name, AI_TYPE_CUSTOM, root.label, marker, root))
    return choices


def install_root(roots: list[ContentRoot]) -> ContentRoot | None:
    return next((r for r in roots if r.kind == content_roots.KIND_INSTALL), None)


def strip_comments(text: bytes) -> bytes:
    return b"\n".join(line.split(b";", 1)[0] for line in text.splitlines())


def load_targets(text: bytes) -> list[bytes]:
    """Every `(load "X")` and `(load-random ... "X" ...)` target outside a
    `;` comment, in order. Conditionals are not evaluated."""
    code = strip_comments(text)
    found = [(m.start(), m.group(1)) for m in _LOAD_RE.finditer(code)]
    for m in _LOAD_RANDOM_RE.finditer(code):
        found.extend((m.start(), q) for q in _QUOTED_RE.findall(m.group(1)))
    return [target for _, target in sorted(found, key=lambda item: item[0])]


def library_key(target: bytes) -> bytes:
    return target + LIBRARY_KEY_SUFFIX + LIBRARY_KEY_TERMINATOR


def library_stem(key: bytes) -> bytes:
    """A library key without its NUL and .per/.per2 extension, case-folded:
    the identity two entries share when they name the same script."""
    stem = key.rstrip(LIBRARY_KEY_TERMINATOR)
    lower = stem.lower()
    for ext in (b".per2", b".per"):
        if lower.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.replace(b"/", b"\\").lower()


def _find_ci(folder: Path, parts: list[str]) -> Path | None:
    """`folder`/parts..., each component matched case-insensitively."""
    current = folder
    for part in parts:
        exact = current / part
        if exact.exists():
            current = exact
            continue
        try:
            match = next((p for p in current.iterdir() if p.name.casefold() == part.casefold()), None)
        except OSError:
            return None
        if match is None:
            return None
        current = match
    return current if current.is_file() else None


def _find_script(target: bytes, search: list[Path]) -> Path | None:
    name = target.decode("utf-8", "surrogateescape")
    parts = [p for p in re.split(r"[\\/]+", name + SCRIPT_SUFFIX) if p]
    if not parts or any(p in (".", "..") for p in parts):
        return None
    for folder in search:
        found = _find_ci(folder, parts)
        if found is not None:
            return found
    return None


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise AiResolveError(f"cannot read {path.name}: {exc.strerror or exc}") from exc


def resolve(choice: AiChoice, roots: list[ContentRoot] | None = None) -> AiChoice:
    """`choice` with text and library filled. Built-ins come back as they
    are (vendored stub, empty library). Raises AiResolveError naming the
    missing script or include."""
    if choice.resolved:
        return choice
    if choice.marker is None or choice.root is None:
        raise AiResolveError(f"{choice.stored_name} has no script on disk")
    ai_dir = choice.marker.parent
    main = _find_ci(ai_dir, [choice.marker.stem + SCRIPT_SUFFIX])
    if main is None:
        raise AiResolveError(f"{choice.stored_name}: missing {choice.marker.stem}{SCRIPT_SUFFIX}")
    text = _read(main)

    if roots is None:
        roots = content_roots.content_roots()
    search = [ai_dir]
    install = install_root(roots)
    if install is not None and (install.common / AI_DIR) != ai_dir:
        search.append(install.common / AI_DIR)

    main_key = library_key(choice.marker.stem.encode("utf-8"))
    library = [(main_key, text)]
    seen = {library_stem(main_key)}
    pending = list(reversed(load_targets(text)))
    while pending:
        target = pending.pop()
        stem = library_stem(library_key(target))
        if stem in seen:
            continue
        seen.add(stem)
        path = _find_script(target, search)
        if path is None:
            name = target.decode("utf-8", "replace")
            raise AiResolveError(f"{choice.stored_name}: missing include {name}{SCRIPT_SUFFIX}")
        included = _read(path)
        library.append((library_key(target), included))
        pending.extend(reversed(load_targets(included)))
    debug_log.log(f"ai_scripts: resolved {choice.key} ({len(library)} library entries)")
    return replace(choice, text=text, library=tuple(library), resolved=True)


def stored_choice(key: str, name: str, ai_type: int, text: bytes, library: tuple[tuple[bytes, bytes], ...]) -> AiChoice:
    """A synthetic, already-resolved choice holding a file's own original
    bytes, so selecting "(stored)" or undoing restores them without disk."""
    return AiChoice(key, name, ai_type, "Stored in file", text=text, library=library, resolved=True)


def matches_stored(choice: AiChoice, name: str, ai_type: int) -> bool:
    """Whether a file's stored (name, ai_type) displays as `choice`: exact
    name bytes and the same type class, never a stem or case-fold match."""
    return choice.stored_name == name and choice.ai_type == ai_type


def library_closure(
    names: list[bytes], entries: list[tuple[bytes, bytes]]
) -> set[bytes]:
    """Library stems a set of custom AI main-script stems pulls in, walking
    loads through `entries` (key, text). What the editor keeps on save."""
    by_stem = {library_stem(k): t for k, t in entries}
    want: set[bytes] = set()
    pending = [library_stem(library_key(n)) for n in names]
    while pending:
        stem = pending.pop()
        if stem in want:
            continue
        want.add(stem)
        text = by_stem.get(stem)
        if text is not None:
            pending.extend(library_stem(library_key(t)) for t in load_targets(text))
    return want


def marker_stem(stored_name: str) -> bytes:
    """The main-script stem of a stored ai_names value ('X.ai' -> b'X'; an
    older save's bare name is its own stem)."""
    name = stored_name
    if name.lower().endswith(MARKER_SUFFIX):
        name = name[: -len(MARKER_SUFFIX)]
    return name.encode("utf-8")
