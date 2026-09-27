"""GH #126 Step 2: descape/ai_scripts.py, the personality scanner and
resolver, against fake content-root trees under tmp_path. The corpus test
at the bottom pins the load-closure rule against every real Files library."""

from __future__ import annotations

import inspect
import re
import struct
from pathlib import Path

import pytest

from descape import ai_scripts as ai
from descape import content_roots, player_fields
from descape.content_roots import KIND_INSTALL, KIND_MOD_SUBSCRIBED, ContentRoot
from descape.scenario_io import load_map_and_units, parse_triggers


def _touch(path: Path, data: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _root(tmp_path: Path, name: str, kind: str = KIND_INSTALL) -> ContentRoot:
    base = tmp_path / name
    (base / "resources" / "_common" / "ai").mkdir(parents=True, exist_ok=True)
    if kind == KIND_INSTALL:
        return ContentRoot(kind, "Install", base)
    return ContentRoot(kind, f"Mod (subscribed): {name}", base, profile_id="0", mod_id=name)


def _ai(root: ContentRoot) -> Path:
    return root.common / "ai"


def _choice(choices, name, root=None):
    return next(c for c in choices if c.stored_name == name and (root is None or c.root == root))


# -- stubs -------------------------------------------------------------------


def test_vendored_stubs_match_the_step0_capture() -> None:
    # sha256 prefixes and sizes measured from the 1.59 editor's own save.
    assert (len(ai.STANDARD_STUB), ai.stub_digest(ai.STANDARD_STUB)) == (969, "91b6b898e1f59baf")
    assert (len(ai.NONE_STUB), ai.stub_digest(ai.NONE_STUB)) == (54, "9c35b35e0992cbd8")


def test_builtins_are_resolved_with_an_empty_library() -> None:
    assert [(c.stored_name, c.ai_type) for c in ai.BUILTINS] == [("PromiDE", 1), ("NoneAi", 2)]
    for choice in ai.BUILTINS:
        assert ai.resolve(choice, []) is choice
        assert choice.library == ()


# -- scan --------------------------------------------------------------------


def test_scan_lists_builtins_then_markers_per_root(tmp_path) -> None:
    install = _root(tmp_path, "install")
    mod = _root(tmp_path, "mod", KIND_MOD_SUBSCRIBED)
    _touch(_ai(install) / "E3-p2.ai")
    _touch(_ai(install) / "Upper.Ai")
    _touch(_ai(install) / "E3-p2.per", b"x")  # a script is not a choice
    _touch(_ai(install) / "Promisory" / "inner.ai")  # subfolders are includes, not choices
    _touch(_ai(mod) / "Mod AI.ai")
    choices = ai.scan_ai_choices([install, mod])
    assert [c.stored_name for c in choices] == ["PromiDE", "NoneAi", "E3-p2.ai", "Upper.Ai", "Mod AI.ai"]
    assert all(not c.resolved and c.text == b"" for c in choices[2:])
    assert choices[-1].source == mod.label


def test_duplicate_names_across_roots_get_distinct_keys(tmp_path) -> None:
    a = _root(tmp_path, "a", KIND_MOD_SUBSCRIBED)
    b = _root(tmp_path, "b", KIND_MOD_SUBSCRIBED)
    for root, body in ((a, b"one"), (b, b"two")):
        _touch(_ai(root) / "Same.ai")
        _touch(_ai(root) / "Same.per", body)
    choices = [c for c in ai.scan_ai_choices([a, b]) if c.stored_name == "Same.ai"]
    assert len(choices) == 2 and choices[0].key != choices[1].key
    assert [ai.resolve(c, [a, b]).text for c in choices] == [b"one", b"two"]


# -- resolve -----------------------------------------------------------------


def test_resolve_walks_the_load_closure_transitively(tmp_path) -> None:
    install = _root(tmp_path, "install")
    _touch(_ai(install) / "Main.ai")
    _touch(_ai(install) / "Main.per", b'(load "Lib\\first")\r\n(load "shared")\r\n')
    _touch(_ai(install) / "Lib" / "first.per", b'(load "Lib\\second")\r\n(load "shared")')
    _touch(_ai(install) / "Lib" / "second.per", b"(defrule)")
    _touch(_ai(install) / "shared.per", b"(defconst x 1)")
    choice = ai.resolve(_choice(ai.scan_ai_choices([install]), "Main.ai"), [install])
    assert choice.text == (_ai(install) / "Main.per").read_bytes()
    assert [k for k, _ in choice.library] == [
        b"Main.per\x00",
        b"Lib\\first.per\x00",
        b"Lib\\second.per\x00",
        b"shared.per\x00",
    ]
    assert dict(choice.library)[b"Lib\\second.per\x00"] == b"(defrule)"


def test_comments_are_skipped_but_every_conditional_branch_loads(tmp_path) -> None:
    install = _root(tmp_path, "install")
    _touch(_ai(install) / "Main.ai")
    _touch(
        _ai(install) / "Main.per",
        b';(load "commented")\r\n'
        b'(defrule (true) => (chat-to-all "x")) ; (load "trailing")\r\n'
        b"#load-if-defined FOO\r\n"
        b'(load "then")\r\n'
        b"#else\r\n"
        b'(load "otherwise")\r\n'
        b"#end-if\r\n"
        b'(include "ailib/Geometry.xs")\r\n',
    )
    for name in ("then", "otherwise"):
        _touch(_ai(install) / f"{name}.per", b"")
    library = ai.resolve(_choice(ai.scan_ai_choices([install]), "Main.ai"), [install]).library
    assert [k for k, _ in library] == [b"Main.per\x00", b"then.per\x00", b"otherwise.per\x00"]


def test_load_random_targets_are_followed() -> None:
    assert ai.load_targets(b'(load-random 20 "a" 30 "b" "c")\n(load "d")') == [b"a", b"b", b"c", b"d"]


def test_missing_include_makes_the_choice_unresolvable(tmp_path) -> None:
    install = _root(tmp_path, "install")
    _touch(_ai(install) / "Main.ai")
    _touch(_ai(install) / "Main.per", b'(load "gone")')
    with pytest.raises(ai.AiResolveError, match=r"missing include gone\.per"):
        ai.resolve(_choice(ai.scan_ai_choices([install]), "Main.ai"), [install])


def test_marker_without_a_script_is_unresolvable(tmp_path) -> None:
    install = _root(tmp_path, "install")
    _touch(_ai(install) / "Orphan.ai")
    with pytest.raises(ai.AiResolveError, match=r"missing Orphan\.per"):
        ai.resolve(_choice(ai.scan_ai_choices([install]), "Orphan.ai"), [install])


def test_includes_resolve_case_insensitively_and_fall_back_to_the_install(tmp_path) -> None:
    install = _root(tmp_path, "install")
    mod = _root(tmp_path, "mod", KIND_MOD_SUBSCRIBED)
    _touch(_ai(install) / "Promisory" / "DefaultConstants.per", b"install copy")
    _touch(_ai(install) / "local.per", b"install local")
    _touch(_ai(mod) / "local.per", b"mod local")
    _touch(_ai(mod) / "Mod AI.ai")
    _touch(_ai(mod) / "Mod AI.per", b'(load "promisory\\defaultconstants")\n(load "LOCAL")')
    choice = ai.resolve(_choice(ai.scan_ai_choices([install, mod]), "Mod AI.ai"), [install, mod])
    # Keys keep the load's own spelling; the mod's own copy wins over the install's.
    assert choice.library == (
        (b"Mod AI.per\x00", b'(load "promisory\\defaultconstants")\n(load "LOCAL")'),
        (b"promisory\\defaultconstants.per\x00", b"install copy"),
        (b"LOCAL.per\x00", b"mod local"),
    )


def test_path_escapes_are_refused(tmp_path) -> None:
    install = _root(tmp_path, "install")
    _touch(tmp_path / "outside.per", b"secret")
    _touch(_ai(install) / "Main.ai")
    _touch(_ai(install) / "Main.per", b'(load "..\\..\\..\\outside")')
    with pytest.raises(ai.AiResolveError):
        ai.resolve(_choice(ai.scan_ai_choices([install]), "Main.ai"), [install])


def test_library_stem_ignores_extension_nul_and_case() -> None:
    assert ai.library_stem(b"Scenario Market.per2\x00") == ai.library_stem(b"scenario market.per\x00")
    assert ai.library_stem(b"AiBuilder/x.per\x00") == ai.library_stem(b"AiBuilder\\x.per\x00")
    assert ai.marker_stem("LeLoi-scn5 player3.Ai") == b"LeLoi-scn5 player3"
    assert ai.marker_stem("Attila-scn1 player 2") == b"Attila-scn1 player 2"


def test_module_never_writes() -> None:
    source = inspect.getsource(ai)
    assert not re.search(r"write_(text|bytes)|\bopen\(|\.unlink\(|\.mkdir\(|shutil", source)


# -- corpus: the load-closure rule ------------------------------------------


def _library_entries(loaded) -> list[tuple[bytes, bytes]] | None:
    parse_triggers(loaded)
    layout = player_fields.ai_library_layout(loaded)
    if layout is None:
        return None
    body = loaded.decompressed_body
    entries = []
    for entry in layout.entries:
        pos = entry.span.offset + 4 + len(entry.name)
        (length,) = struct.unpack_from("<i", body, pos)
        entries.append((entry.name, bytes(body[pos + 4 : pos + 4 + length])))
    return entries


@pytest.mark.corpus
def test_library_is_the_load_closure_of_every_custom_row(scenario_path) -> None:
    """Every P1..P8 ai_type-0 row counts, active or not (R4_LeLoi_4's inactive
    P7 keeps its closure), and each main script equals the embedded text,
    modulo one trailing NUL some 1.41 saves append to every library text."""
    loaded = load_map_and_units(scenario_path)
    entries = _library_entries(loaded)
    layout = player_fields.personality_layout(loaded)
    if entries is None or layout is None:
        pytest.skip("no verified Files library or personality block")
    retrievers = loaded._scenario.sections["PlayerDataTwo"].retriever_map
    body = loaded.decompressed_body
    custom = [
        (retrievers["ai_names"].data[i], layout.ai_files[i])
        for i in range(player_fields.NUM_PLAYERS)
        if retrievers["ai_type"].data[i] == ai.AI_TYPE_CUSTOM and retrievers["ai_names"].data[i]
    ]
    stems = ai.library_closure([ai.marker_stem(name) for name, _ in custom], entries)
    assert stems == {ai.library_stem(k) for k, _ in entries}
    by_stem = {ai.library_stem(k): t for k, t in entries}
    for name, span in custom:
        (length,) = struct.unpack_from("<i", body, span.offset + 8)
        embedded = bytes(body[span.offset + 12 : span.offset + 12 + length])
        stored = by_stem[ai.library_stem(ai.library_key(ai.marker_stem(name)))]
        assert stored in (embedded, embedded + b"\x00"), name


# -- real install smoke (plan Verification 5) --------------------------------


@pytest.mark.corpus
def test_real_install_scan_lists_install_and_mod_ais() -> None:
    install = content_roots.content_roots()
    if not install or install[0].kind != KIND_INSTALL:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    choices = ai.scan_ai_choices(install)
    kinds = {c.root.kind for c in choices if c.root is not None}
    assert "E3-p2.ai" in {c.stored_name for c in choices if c.root is not None and c.root.kind == KIND_INSTALL}
    if len(install) > 1:
        assert kinds & {KIND_MOD_SUBSCRIBED, content_roots.KIND_MOD_LOCAL, content_roots.KIND_PROFILE}
    resolved = ai.resolve(_choice(choices, "E3-p2.ai", install[0]), install)
    assert resolved.library[0][0] == b"E3-p2.per\x00"
