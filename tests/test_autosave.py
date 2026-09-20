"""Verifies descape/autosave.py -- slot naming, rotation, the index, the
sidecar fallback, and stale pruning. Qt-free, so it runs in the default
tier; the viewer's timer/tick behaviour is tests/test_autosave_viewer.py's
job.

Every case rides conftest's autouse _isolated_settings fixture, which
redirects asset_source.CONFIG_PATH into tmp_path -- autosave_dir() resolves
under that at call time, so nothing here can touch the real
~/.config/DEscape/autosave/.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from descape import autosave
from descape.scenario_io import BLANK_TEMPLATE_PATH, TEMPLATE_DIR, load_map_and_units
from descape.scenario_write import write_scenario

DOC_ID = "0123456789abcdef"


def _slot(key: str, name: str = "map.aoe2scenario", source: Path | None = None,
          location: str = "central", when: float | None = None) -> Path:
    path = autosave.slot_path(key, name, source, location, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"slot")
    autosave.record(key, path, str(source or name), untitled=source is None, timestamp=when)
    return path


def test_a_slot_is_not_matched_by_the_corpus_scenario_glob(tmp_path):
    """The appended-suffix rationale, as a test: substituting the extension
    instead (foo.autosave.aoe2scenario) would put every slot into the Open
    dialog's filter and into tests/conftest.py's corpus glob."""
    source = tmp_path / "June_Event.aoe2scenario"
    source.write_bytes(b"x")
    slot = autosave.slot_path(autosave.doc_key(source, DOC_ID), source.name, source, "sidecar")

    assert slot.name.endswith(autosave.AUTOSAVE_SUFFIX)
    assert slot.parent == tmp_path
    slot.write_bytes(b"y")
    assert sorted(tmp_path.glob("*.aoe2scenario")) == [source]


def test_rotation_keeps_exactly_the_newest_n_slots():
    key = autosave.doc_key(None, DOC_ID)
    now = time.time()
    written = [_slot(key, when=now + i) for i in range(5)]

    deleted = autosave.rotate(key, 3)

    assert sorted(deleted) == sorted(written[:2])
    assert [e.path for e in autosave.entries()] == written[:1:-1]
    assert all(not p.exists() for p in written[:2])


def test_two_documents_rotate_independently():
    """One central directory holds every document's slots, so a rotate()
    that walked the directory rather than the key would prune a document
    the user never even had open."""
    key_a, key_b = autosave.doc_key(None, "aaa"), autosave.doc_key(None, "bbb")
    now = time.time()
    a_slots = [_slot(key_a, "a.aoe2scenario", when=now + i) for i in range(3)]
    b_slots = [_slot(key_b, "b.aoe2scenario", when=now + i) for i in range(3)]

    autosave.rotate(key_a, 1)

    assert [p.exists() for p in a_slots] == [False, False, True]
    assert all(p.exists() for p in b_slots)


@pytest.mark.parametrize("name", ["compatdata", "template", "untitled"])
def test_sidecar_falls_back_to_central_where_it_cannot_be_served(tmp_path, name):
    """The AGENTS.md hard rule as a test for the first case: because the
    slot never lands under compatdata/, write_scenario()'s own guard stays
    the thing enforcing it and nothing here weakens it."""
    sources = {
        "compatdata": tmp_path / "compatdata" / "1234" / "map.aoe2scenario",
        "template": TEMPLATE_DIR / "blank_120x120.aoe2scenario",
        "untitled": None,
    }
    source = sources[name]
    key = autosave.doc_key(source, DOC_ID)

    slot = autosave.slot_path(key, "map.aoe2scenario", source, "sidecar")

    assert slot.parent == autosave.autosave_dir()
    assert "compatdata" not in str(slot)
    assert slot.resolve().parent != TEMPLATE_DIR.resolve()


def test_reopening_the_same_named_path_reuses_its_key(tmp_path):
    """Keyed on the path, not the per-document doc_id: otherwise every
    session would start a fresh set of slots for the same file and they
    would grow without bound."""
    source = tmp_path / "map.aoe2scenario"
    source.write_bytes(b"x")

    assert autosave.doc_key(source, "session-one") == autosave.doc_key(source, "session-two")
    assert autosave.doc_key(None, "session-one") != autosave.doc_key(None, "session-two")


def test_entries_skips_a_row_whose_file_vanished():
    key = autosave.doc_key(None, DOC_ID)
    kept = _slot(key, when=time.time())
    gone = _slot(key, when=time.time() + 1)
    gone.unlink()

    assert [e.path for e in autosave.entries()] == [kept]


def test_discard_removes_both_the_files_and_the_index_rows():
    key = autosave.doc_key(None, DOC_ID)
    slots = [_slot(key, when=time.time() + i) for i in range(2)]

    autosave.discard(key)

    assert autosave.entries() == []
    assert all(not p.exists() for p in slots)


def test_a_written_slot_is_byte_identical_to_a_normal_save(tmp_path):
    """What earns the "a legitimate scenario, not a second-class artifact"
    claim. Terrain equality alone would not: the header goes through
    _patch_header_trigger_count() on both paths, so a slot that differed
    only there would still load with matching terrain."""
    loaded = load_map_and_units(BLANK_TEMPLATE_PATH)
    key = autosave.doc_key(None, DOC_ID)
    slot = autosave.slot_path(key, "blank.aoe2scenario", None, "central")
    slot.parent.mkdir(parents=True, exist_ok=True)
    normal = tmp_path / "normal.aoe2scenario"

    write_scenario(loaded, slot, backup=False)
    write_scenario(loaded, normal, backup=False)

    assert slot.read_bytes() == normal.read_bytes()
    assert load_map_and_units(slot).map_manager.map_width == loaded.map_manager.map_width


def test_prune_stale_drops_an_old_key_and_keeps_a_fresh_one():
    now = time.time()
    old_key, fresh_key = autosave.doc_key(None, "old"), autosave.doc_key(None, "fresh")
    old = _slot(old_key, "old.aoe2scenario", when=now - 60 * 60 * 24 * 40)
    fresh = _slot(fresh_key, "fresh.aoe2scenario", when=now)

    assert autosave.prune_stale() == [old_key]
    assert not old.exists()
    assert fresh.exists()
