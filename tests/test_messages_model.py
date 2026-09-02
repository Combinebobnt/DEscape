"""MessagesEditModel -- byte-level read/write state for Messages mode, no Qt.

Default tier runs against the blank template (empty text, every string-table
id unset, matching every other shipped fixture -- no shipped fixture
exercises the set-string-id or lone-`\\r`-newline paths). Those branches
are exercised at corpus tier instead, against the real F7_3_York file, the
one corpus scenario with non-empty instructions and a set string id on two
other fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.messages_fields import STRING_ID_UNSET
from descape.messages_model import MessageEditsUnavailableError, MessagesEditModel
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import load_map_and_units

F7_YORK_PATH = Path(__file__).resolve().parent.parent / "examples" / "F7_3_York (865).aoe2scenario"


def _loaded(path=FIXTURE_PATH):
    return load_map_and_units(path)


def test_serialize_of_unedited_model_is_byte_identical() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    start, end = model.section_span()
    assert model.serialize() == loaded.decompressed_body[start:end]


@pytest.mark.corpus
def test_serialize_of_unedited_model_is_byte_identical_corpus(scenario_path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.messages_write_supported:
        pytest.skip(f"{scenario_path.name}: messages_write_supported is False")
    model = MessagesEditModel(loaded)
    start, end = model.section_span()
    assert model.serialize() == loaded.decompressed_body[start:end]


def test_has_edits_false_after_set_to_same_value() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    model.set_value("hints", model.current_value("hints"))
    assert not model.has_edits


def test_has_edits_true_after_a_real_edit() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    model.set_value("hints", "some hint text")
    assert model.has_edits
    assert model.current_value("hints") == "some hint text"


def test_setting_back_to_original_clears_has_edits() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    original = model.current_value("hints")
    model.set_value("hints", "temporary")
    model.set_value("hints", original)
    assert not model.has_edits


def test_id_field_set_and_clear() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    assert model.current_value("hints_id") == STRING_ID_UNSET
    model.set_value("hints_id", 12345)
    assert model.has_edits
    assert model.current_value("hints_id") == 12345
    model.set_value("hints_id", STRING_ID_UNSET)
    assert not model.has_edits


def test_header_patch_none_when_instructions_unchanged() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    model.set_value("hints", "unrelated edit")
    assert model.header_patch() is None


def test_header_patch_none_when_header_sync_unverified() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    model._header_sync_verified = False
    model.set_value("instructions", "new text")
    assert model.header_patch() is None


def test_header_patch_present_when_instructions_changed_and_synced() -> None:
    loaded = _loaded()
    model = MessagesEditModel(loaded)
    if not model._header_sync_verified:
        pytest.skip("this fixture's header sync did not verify")
    model.set_value("instructions", "new text")
    patch = model.header_patch()
    assert patch is not None
    start, end, payload = patch
    assert payload == b"\x08\x00\x00\x00new text"


def test_unavailable_model_raises_when_write_not_supported() -> None:
    loaded = _loaded()
    loaded.messages_write_supported = False
    with pytest.raises(MessageEditsUnavailableError):
        MessagesEditModel(loaded)


# -- corpus-only branches: set string id, lone-\r newline --------------------


def _requires_york():
    if not F7_YORK_PATH.exists():
        pytest.skip("examples/ corpus not present")
    return load_map_and_units(F7_YORK_PATH)


@pytest.mark.corpus
def test_set_string_id_field_is_readable_and_clearable() -> None:
    loaded = _requires_york()
    model = MessagesEditModel(loaded)
    assert model.current_value("hints_id") != STRING_ID_UNSET
    original_id = model.current_value("hints_id")
    model.set_value("hints_id", STRING_ID_UNSET)
    assert model.has_edits
    model.set_value("hints_id", original_id)
    assert not model.has_edits


@pytest.mark.corpus
def test_lone_cr_newline_field_round_trips_through_display() -> None:
    loaded = _requires_york()
    model = MessagesEditModel(loaded)
    # instructions is '\r\r' on this file -- see the plan's corpus findings.
    original = model.current_value("instructions")
    assert original == "\n\n"  # normalize_for_display() maps \r -> \n
    model.set_value("instructions", original)
    assert not model.has_edits, "re-setting the normalized display text must not mark this dirty"


@pytest.mark.corpus
def test_header_instructions_mirror_verified_on_york() -> None:
    loaded = _requires_york()
    model = MessagesEditModel(loaded)
    assert model._header_sync_verified
