"""Qt-free newline/encoding rules for Messages mode -- no scenario file
needed, purely descape/messages_fields.py's two functions."""

from __future__ import annotations

import pytest

from descape.messages_fields import MAX_FIELD_BYTES, encode_for_write, normalize_for_display


def test_normalize_for_display_maps_lone_cr_to_lf() -> None:
    display, token = normalize_for_display("a\rb\rc")
    assert display == "a\nb\nc"
    assert token == "\r"


def test_normalize_for_display_leaves_lone_lf_alone() -> None:
    display, token = normalize_for_display("a\nb\nc")
    assert display == "a\nb\nc"
    assert token == "\n"


def test_normalize_for_display_reports_none_for_no_newline() -> None:
    display, token = normalize_for_display("just one line")
    assert display == "just one line"
    assert token is None


def test_newline_round_trip_cr() -> None:
    original = "para one\r\rpara two"
    display, token = normalize_for_display(original)
    assert encode_for_write(display, token) == original.encode("utf-8")


def test_newline_round_trip_lf() -> None:
    original = "para one\n\npara two"
    display, token = normalize_for_display(original)
    assert encode_for_write(display, token) == original.encode("utf-8")


def test_newline_round_trip_none() -> None:
    original = "one line, no newline"
    display, token = normalize_for_display(original)
    assert encode_for_write(display, token) == original.encode("utf-8")


def test_encode_for_write_defaults_unset_token_to_lf() -> None:
    assert encode_for_write("a\nb", None) == b"a\nb"


def test_encode_for_write_rejects_nul() -> None:
    with pytest.raises(ValueError):
        encode_for_write("a\x00b", None)


def test_encode_for_write_rejects_over_length() -> None:
    with pytest.raises(ValueError):
        encode_for_write("x" * (MAX_FIELD_BYTES + 1), None)


def test_encode_for_write_accepts_exactly_max_length() -> None:
    text = "x" * MAX_FIELD_BYTES
    assert len(encode_for_write(text, None)) == MAX_FIELD_BYTES


def test_encode_for_write_handles_utf8_multibyte() -> None:
    text = "Orléans"  # Orléans -- a corpus-confirmed round trip
    encoded = encode_for_write(text, None)
    assert encoded == text.encode("utf-8")
    display, token = normalize_for_display(text)
    assert encode_for_write(display, token) == text.encode("utf-8")
