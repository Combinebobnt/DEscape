"""Qt-free newline/encoding rules for Messages mode -- no scenario file
needed, purely descape/messages_fields.py's two functions."""

from __future__ import annotations

import pytest

from descape.messages_fields import (
    MAX_FIELD_BYTES,
    encode_for_write,
    normalize_for_display,
    substitute_newlines,
)


def test_normalize_for_display_maps_lone_cr_to_lf() -> None:
    display, token = normalize_for_display("a\rb\rc")
    assert display == "a\nb\nc"
    assert token == "\r"


def test_normalize_for_display_leaves_lone_lf_alone() -> None:
    display, token = normalize_for_display("a\nb\nc")
    assert display == "a\nb\nc"
    assert token == "\n"


def test_normalize_for_display_maps_crlf_to_one_lf() -> None:
    """Not a Messages-tab shape (zero CRLF across the 22-file corpus), but
    trigger `description` uses it, and the old two-branch form answered "\\r"
    here -- which displayed a phantom blank line and wrote back "a\\r\\rb"."""
    display, token = normalize_for_display("a\r\nb\r\nc")
    assert display == "a\nb\nc"
    assert token == "\r\n"


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


def test_newline_round_trip_crlf() -> None:
    original = "para one\r\n\r\npara two"
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


@pytest.mark.parametrize(
    "display,token",
    [
        ("a\nb", "\r"),
        ("a\nb", "\r\n"),
        ("a\nb", "\n"),
        ("a\nb", None),
        ("no newline", None),
    ],
)
def test_the_str_helper_and_the_bytes_wrapper_agree(display: str, token: str | None) -> None:
    """encode_for_write() is a thin wrapper over substitute_newlines() plus
    the Messages block's own NUL and length guards, so the trigger panel's
    str-returning path cannot drift from what the Messages tab writes."""
    assert encode_for_write(display, token) == substitute_newlines(display, token).encode("utf-8")


def test_the_str_helper_carries_no_nul_or_length_guard() -> None:
    """The guards belong to the Messages block's <H length prefix, not to a
    trigger field, so only the bytes wrapper enforces them."""
    assert substitute_newlines("a\x00b", None) == "a\x00b"
    assert len(substitute_newlines("x" * (MAX_FIELD_BYTES + 1), None)) == MAX_FIELD_BYTES + 1


def test_encode_for_write_handles_utf8_multibyte() -> None:
    text = "Orléans"  # Orléans -- a corpus-confirmed round trip
    encoded = encode_for_write(text, None)
    assert encoded == text.encode("utf-8")
    display, token = normalize_for_display(text)
    assert encode_for_write(display, token) == text.encode("utf-8")
