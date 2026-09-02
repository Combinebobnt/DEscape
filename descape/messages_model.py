"""Per-document edit state for Messages mode (the scenario's Instructions/
Hints/Victory/Loss/History/Scouts prose), and the length-changing splice
this codebase's write path has not needed before.

MessagesEditModel follows descape/options_model.py's OptionsEditModel API
surface so the viewer wiring is copy-shaped, but it differs from every other
edit model here in one load-bearing way: every other model patches a
fixed-width byte range, so its write is an in-place overwrite that shifts no
offset. A message field is a length-prefixed string, so an edit can change
the section's total length -- serialize() always emits the *whole* 12-field
section, and descape/scenario_write.py splices that block in wholesale
rather than patching individual bytes.

FileHeader.scenario_instructions is a second, independent copy of the
Instructions field (in the *uncompressed* header, outside the compressed
body Messages lives in). header_patch() keeps it in sync when instructions
changes, but only when this model can trust doing so -- see
_header_sync_verified()'s docstring for the on-disk NUL-encoding quirk that
makes a naive byte-identity check between the two copies wrong.
"""

from __future__ import annotations

import struct

from descape.messages_fields import (
    MAX_FIELD_BYTES,
    MESSAGE_FIELDS,
    encode_for_write,
    normalize_for_display,
)
from descape.scenario_io import LoadedScenario

_ID_STRUCT = struct.Struct("<I")
_TEXT_LEN_STRUCT = struct.Struct("<H")
_HEADER_LEN_STRUCT = struct.Struct("<I")

_INSTRUCTIONS_FIELD_ID = "instructions"


class MessageEditsUnavailableError(Exception):
    """Raised by MessagesEditModel() for a file whose Messages section
    failed its load-time forward-walk verification -- see
    scenario_io.LoadedScenario.messages_write_supported."""


def _id_field_id(field_id: str) -> str:
    return f"{field_id}_id"


def _strip_one_trailing_nul(data: bytes) -> bytes:
    """FileHeader's str32 fields (unlike Messages' str16 fields, which the
    corpus never shows carrying one -- see the plan's "No NUL trails
    anywhere" finding) sometimes store a single trailing NUL on disk even
    for an "empty" value; AoE2ScenarioParser's own del_str_trail() strips
    exactly one such byte when parsing to a Python str. Mirrored here
    (rather than imported) so this module has no runtime dependency on the
    library's parse-side helpers, only its already-parsed retriever values."""
    return data[:-1] if data.endswith(b"\x00") else data


class MessagesEditModel:
    """Six text fields plus their six string-table ids, all lazily dirty
    until set_value() actually changes one -- has_edits/serialize() drive
    descape/scenario_write.py's splice, exactly the "nothing patched for a
    browse-only document" contract every other edit model here keeps.
    """

    def __init__(self, loaded: LoadedScenario):
        if not loaded.messages_write_supported:
            raise MessageEditsUnavailableError(
                "This file's Messages section failed its load-time verification -- "
                "splicing would land at an offset that cannot be trusted."
            )
        self.loaded = loaded

        retriever_map = loaded._scenario.sections["Messages"].retriever_map
        self._original_text: dict[str, str] = {}
        self._original_newline: dict[str, str | None] = {}
        self._original_id: dict[str, int] = {}
        self._pending_text: dict[str, str] = {}
        self._pending_id: dict[str, int] = {}

        for spec in MESSAGE_FIELDS:
            text = retriever_map[spec.retriever].data
            display, newline_token = normalize_for_display(text)
            self._original_text[spec.field_id] = display
            self._original_newline[spec.field_id] = newline_token
            self._original_id[spec.field_id] = retriever_map[spec.id_retriever].data

        self._header_sync_verified = self._check_header_sync(retriever_map)

    # -- header sync ----------------------------------------------------

    def _check_header_sync(self, messages_retriever_map: dict) -> bool:
        """True iff FileHeader.scenario_instructions' payload span can be
        trusted *and* its parsed value agrees with the Messages copy --
        header_patch() refuses to touch the header at all unless this is
        True, matching the plan's "two disagreeing copies is a file we
        don't understand, not a merge to guess at" guard."""
        start, end = self.loaded.header_instructions_span
        if start < 0:
            return False
        header_retriever = self.loaded._scenario.sections["FileHeader"].retriever_map[
            "scenario_instructions"
        ]
        header_value = header_retriever.data
        if not isinstance(header_value, str):
            return False
        header_bytes = _strip_one_trailing_nul(self.loaded.header_bytes[start:end])
        if header_bytes != header_value.encode("utf-8"):
            return False
        return header_value == messages_retriever_map[
            MESSAGE_FIELDS[0].retriever
        ].data  # "instructions" is always MESSAGE_FIELDS[0]

    # -- state ------------------------------------------------------------

    @property
    def has_edits(self) -> bool:
        return bool(self._pending_text) or bool(self._pending_id)

    def original_value(self, field_id: str):
        if field_id in self._original_text:
            return self._original_text[field_id]
        return self._original_id[field_id[: -len("_id")]]

    def current_value(self, field_id: str):
        if field_id in self._original_text:
            return self._pending_text.get(field_id, self._original_text[field_id])
        base = field_id[: -len("_id")]
        return self._pending_id.get(base, self._original_id[base])

    def pending_values(self) -> dict[str, str | int]:
        """Only the fields that differ from the file, for the panel's
        `values` override -- empty for a document nothing has been changed
        in."""
        values: dict[str, str | int] = dict(self._pending_text)
        values.update({_id_field_id(field_id): value for field_id, value in self._pending_id.items()})
        return values

    def set_value(self, field_id: str, value: str | int) -> None:
        """Record `field_id` (a text field id or an "<id>_id" id field) as
        set to `value`. Must be wrapped in an undo record by the caller.

        For a text field, dirtiness is decided by comparing *serialized*
        bytes against the original, not display strings -- see this
        module's docstring on `has_edits`. A field the user focused and
        left unchanged must report clean even if normalize_for_display()
        was lossy in some way this module didn't anticipate.
        """
        if field_id in self._original_text:
            newline_token = self._original_newline[field_id]
            encoded = encode_for_write(value, newline_token)  # raises ValueError
            original_encoded = encode_for_write(self._original_text[field_id], newline_token)
            if encoded == original_encoded:
                self._pending_text.pop(field_id, None)
            else:
                self._pending_text[field_id] = value
            return

        if not field_id.endswith("_id"):
            raise KeyError(f"{field_id!r} is not a Messages mode field")
        base = field_id[: -len("_id")]
        if base not in self._original_id:
            raise KeyError(f"{field_id!r} is not a Messages mode field")
        try:
            _ID_STRUCT.pack(value)
        except struct.error as e:
            raise ValueError(f"{value!r} does not fit a string-table id: {e}") from e
        if value == self._original_id[base]:
            self._pending_id.pop(base, None)
        else:
            self._pending_id[base] = value

    # -- serialization ------------------------------------------------------

    def _encoded(self, field_id: str) -> bytes:
        return encode_for_write(self.current_value(field_id), self._original_newline[field_id])

    def section_span(self) -> tuple[int, int]:
        return self.loaded.messages_section_start, self.loaded.messages_section_end

    def serialize(self) -> bytes:
        """The whole 12-retriever Messages section: six <I string-table ids
        in field order, then six <H-length-prefixed UTF-8 payloads in the
        same field order. Never delegates to AoE2ScenarioParser's own
        serializer -- see this module's docstring."""
        out = bytearray()
        for spec in MESSAGE_FIELDS:
            out += _ID_STRUCT.pack(self.current_value(_id_field_id(spec.field_id)))
        for spec in MESSAGE_FIELDS:
            encoded = self._encoded(spec.field_id)
            out += _TEXT_LEN_STRUCT.pack(len(encoded))
            out += encoded
        return bytes(out)

    def header_patch(self) -> tuple[int, int, bytes] | None:
        """(start, end, payload) to splice into header_bytes -- the *whole*
        FileHeader.scenario_instructions field (its 4-byte length prefix
        plus payload), replaced with the Instructions field's current
        value. None when instructions is unchanged, or when the header
        span/sync check failed at construction (the header is then left
        untouched; Instructions editing still works for the Messages copy).
        """
        if _INSTRUCTIONS_FIELD_ID not in self._pending_text:
            return None
        if not self._header_sync_verified:
            return None
        payload_start, payload_end = self.loaded.header_instructions_span
        encoded = self._encoded(_INSTRUCTIONS_FIELD_ID)
        if len(encoded) > MAX_FIELD_BYTES:
            raise ValueError("instructions text exceeds the field size limit")
        replacement = _HEADER_LEN_STRUCT.pack(len(encoded)) + encoded
        return payload_start - _HEADER_LEN_STRUCT.size, payload_end, replacement
