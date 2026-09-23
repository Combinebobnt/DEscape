"""Qt-free newline/encoding rules for Messages mode's six prose fields
(Instructions/Hints/Victory/Loss/History/Scouts) -- the field spec table and
the two pure functions descape/messages_model.py's read/write path builds
on, mirroring descape/diplomacy_fields.py's split from its panel.

Newlines are the one real gotcha here, confirmed against the corpus: a
stored Messages field uses a lone `\r` *or* a lone `\n` as its line
separator, and the choice varies
per field within one file (2_Joan_coop_3 has `\r\r` hints and `\n\n`
scouts). QPlainTextEdit.setPlainText()/toPlainText() round-trips a lone `\r`
1:1 to `\n` and back, so normalize_for_display()/encode_for_write() just
track which token a field originally used and substitute on the way back
out -- never mixing tokens across fields, and never touching a field with
no newline at all.

`\r\n` is a third token, absent from all 132 Messages values in the 22-file
corpus but present in trigger `description`, whose prose editor
(descape/text_edits.py's ProseTextEdit) shares these two functions. That
field is why substitute_newlines() exists as a str -> str helper: a trigger
field is stored as a str on the entry, and the <H length prefix and NUL
guard encode_for_write() adds belong to the Messages block alone.
"""

from __future__ import annotations

from dataclasses import dataclass

# 0xFFFFFFFE means "unset" for a string-table id -- the same sentinel across
# all six id retrievers.
STRING_ID_UNSET = 0xFFFFFFFE

# The <H length prefix on each str16 payload is a hard ceiling: any text
# whose UTF-8 encoding needs more bytes than this cannot be stored at all.
MAX_FIELD_BYTES = 0xFFFF


@dataclass(frozen=True)
class MessageFieldSpec:
    field_id: str  # "instructions", "hints", "victory", "loss", "history", "scouts"
    label: str  # panel-facing label
    retriever: str  # the Messages section's str16 retriever name
    id_retriever: str  # the Messages section's u32 string-table id retriever name


# File order -- also the order serialize() emits in, six ids then six texts.
MESSAGE_FIELDS: tuple[MessageFieldSpec, ...] = (
    MessageFieldSpec("instructions", "Instructions", "ascii_instructions", "instructions"),
    MessageFieldSpec("hints", "Hints", "ascii_hints", "hints"),
    MessageFieldSpec("victory", "Victory", "ascii_victory", "victory"),
    MessageFieldSpec("loss", "Loss", "ascii_loss", "loss"),
    MessageFieldSpec("history", "History", "ascii_history", "history"),
    MessageFieldSpec("scouts", "Scouts", "ascii_scouts", "scouts"),
)


def normalize_for_display(text: str) -> tuple[str, str | None]:
    """Maps a field's stored text to what QPlainTextEdit should show, plus
    which newline token the field originally used ("\\r\\n", "\\r", "\\n", or
    None for a field with no newline at all -- distinct from "\\n" so a clean
    field's round trip through encode_for_write() reproduces the original
    bytes exactly rather than silently adopting "\\n" as this field's
    convention).

    `\\r\\n` is tested before the bare `\\r`, or a CRLF field would latch
    `\\r`, display a phantom blank line per break and write back doubled.

    A field may not mix its separators -- not seen in the corpus for any of
    these fields, but if it somehow does, the first token matched is treated
    as this field's, an over-cautious default rather than a raise on a
    never-observed shape.
    """
    if "\r\n" in text:
        return text.replace("\r\n", "\n"), "\r\n"
    if "\r" in text:
        return text.replace("\r", "\n"), "\r"
    if "\n" in text:
        return text, "\n"
    return text, None


def substitute_newlines(display: str, newline_token: str | None) -> str:
    """Inverse of normalize_for_display(): substitutes `display`'s `\\n` back
    to `newline_token`, defaulting to `\\n` when the field had none.

    Carries no NUL or length guard: those are encode_for_write()'s, and they
    belong to the Messages block's <H length prefix rather than to every
    caller (descape/trigger_panel.py's prose fields store a str on a trigger
    or effect, with no such prefix in the way).
    """
    token = newline_token if newline_token is not None else "\n"
    return display.replace("\n", token) if token != "\n" else display


def encode_for_write(display: str, newline_token: str | None) -> bytes:
    """substitute_newlines(), then UTF-8 encode, for the Messages block.

    Raises ValueError on a NUL character (the game's strings are C-style;
    the library's del_str_trail would silently eat one on the next read) or
    on exceeding MAX_FIELD_BYTES once encoded -- refused, not clamped.
    """
    if "\x00" in display:
        raise ValueError("message text cannot contain a NUL character")
    encoded = substitute_newlines(display, newline_token).encode("utf-8")
    if len(encoded) > MAX_FIELD_BYTES:
        raise ValueError(
            f"message text is {len(encoded)} UTF-8 bytes, over the {MAX_FIELD_BYTES}-byte field limit"
        )
    return encoded
