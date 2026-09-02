"""Qt-free newline/encoding rules for Messages mode's six prose fields
(Instructions/Hints/Victory/Loss/History/Scouts) -- the field spec table and
the two pure functions descape/messages_model.py's read/write path builds
on, mirroring descape/diplomacy_fields.py's split from its panel.

Newlines are the one real gotcha here, confirmed against the corpus: a
stored field uses a lone `\r` *or* a lone `\n` as its line separator, never
`\r\n`, and the choice varies
per field within one file (2_Joan_coop_3 has `\r\r` hints and `\n\n`
scouts). QPlainTextEdit.setPlainText()/toPlainText() round-trips a lone `\r`
1:1 to `\n` and back, so normalize_for_display()/encode_for_write() just
track which token a field originally used and substitute on the way back
out -- never mixing tokens across fields, and never touching a field with
no newline at all.
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
    which newline token the field originally used ("\\r", "\\n", or None for
    a field with no newline at all -- distinct from "\\n" so a clean field's
    round trip through encode_for_write() reproduces the original bytes
    exactly rather than silently adopting "\\n" as this field's convention).

    A field may not mix `\\r` and `\\n` -- not seen in the corpus, but if it
    somehow does, `\\r` is treated as this field's token, an over-cautious
    default rather than a raise on a never-observed shape.
    """
    if "\r" in text:
        return text.replace("\r", "\n"), "\r"
    if "\n" in text:
        return text, "\n"
    return text, None


def encode_for_write(display: str, newline_token: str | None) -> bytes:
    """Inverse of normalize_for_display(): substitutes `display`'s `\\n`
    back to `newline_token` (defaulting to `\\n` when the field had none),
    then UTF-8 encodes.

    Raises ValueError on a NUL character (the game's strings are C-style;
    the library's del_str_trail would silently eat one on the next read) or
    on exceeding MAX_FIELD_BYTES once encoded -- refused, not clamped.
    """
    if "\x00" in display:
        raise ValueError("message text cannot contain a NUL character")
    token = newline_token if newline_token is not None else "\n"
    raw = display.replace("\n", token) if token != "\n" else display
    encoded = raw.encode("utf-8")
    if len(encoded) > MAX_FIELD_BYTES:
        raise ValueError(
            f"message text is {len(encoded)} UTF-8 bytes, over the {MAX_FIELD_BYTES}-byte field limit"
        )
    return encoded
