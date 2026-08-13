"""v2's write path: patch the terrain struct array directly in the original
decompressed body and recompress, rather than going through
AoE2ScenarioParser's own commit()/write_to_file().

Why not the library's own write path -- confirmed this way, not assumed:
re-serializing Map/Units through AoE2ScenarioParser's managers and writing via
write_to_file() was tested directly against every example file in this repo.
It fails on 10 of 12: two versions crash entirely (a class-level property-
disabling bug in the library that persists across scenario instances loaded in
the same process -- this tool routinely loads several versions back-to-back,
so that's not an edge case here), and the other eight "succeed" but produce a
body that isn't byte-identical to the original even with zero edits (e.g. a
str16 field silently drops a trailing NUL). None of that is a bug worth
chasing upstream -- it's the general shape of "reserialize through an object
model that wasn't written to be byte-for-byte."

What works, confirmed byte-identical on all 12 example files for a zero-edit
round trip: never touch the parsed object graph's serialization at all. Take
the *original* decompressed body (kept verbatim since load, see
scenario_io.LoadedScenario), overwrite only the terrain struct array's bytes
from the current (possibly edited) in-memory tile values, recompress with the
same zlib call AoE2ScenarioParser itself uses, and concatenate after the
verbatim original header. Everything else in the file -- every other Map
retriever, Units, the never-parsed trigger tail -- passes through as pure
untouched bytes.

See tests/test_write_path.py for the evidence.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from descape.scenario_io import (
    TERRAIN_STRUCT_SIZE,
    FORBIDDEN_WRITE_MARKER,
    TEMPLATE_DIR,
    LoadedScenario,
)

_LAYER_STRUCT = struct.Struct("<h")


class WriteBlockedError(Exception):
    """Raised instead of writing, e.g. for the compatdata guard or an
    unsupported (terrain_write_supported=False) file."""


def _compress_bytes(data: bytes) -> bytes:
    # Same call AoE2ScenarioParser.scenarios.aoe2_scenario._compress_bytes()
    # uses -- kept independent (not imported) so this module has zero
    # dependency on the library's write path, which is the entire point.
    deflate_obj = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return deflate_obj.compress(data) + deflate_obj.flush()


def _patch_terrain_block(scenario: LoadedScenario) -> bytes:
    """Returns a new decompressed body: scenario.decompressed_body with the
    terrain struct array overwritten from the *current* MapManager.terrain
    tile values. Only terrain_id, elevation, and layer are touched -- the
    per-tile 'unused' 3 bytes are carried through untouched, exactly as
    loaded."""
    body = bytearray(scenario.decompressed_body)
    offset = scenario.terrain_block_offset
    for i, tile in enumerate(scenario.map_manager.terrain):
        o = offset + TERRAIN_STRUCT_SIZE * i
        body[o] = tile.terrain_id
        body[o + 1] = tile.elevation
        _LAYER_STRUCT.pack_into(body, o + 5, tile.layer)
    return bytes(body)


def write_scenario(scenario: LoadedScenario, out_path: str | Path) -> None:
    """Writes `scenario`'s current Map terrain state (edited or not) to
    out_path as a full .aoe2scenario file. Units and the trigger tail are
    unmodified regardless of what's in memory -- v2 only writes terrain.

    Reads from MapManager.terrain directly, so this is automatically correct
    at any edit_history cursor position: saving after an undo writes the
    undone state, with no separate bookkeeping needed.

    Raises:
        WriteBlockedError: if out_path's path contains FORBIDDEN_WRITE_MARKER
            (never write into a Workshop/Proton compatdata folder -- the
            user's only copy of these files), if out_path's parent is
            TEMPLATE_DIR (never overwrite a shipped File > New template --
            covers the whole directory, so future sibling templates are
            protected without another edit here), or if this scenario failed
            its load-time terrain-block verification (terrain_write_supported
            is False) and so has no reliable offset to patch.
    """
    out_path = Path(out_path)
    if FORBIDDEN_WRITE_MARKER in str(out_path):
        raise WriteBlockedError(
            f"Refusing to write to a path containing {FORBIDDEN_WRITE_MARKER!r}: {out_path}"
        )
    if out_path.resolve().parent == TEMPLATE_DIR.resolve():
        raise WriteBlockedError(
            f"Refusing to write into the shipped template directory ({TEMPLATE_DIR}): {out_path}"
        )
    if not scenario.terrain_write_supported:
        raise WriteBlockedError(
            "This file's terrain block failed load-time verification -- "
            "writing would silently patch the wrong bytes. Edit mode should "
            "already be disabled for this file; write_scenario() should "
            "never be reached here."
        )

    patched_body = _patch_terrain_block(scenario)
    compressed = _compress_bytes(patched_body)
    out_path.write_bytes(scenario.header_bytes + compressed)
