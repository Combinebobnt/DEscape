"""A Return of Rome scenario built from the blank template (GH #67).

The user's RoR samples are local-only, so tests build one: the template's
FileHeader gets the variant field (unknown_value_2) set to 2 and its DLC list
(unknown_numbers) set to [11], as every 1.59 RoR sample stores it. The header
is uncompressed and its stored header_length is 0 in both, so splicing the
list from 17 entries to 1 (68 bytes to 4) leaves the body byte-identical.
"""

from __future__ import annotations

import struct
from pathlib import Path

from descape import game_dataset, scenario_io

ROR_DLCS = (11,)


def ror_copy(directory: Path, source: Path = scenario_io.BLANK_TEMPLATE_PATH, name: str = "ror_blank.aoe2scenario") -> Path:
    """`source` rewritten as a Return of Rome file in `directory`; the new path."""
    raw = source.read_bytes()
    loaded = scenario_io.load_map_and_units(source)
    spans = scenario_io._header_field_spans(loaded.header_bytes, loaded._scenario.sections["FileHeader"].retriever_map)
    variant_start, variant_end = spans["unknown_value_2"]
    count_start, count_end = spans["amount_of_unknown_numbers"]
    list_start, list_end = spans["unknown_numbers"]
    assert variant_end - variant_start == 4 and count_end - count_start == 4
    assert variant_end <= count_start and count_end == list_start
    out = bytearray(raw[:list_start])
    struct.pack_into("<I", out, variant_start, game_dataset.ROR_VARIANT)
    struct.pack_into("<I", out, count_start, len(ROR_DLCS))
    out += b"".join(struct.pack("<I", dlc) for dlc in ROR_DLCS)
    out += raw[list_end:]
    path = directory / name
    path.write_bytes(bytes(out))
    return path
