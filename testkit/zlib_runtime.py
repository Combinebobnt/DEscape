"""Whether this interpreter's zlib is stock zlib, and a scenario file's
comparable parts.

The committed fixtures and the shipped blank template are pinned byte for
byte against their generators, but deflate output is only reproducible on
the zlib that wrote them. zlib-ng (Fedora's zlib-ng-compat, python.org's
3.14 Windows build) writes different, equally valid deflate for the same
body, so each byte check is paired with a header-plus-decompressed-body
check that holds on any zlib, and the byte check skips off stock zlib.
"""

from __future__ import annotations

import sys
import zlib


def _is_stock(module) -> bool:
    # A zlib-ng-compat build says so in the version string; zlib-ng's own module has the attribute.
    return "zlib-ng" not in module.ZLIB_RUNTIME_VERSION and not hasattr(module, "ZLIBNG_VERSION")


STOCK_ZLIB = _is_stock(zlib)

# zlib-ng's own module reports its compat API (1.2.12) as ZLIB_RUNTIME_VERSION.
_RUNTIME = f"zlib-ng {zlib.ZLIBNG_VERSION}" if hasattr(zlib, "ZLIBNG_VERSION") else zlib.ZLIB_RUNTIME_VERSION

NOT_STOCK_REASON = f"zlib runtime {_RUNTIME} is not stock zlib: deflate output differs, decompressed equality is checked separately"

BYTES_DIFFER = (
    "compressed bytes differ while header and body match (the sibling test without _byte_identical): "
    "the file was likely regenerated off stock zlib, so regenerate it on stock zlib"
)


def scenario_parts(raw: bytes) -> tuple[bytes, bytes]:
    """(header bytes, decompressed body), split by the real loader. Raises on
    bytes after the deflate stream, which the loader's inflate would ignore."""
    from descape.scenario_io import load_map_and_units_from_bytes

    loaded = load_map_and_units_from_bytes(raw, "<scenario_parts>")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    inflater.decompress(loaded.original_compressed_body)
    if not inflater.eof or inflater.unused_data:
        raise ValueError(f"{len(inflater.unused_data)} bytes after the deflate stream (eof={inflater.eof})")
    return loaded.header_bytes, loaded.decompressed_body


def warn_if_not_stock(written: str) -> None:
    """One stderr line for a generator run off stock zlib; the write still happens."""
    if not STOCK_ZLIB:
        print(
            f"warning: zlib runtime {_RUNTIME} is not stock zlib, so {written} "
            "will not match CI's bytes (the decompressed body still does)",
            file=sys.stderr,
        )
