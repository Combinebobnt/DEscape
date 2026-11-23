# Why `scenario_io.py` doesn't just call `AoE2DEScenario.from_file()`

AoE2ScenarioParser's own loader refuses to open scenario files where the internal
scenario version is `1.54` and the trigger-data sub-version is `3.9` — an older
trigger format the library's structs can't parse — raising
`UnsupportedVersionError` before returning anything at all, *even though* Map and
Units parse cleanly before the loader ever reaches Triggers. This isn't
overcautious: bypassing the check experimentally confirmed Triggers then crashes
for real, deeper in, on a genuine structural field mismatch. Four of the six
scenario files this tool was built against hit exactly this case.

`scenario_io.load_map_and_units()` replicates `from_file()`'s own section-walking
logic (using the library's real per-version structure definitions, not guesswork)
but stops right after the `Units` section instead of continuing into `Triggers`.
Everything from that byte offset onward is kept as one untouched `bytes` blob
(`LoadedScenario.trigger_tail`) — never parsed, and preserved byte-exact for a
future write path to splice back verbatim.

## Older formats

Pre-DE scenarios (Age of Kings `.scn`, The Conquerors `.scn`/`.scx`, HD
Edition `.scx`) go through the same section walk, with structures this repo
ships under `descape/versions/legacy/`. Their file version string does not
identify the layout: HD Edition's `.aoe2scenario` and The Conquerors' `.scx`
both say `1.21`. So `descape/scenario_format.py` keys a FileHeader-version-2
file by its DataHeader version (the f32 4 bytes into the inflated body):
`v1.20` Age of Kings, `v1.22` The Conquerors, `v1.26` HD Edition. A header-2
file with any other data version is refused with a named reason and never
falls through to the HD `1.21` structure. FileHeader version 3 and later
(HD's `.aoe2scenario`, every DE file) keep the path above.

Such a document is read-only (`LoadedScenario.source_format.legacy`,
`scenario_write.refuse_legacy()`), its triggers are never read, and it is
shown with AoE2:DE data, so a few ids draw as their DE meaning.
