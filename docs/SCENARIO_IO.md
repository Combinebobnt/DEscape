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
