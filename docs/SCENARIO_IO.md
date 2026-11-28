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

Pre-DE scenarios (Age of Empires and Rise of Rome `.scn`/`.scx`, Age of Kings `.scn`, The
Conquerors `.scn`/`.scx`, UserPatch `.scx2`, HD Edition `.scx`) go through the same section walk, with structures this repo
ships under `descape/versions/legacy/`. Their file version string does not
identify the layout: HD Edition's `.aoe2scenario` and The Conquerors' `.scx`
both say `1.21`. So `descape/scenario_format.py` keys a FileHeader-version-2
file by its DataHeader version (the f32 4 bytes into the inflated body):
`v1.20` Age of Kings, `v1.22` The Conquerors, and `v1.21`, `v1.23`
(including UserPatch `.scx2`), `v1.24` and `v1.26` HD Edition. Each is a
neighbour with one section changed: `v1.21` is The Conquerors' without the
scouting message, `v1.23` is The Conquerors' plus 4 unnamed bytes at the end of
Diplomacy (only the first is ever nonzero, so they are not read as HD's team
options), `v1.24` is `v1.26` with 20 disabled buildings per player, not 30. A header-2
file with any other data version is refused with a named reason and never
falls through to the HD `1.21` structure. FileHeader version 3 and later
(HD's `.aoe2scenario`, every DE file) keep the path above.

`v1.15` is Age of Empires (1997) and Rise of Rome (label `aoe1-1.15`): the
two games share it, so the format cannot tell their files apart, and
`games.ROR` claims it. Against `v1.20` it has no string-table ids for player
names or messages, no AI type byte, a three-length AI record (AI, city plan,
personality, then their texts), 4 resources per player, an Options section
of 320 unnamed per-player flags before the modes and starting ages, no
camera before the map size, and 23-byte units with no
`initial_animation_frame` or `garrisoned_in_id`. The body ends with Units:
there is no Triggers section. A background bitmap's rows are padded to 4
bytes, so its length is `image_size`. The two absent unit fields read as 0
and -1 (`scenario_format.absent_unit_fields()`), which only the fast unit
reader can supply: the library's walk is refused by name. Such a file saves
no edit family and does not convert: DEscape offers no conversion for it.

Such a document (`LoadedScenario.source_format.legacy`) saves back to its own
format only the edit families its band's game has been shown to open
(`descape/legacy_write.py`, below), and is
shown with AoE2:DE data, so a few ids draw as their DE meaning. Two values
use an older encoding, which `descape/legacy_import.py` reads the way AoE2:DE's
own importer rewrites them, for The Conquerors' band only (the band with a
measured pair of a legacy file and the game's resave of it). A starting age
(`-1` or `0..4` stored) shows as DE's `0` or `2..6`
(`player_fields.shown_value()`), and a tree's variant `v` (`0..13`, consts
348-351, 411, 413) draws as one of DE's `3v..3v+2`, a draw-only override beside
the wall shapes. The stored values stay as they are: `current_value()` and
the Units panel show them raw.

Its triggers are read for trigger version 1.6, which every Age of Kings, The
Conquerors and HD header-2 file in the corpus stores except two (1.3, never
read: `parse_triggers()` returns None and the panel says so). A 1.6 effect
or condition stores its own field count, and the fields present are the
leading prefix of DE's (23 effect fields from `ai_script_goal` to
`instruction_panel_position`, 16 condition fields from `quantity` to
`ai_signal`); a trigger has no short description and 4 unnamed bytes after
`objective_description_order`, and an AI-files trailer follows the display
order. `descape/legacy_triggers.py` rewrites the section into the DE-shaped
layout the legacy structures declare (absent fields `-1`, DE-only trigger
fields their defaults, no variables) and refuses one that does not walk to
the exact end of the bytes; the library parses that. `trigger_tail` stays
verbatim, and the document keeps `trigger_write_supported` False with no
section offsets. Its vocabulary is keyed `legacy-<data version>`
(`LoadedScenario.vocabulary_version`, also the trigger clipboard's stamp, so
a legacy block never pastes into a DE document): each is the newest DE
vocabulary cut to the band's ids and to the fields a 1.6 entry stores, by
`tools/gen_legacy_vocabulary.py`, with the old game's name as `legacy_name`
where it differs (shown; rule tables key on DE's `name`). The execution
order reads as ID order, as AoE2:DE's importer writes it.

File > Convert to .aoe2scenario (`descape/legacy_convert.py`,
`batch_api.convert_legacy()`) makes a new DE document from a legacy one,
the way AoE2:DE's editor import does, and never writes the legacy file. It
starts from the DE blank template at the legacy map's size and fills it
through the edit models a save uses: terrain and elevation (33 and 34
remapped with the layer the importer writes), options, players, diplomacy,
disables, AIs, messages (re-decoded as cp1252) and every unit with its
`reference_id` (`UnitEditModel.import_units()`), then the player cameras.
Unit images follow the importer's measured rules (`unit_rule()`): trees,
re-derived wall shapes, fish and type-80 buildings take a new variant; every
other unit keeps its stored rotation and frame. Triggers, unknown objects,
background pictures and cinematics are left out, and the returned
`ConversionReport` lists everything that did not come across.
`tests/test_legacy_convert_oracle.py` checks the result against the
game's own resaves of two The Conquerors files. The same rules, the two
`legacy_import` encodings included, apply to every band; an Age of Kings or
HD file is converted by them unmeasured, and the report says so.

### Saving back to the old format

`scenario_write.build_patched()` saves a legacy document through the same
byte splices as a DE one, gated per edit family: terrain, elevation, Messages
text and diplomacy stances, each allowed per band in
`legacy_write.ALLOWED` once that family's pair from
`tools/legacy_oracle_pairs.py` opened in the band's own game. Every other
kind of pending edit (units, other options, allied victory, Messages
string-table ids, triggers) and any family the band lacks raises
`LegacyEditNotAllowed` before a byte is written, as does a changed tile whose
terrain id or elevation the band's corpus never holds
(`LEGAL_TERRAIN`, `BAND_MAX_ELEVATION`). A no-edit save is byte-identical.
`check_destination()` refuses an `.aoe2scenario` name, so a save keeps the
file's suffix; resize, Convert to Version and the fixture tools still call
`refuse_legacy()`. The viewer's locks ask the same table per family
(`game_data_lock_reason(family)`), and a Save there always writes
`.bak`/`.orig`.

Header-2 details the DE path never meets:

- `FileHeader.header_length` (the u32 at byte 4) is what a header-2 reader
  skips by, so after any header splice it is rewritten to `len(header) - 8`
  (`legacy_write.fix_header_length()`). DE stores 0 there.
- The header holds `individual_victories_used` between the Instructions
  copy and `player_count`, so the header walk includes it on a legacy
  document only (`_LEGACY_HEADER_WALK_ORDER`); some DE versions also have
  the field but keep the DE walk. A legacy file may leave that header copy
  empty while Messages holds text: the header is then left as it is.
- Text is one byte per character in the game's code page, mostly with a
  trailing NUL. `legacy_write.decode()`/`encode()` are cp1252 with its five
  undefined bytes mapped to latin-1, a bijection over all 256 bytes, so every
  stored string (a DBCS one included) round-trips byte for byte; an
  untouched field writes its stored bytes, an edited one keeps its NUL, and
  a character outside the code page is refused. The Messages walk reads each
  str16 length off its own prefix, since re-serializing drops that NUL.
- The Diplomacy section's expected length is summed from the file's own
  structure (12612 bytes for Age of Kings and The Conquerors, without the
  four HD team fields), not pinned to DE's 12616.
