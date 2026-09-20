# XS scripting in AoE2:DE scenarios

How XS script travels inside an `.aoe2scenario` file, which parts of it
DEscape reads and edits, and the constraints a script has to respect to run
at all. Written for scenario authors who want XS longer or more structured
than the in-game editor allows.

## Sources

- Primary: [AoE2ScenarioParser](https://github.com/KSneijders/AoE2ScenarioParser),
  the library DEscape builds on: its XS cheatsheet (`docs/cheatsheets/xs.md`),
  `objects/managers/xs_manager.py`, and the per-version structure
  definitions under `versions/DE/`.
- [AoE2DE UGC Guide, XS section](https://divy1211.github.io/AoE2DE_UGC_Guide/general/xs/beginner/)
  for the language itself. Not reproduced here.
- On-disk evidence from a real AoE2:DE install: the auto-generated
  `default0.xs`, `Constants.xs` and `Effects.xs` under the profile's
  `resources/_common/xs/` directory.
- A read-only census of 230 loadable real scenario files, run with
  `tools/census_xs_usage.py`.

All content below is paraphrased or measured, not copied. The worked example
in section 5 is original.

## 1. How XS reaches the game

A scenario can carry XS in two ways (section 2), but only one of them works
for everyone in a multiplayer lobby.

A file attached through the map's script-file setting is transferred to the
other *players* in a lobby, but not to spectators, who then cannot watch the
game.

Script placed in a **Script Call** trigger effect or condition avoids that.
When a scenario loads, the game collects every Script Call body and writes
them into `default0.xs`, a file it regenerates on each machine separately.
Nothing needs transferring, so spectators get the script too. The generated
file says so in its own banner: it is auto-generated, its functions are
written from the Script Call conditions and effects, and it must not be used
as a main XS file because it changes with the scenario.

## 2. The two surfaces

| Surface | Where it is stored | Used in practice |
|---|---|---|
| Trigger-level | Script Call effect (`message`) and Script Call condition (`xs_function`), inside the Triggers section | Yes: 30 effects across 20 of 230 loadable files, nearly all coop campaign scenarios. 0 conditions. |
| File-level | `Map` section `script_name`, plus `Files` section `script_file_path` and `script_file_content` at the end of the file | No: 0 of 230 files name a script, and 0 of the 206 with a readable Triggers section embed one. |

Both exist only from scenario version 1.40 onward. Older structure versions
store none of these fields.

DEscape edits the trigger-level surface in Triggers mode, through the same
write path as every other trigger field. The file-level surface is
**read-only**: the info panel shows the attached script name and the size of
any embedded script. It is not editable because `script_name` is a
variable-length string placed before the terrain data inside `Map`, so
changing its length would move every later offset a save patches by
position.

## 3. The in-game character limit

The in-game editor caps how much text a Script Call accepts. The cap is a
property of that editor's text box, not of the file: the effect's `message`
is a `str32`, a string with a 4-byte length prefix and no format-level
maximum. A script written with an external tool loads and runs at whatever
length it was saved with.

That is the main reason to author XS outside the game.

## 4. Constraints

- **Calls with parameters silently fail.** A Script Call that invokes a
  function with arguments, such as `spawnWave(3)`, does nothing, with no
  error shown. Wrap the call in a function that takes no parameters and call
  that instead. AoE2ScenarioParser refuses to emit such a call for this
  reason.
- **A Script Call holds either definitions or a call.** A body containing
  function definitions contributes code to `default0.xs`. A body that is a
  bare no-argument call, such as `announceWave();`, invokes that function.
  AoE2ScenarioParser's XS validation draws the same line between the two
  forms.
- **Line endings are CR only.** Every multi-line payload measured in real
  files separates lines with a lone carriage return (`\r`), not LF or CRLF.
  A tool that rewrites line endings changes the stored bytes even when the
  script text looks identical.
- **The builtins are undeclared on disk.** `Constants.xs` in the install
  holds only `extern const int` values, and `Effects.xs` holds tech-effect
  callbacks. The `xs*` builtin functions are part of the engine, so there is
  no local header to check a call against. Use the UGC guide's function
  reference.
- **Function names share one namespace.** Every Script Call body in the
  scenario lands in the same `default0.xs`, so two effects defining the same
  function name would collide there.

## 5. A worked example

A scenario that counts how many times a trigger has fired and exposes the
count to other triggers through trigger variable 3.

Effect on a trigger that runs once at game start, holding the definitions:

```xs
int waveCount = 0;

void countWave()
{
  waveCount = waveCount + 1;
  xsSetTriggerVariable(3, waveCount);
}
```

Effect on the repeating wave trigger, holding only the call:

```xs
countWave();
```

Points to note:

- `countWave` takes no parameters, so the call in the second effect is
  allowed (section 4).
- The trigger variable number is a plain literal inside the function rather
  than an argument, for the same reason.
- Other triggers read the count with an ordinary Variable Value condition on
  variable 3. No further XS is needed.
- As stored in the file, each line break in the first body is a single `\r`.

## 6. Further reading

- [AoE2DE UGC Guide, XS section](https://divy1211.github.io/AoE2DE_UGC_Guide/general/xs/beginner/):
  syntax, types, and the builtin function reference.
- `Constants.xs` and `Effects.xs` in the game install's
  `resources/_common/xs/` directory: the constants and callbacks that are
  auto-included in every script.
- `docs/INGAME_EDITOR_REFERENCE.md`: where Script Call and Script Filename
  sit in the in-game editor.
