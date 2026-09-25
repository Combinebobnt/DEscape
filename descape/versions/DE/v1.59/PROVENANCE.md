# Provenance: scenario version 1.59 definitions

`structure.json`, `conditions.json` and `effects.json` in this directory are
byte-for-byte copies of the same files from AoE2ScenarioParser
(https://github.com/KSneijders/AoE2ScenarioParser), taken from:

- branch: `feat/v1-59-support` (unmerged, not on PyPI)
- commit: `faadf3fde46c9027c43e6c332d84c614f4139e27` (2026-09-22)
- path: `AoE2ScenarioParser/versions/DE/v1.59/`

AoE2ScenarioParser is licensed under GPL-3.0, the same license as DEscape.
Copyright for these files remains with the AoE2ScenarioParser authors.

Upstream's `v1.59/default.aoe2scenario` is deliberately not copied: at that
commit it is byte-identical to their v1.58 default, and its header reads
`1.58`.

Why they are here: DEscape pins AoE2ScenarioParser 0.8.3, which ships
definitions up to scenario version 1.58. The library's own definition always
wins over this one (`scenario_io._repo_structure_path()`), so this directory
goes dormant, and should be deleted, once the pin moves to a release that
ships 1.59.

The 1.58 to 1.59 layout delta is two fields, neither linked by 0.8.3's
`Unit`/`Condition` classes:

- `UnitStruct.capture_flag` (s8, after `garrisoned_in_id`)
- `ConditionStruct.allow_in_fog` (s32, before `xs_function`)

`descape/unlinked_fields.py` carries both through edits so they follow their
unit or condition rather than a list slot.
