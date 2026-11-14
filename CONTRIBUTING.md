# Contributing to DEscape

Bug reports, feature ideas and pull requests are all welcome. This page says
where each goes and what a pull request needs before it can be merged.

Posts written by an agent should open with `Automated comment: `

## Reporting a bug or proposing a feature

Open a [GitHub issue](https://github.com/Combinebobnt/DEscape/issues). For a
bug, include:

- the DEscape version (Help > About) or the commit you run from source;
- your OS, and whether you use a download or run from source;
- the scenario's game version, if you know it;
- the steps that show the problem, and what you expected instead;
- the scenario file or a screenshot, when you can share one.

## Pull requests

Pull requests are welcome. A small fix (a typo, an obvious bug together with
its test) can go straight to a PR. Anything larger starts as an issue, so the
approach is agreed before you write the code. Keep one change per PR.

CI runs the quick test suite on Linux and Windows for every PR, about an hour
each, and only the native kernel tests on macOS. A first-time contributor's run waits for the
maintainer to approve it. The corpus tests are not in CI, so still say in the
PR whether you ran them (below).

## Setup

Follow README's "Manual setup", then install the development tools into the
same venv:

```
pip install -r requirements-dev.txt
```

That brings pytest, pytest-qt, pytest-xdist and the pinned ruff. CI uses
Python 3.11. The native renderer is optional (README, "Faster renderer").

## The bar for a PR

- `./run_quick_tests.sh` (Linux/macOS) or `run_quick_tests.bat` (Windows)
  passes. It runs the quick test suite, which includes ruff through
  `tests/test_lint.py`.
- `bash tools/ci_parity.sh` reproduces CI's environment on Linux/macOS, if you
  want to rule out something your machine has and CI won't.
- New behavior comes with a quick-suite test. `tests/README.md`, "Adding a
  test", says which marker a test needs.
- A change to something the user guide describes updates its topic in
  `descape/help/` in the same PR, and a new menu item, mode or tool gets a
  line in its topic. `tests/test_help_guide.py` checks the guide's style,
  menu paths and links; the guide names commands, never their default keys.
- A `tests/test_module_size.py` failure means a file outgrew its line
  ceiling. Put the new code in a sibling module, or, if it belongs in that
  file, raise its ceiling in that test's table with a dated reason on the
  line (the failure message prints the line to paste). The review decides.
- Never skip or loosen a failing test to get a green run. If a test looks
  wrong, say so in the PR. A reasoned ceiling raise in
  `tests/test_module_size.py`, as above, is not loosening a test.

## When the full suite matters

Most per-file corpus tests also run on a generated test map
(`tests/fixtures/test_map_240x240.aoe2scenario`). Those items are marked
`test_map` and are part of the quick suite, so they need no setup.

Run the full suite for a change to rendering or to anything that writes a
scenario. The scenario corpus (`examples/`) is not in the repo, so point the
suite at your own scenarios:

```
./run_full_tests.sh --scenario-dir=/path/to/your/scenarios
```

A few corpus checks compare counts measured on the maintainer's own
scenarios; they skip when `--scenario-dir` points elsewhere.

If you can't run it, say so in the PR. The maintainer runs it before merging.

## Changes that write scenario files

Read the "Hard rules" in `AGENTS.md` first: never write into a
`mods/subscribed/` folder, and pass a unit's `rotation` through verbatim
except in the exceptions listed there; `docs/WRITE_RULES.md` gives each
one's exact scope. Say in the PR how you checked the
written file. The maintainer confirms write-path changes in the real AoE2:DE
editor before a release.

## Changelog

Leave `CHANGELOG.md` alone. Describe the user-visible effect in the PR and the
maintainer writes the entry at merge.

## License

DEscape is GPL-3.0-or-later, and contributions are accepted under the same
license. Submit only code you have the right to submit, and no game assets
(README's "License" section covers Microsoft's content).
