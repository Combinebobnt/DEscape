# Contributing to DEscape

Bug reports, feature ideas and pull requests are all welcome. This page says
where each goes and what a pull request needs before it can be merged.

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

CI does not run on pull requests. Say in the PR description what you ran and
what it printed.

## Setup

Follow README's "Manual setup", then install the development tools into the
same venv:

```
pip install -r requirements-dev.txt
```

That brings pytest, pytest-qt, pytest-xdist and the pinned ruff. CI uses
Python 3.11. The native renderer is optional (README, "Faster renderer").

## The bar for a PR

- `./run_all_tests.sh` (Linux/macOS) or `run_all_tests.bat` (Windows) passes.
  It runs the default test tier, which includes ruff through
  `tests/test_lint.py`.
- `bash tools/ci_parity.sh` reproduces CI's environment on Linux/macOS, if you
  want to rule out something your machine has and CI won't.
- New behavior comes with a default-tier test. `tests/README.md`, "Adding a
  test", says which tier a test belongs in.
- Never skip or loosen a failing test to get a green run. If a test looks
  wrong, say so in the PR.

## When the corpus tier matters

Run it for a change to rendering or to anything that writes a scenario. The
scenario corpus (`examples/`) is not in the repo, so point the tier at your
own scenarios:

```
.venv/bin/python3 -m pytest --scenario-dir=/path/to/your/scenarios -m "corpus or slow"
```

If you can't run it, say so in the PR. The maintainer runs it before merging.

## Changes that write scenario files

Read the "Hard rules" in `AGENTS.md` first: never write into a
`mods/subscribed/` folder, and pass a unit's `rotation` through verbatim
except in the exceptions listed there. Say in the PR how you checked the
written file. The maintainer confirms write-path changes in the real AoE2:DE
editor before a release.

## Changelog

Leave `CHANGELOG.md` alone. Describe the user-visible effect in the PR and the
maintainer writes the entry at merge.

## License

DEscape is GPL-3.0-or-later, and contributions are accepted under the same
license. Submit only code you have the right to submit, and no game assets
(README's "License" section covers Microsoft's content).
