# AGENTS.md

External viewer/editor for Age of Empires 2: Definitive Edition
`.aoe2scenario` files. Start with `README.md` — it covers setup, build/test
commands, and architecture.

## Hard rules

- Never write to a path under a Proton `compatdata/` prefix — that's the user's
  only copy of Workshop-synced scenario files. v1 has no write path at all; a
  future one must refuse any output path containing `compatdata`.
- GAIA units' `rotation` field is not an angle for ~65% of GAIA objects — it's a
  tree/doodad graphic-variant index (integer values well outside `[0, 2π)`, e.g.
  7..53). Any future write path must pass it through verbatim, never normalize it.

## Private maintainer repo

A `maintainer/` directory may exist alongside this repo on disk, gitignored
here and not part of this distribution — it's a separate, private git repo
with its own planning docs and its own fuller `AGENTS.md`. If you have local
filesystem access to it, check there for open-work tracking and deeper
context before starting non-trivial work; if you don't, it simply isn't part
of what you're working with.
