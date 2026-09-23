# Review packs: having an agent judge a render it knows nothing about

`tools/gen_review_pack.py` renders a **review pack** - a set of framed captures
plus a checklist fixed in code before the images exist - for a blind agent to
judge.

## Install

Nothing to install beyond the repo's own dev requirements; the tool is Qt-free
and renders off-engine.

## Quick usage

```sh
.venv/bin/python3 tools/gen_review_pack.py --pack shadow_band
.venv/bin/python3 tools/gen_review_pack.py --pack shadow_band --inject no-apex-wedge
.venv/bin/python3 tools/gen_review_pack.py --pack shadow_band --inject notch-inject
.venv/bin/python3 tools/gen_review_pack.py --pack shadow_band --inject no-tip
.venv/bin/python3 tools/gen_review_pack.py --pack shadow_band --replica 1   # 2nd clean copy
.venv/bin/python3 tools/gen_review_pack.py --check      # deterministic, writes nothing
```

Each run prints the opaque directory it wrote to. Record that pairing somewhere
the reviewer will never see, then spawn one agent per run with a one-line
prompt and nothing else:

> Read `<abs path>/<dir>/REVIEW.md` and follow it exactly. Reply with only the
> response block it asks for.

## Why it exists

Not because a human has to look. Because **the session that generated a capture
already knows what it was supposed to prove, and passes it.** This repo has that
recorded twice: a sprite floated half a tile above ground for days behind a
visual confirmation that passed, and "is the seam line continuous" was read as
answering "does the shadow read as connected".

So the deliverable is a checklist written before any image exists, handed to a
reviewer told nothing about what changed, plus controls proving the reviewer is
not vacuous.

## The protocol

1. **Pre-register** the expected verdict for every check, on every run - clean
   and each inject. Not just the clean column: a check that flips under the
   *wrong* inject is not measuring what its question says.
2. **Generate all runs before spawning anything.**
3. **One agent per run, separate calls.** Never one agent over several runs:
   there would be an answer key to read ahead.
4. **A `PASS` on the clean run is only trusted if the same check came back
   `FAIL` on its matching inject.** A check with no matching inject returns an
   untrusted verdict, and must be logged as such.
5. **Check the specificity frame in every run.** The pack ships one frame where
   the artifact is geometrically impossible. Marks reported there mean
   noise-level detection, and invalidate every positive verdict in that run.
6. **`CANNOT-TELL` is a bug in the pack**, not a pass. It means the framing
   failed.
7. **Re-word one check and you have re-run the whole pack.** The checks share a
   `REVIEW.md` and the reviewer reads all of it, so a verdict measured before an
   unrelated check was re-worded does not carry forward. Re-run every run.
8. **Run the clean pack three times** (`--replica 1` and `--replica 2` give the
   second and third copies their own opaque directories). A check whose three
   clean answers disagree is logged `UNSTABLE`, never `PASS`, whatever the
   majority says. Three agreeing answers show the verdict is stable within that
   round only. A later round can still come back the other way, so a clean 3/3
   is not proof across rounds.

## What a verdict may not close

- **Anything whose oracle is AoE2:DE itself** - ramp geometry, team-colour
  multiply against the running game, anything calibrated against real captures.
  Rendering an app's own output only ever proves the app agrees with itself.
- **Anything gated on an in-app acceptance pass**, or on the user's own taste.
- **Anything countable.** Extent, pixel equality, component counts and bbox
  coverage belong in an assertion, where they cannot drift.

A review can *disambiguate a symptom* so a fix can be designed. That is its
whole remit.

## Framing rules, enforced by `--check`

Measured by reading real captures out of `build/`, not chosen by preference: a
144x112 seam thumbnail was literally unjudgeable, a 1317x844 full-window grab
useless for render detail, and a 576x448 amplified A/B diff the most productive
class of all.

- Long edge **900-1500 px**. Below that there is nothing to judge; above it the
  vision pipeline downsamples anyway.
- **Integer nearest-neighbour upscale only**, factor recorded in the manifest.
  A smooth resample turns the 1px features under review into gradients.
- **Crop, never downscale.** An oversized frame is a cropping bug.
- **One feature per frame.** A frame cannot serve a chrome review and a
  render-detail review at once.
- Every check names at least one frame, and every frame is named by at least
  one check. Both directions are `--check` failures.

## Writing a pack

A pack is a `PackSpec` in the tool: frames, checks, and the injects that
validate them. `testkit/review_pack.py` holds the framing primitives and the
validation, with no Qt and no rendering, so the default tier exercises it
cheaply via `tests/test_review_pack.py`.

Three rules the pilot learned the hard way:

- **Nothing the reviewer can read may name the artifact or the run.** Not the
  question, not the check id, not the directory the frames sit in. The pilot's
  checks were renamed mid-build for exactly this: `straight_run_notch` named
  what it was hunting, and `flat_ground_control` announced that its frame was a
  control.
- **Ask the open question before the closed one**, or the reviewer agrees with
  whichever phrasing it was handed. Where two reports contradict each other, a
  forced choice between the symptoms (`GAP` / `EXTRA` / `NEITHER`) is the right
  shape, because `PASS`/`FAIL` would presuppose which one is present.
- **A closed question must name what to EXCLUDE, not just what to find**, and
  name it on a property measured off the frames rather than assumed. The pilot's
  straight-edge check returned the same verdict on the clean run and on its own
  inject, because the shading's expected per-tile taper answers "does anything
  stick out from the edge" just as well as the defect does. The separating
  property turned out not to be the one the pre-registration assumed, either:
  both marks sit on the *same* side of the contour, and what tells them apart is
  that the defect is a thin detached line of near-uniform darkness while the
  taper is a graded wedge attached to the line. Measure the discriminator before
  writing it into the question.
- **Write a check's answer options against the injects' measured masks**, not
  against the symptom's name. The corner check's `GAP` option described bare
  ground at the vertex, but its matching inject opens a 2-column slot through a
  dark mass and changes nothing else. So the re-posed options are `SPLIT` /
  `WHOLE`: is that mass cut through, full depth. `--check` asserts the mask
  topology each option rests on (clean corner 1 component, `no-tip` 2), so a
  render change that stops the inject producing its symptom fails loudly
  instead of quietly emptying the check.
- **A frame can carry a feature from a pass the control does not switch off.**
  The band's raw and control frames both keep the seam line, which is continuous
  under every inject. A break question asked of the raw frames can be answered
  off that line alone. Ask it of the diff frames, and exclude by name any line
  that also appears in the switched-off frame.
- **Every pack brings at least two injects, and they must differ in size.** A
  coarse one proves the reviewer is looking at the right frame; a localized one
  proves it resolves defects at the size actually under review. `--check`
  asserts each localized inject stays inside its own measured bound *and* stays
  under the coarse one's largest delta on every frame they share, so it cannot
  quietly drift coarse and leave the sensitivity control vacuous.
- **Beyond those two, a check wants its own matching inject** - one that
  reproduces the exact symptom that check asks about, so a clean-run verdict is
  trustworthy rather than merely recorded. `no-tip` is the pilot's: it gates the
  inner-corner tip pass off, which is literally the pre-`4bd5f07` render, and it
  is the only inject in the pack that cuts into a mass rather than adding a mark.
  Its size is the real defect's size, not a number this tool picked.
- **Measure each inject's reach; do not argue it from geometry.** `--check`
  prints, per inject, every frame it touched and by how much. The pilot's
  pre-registration asserted from the geometry that neither inject could reach
  the corner frames, and that was simply false - `no-apex-wedge` changes 68 px
  there. An off-diagonal cell predicted from an unmeasured reach is a guess.

## What the pilot's rounds actually found

All against the contact-shadow band. On 2026-09-18 there was a pilot of three
runs, then a second round of four after two of its checks were reworked. On
2026-09-22 a six-run test followed, on clean frames only, of why one verdict
moved between those rounds. Round 3 came the same day: six runs (clean x3 and
the three injects) after the corner and band checks were re-posed.

- The **specificity control held in every run to date** (the pilot's seven,
  the priming test's six, round 3's six) - no false marks on flat ground - so no positive verdict
  in any round is noise.
- **A reworked check went from vacuous to discriminating.** The straight-edge
  check returned the same verdict on the clean render and on its own inject in
  round 1, and the full pre-registered row in round 2 once its closed question
  named the expected shading as the thing to exclude. That rework only worked
  after the discriminator was *measured*: the property the first round assumed
  separated defect from expected shading turned out to be shared by both.
- **An inject can bite and still not make its check resolve.** Round 2 added the
  corner check's matching inject. It changes the render, is bounded, and reaches
  only the frames it should - and the check returned the same answer with it as
  without, because the symptom it reopens is a 2-column slot inside a dark mass
  while the check's `GAP` option describes bare ground. Shipping the control is
  not the same as earning the verdict.
- **A verdict moved with no change to its own question or its render.** The
  connectivity check returned `FAIL` on the clean run in round 1 and `PASS` on
  the same clean frames in round 2. Its question is byte-for-byte the same, and
  the render is too: between the two commits `iso_geometry.py` is untouched and
  `render.py` is 77 insertions with zero deletions, three new functions for a
  unit-drag preview that the pack, rendering with `with_units=False`, never
  reaches. The only edit a reviewer could see was a *neighbouring* check's
  wording, and the round-2 answers reach for the vocabulary that edit
  introduced. That priming hypothesis was then tested directly, and **did not
  hold** (2026-09-22). Six clean runs were made: three with round 2's exact
  `REVIEW.md` and three with only the neighbouring question put back to round
  1's. All six answered `FAIL`. The neighbour's wording did move its own check
  as expected (`PASS` x3 against `FAIL` x3), but not this one. So round 2's
  exact document, on the same frames, returned `PASS` three times in one round
  and `FAIL` three times in another. The verdict varies from round to round, and
  within a round it can agree with itself. Rule 7 stays anyway, because it is
  cheap. Rule 8 exists because of this.
- **One check returned the same verdict on all three runs** in round 1, which
  means it has no demonstrated resolution as phrased, whatever its clean-run
  answer says. That is a bug in the pack, and it is the single most useful thing
  the first round produced: without the inject runs, that check's clean-run
  `FAIL` would have been read as a finding.
- **Answer options written against measured masks resolved both re-posed
  checks** (round 3). The corner check became `SPLIT` / `WHOLE` on new diff
  frames, and the band check became broken-or-not on the diff frames with the
  seam line excluded. Each returned its full pre-registered row. The matching
  inject was the only run to flip it, and the three clean replicas agreed 3/3.
  29 of 30 cells matched. Reviewers placed the defects where they were
  measured: the corner slot at x=740-760 against a measured 740-759, and the
  band gaps at 40-90 px against 40 px and 72 px or more. The coarse inject
  breaks the thin lines in the corner frames, and both corner checks still said
  `WHOLE` there, so the named exclusion held.
- **The one miss was a low-confidence cell, and it went the informative way.**
  The raw-frame corner check was predicted `WHOLE` on `no-tip`, because the
  slot is only about 1.2 sigma of ground texture at pct 100. It came back
  `SPLIT`, with `WHOLE` on all three clean replicas. So a reviewer sees a slot
  at texture-noise level at 10x, unamplified. At 10x that is still not a 1:1
  answer. It is logged as evidence, and it closes nothing.

The lesson generalizes: **a verdict is worth what its control is worth.** Run
the injects, or do not report the verdict. Round 2 adds the corollary: a control
that runs and changes nothing about the answer has not bought you a verdict
either, it has told you the question is aimed wrong.
