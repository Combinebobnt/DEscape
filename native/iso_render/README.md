# iso_render — Phase 0 backend-decision benchmark evidence

**Outcome: pure numpy was chosen for Phase 1 onward — this directory's
Cython kernel is kept as reproducible benchmark evidence, not shipped or
built on by later phases.** The unaccelerated numpy prototype
already matched the existing flat renderer's per-tile cost on every real
example file, and a naive Cython port was measurably *slower* than numpy
on those same real files (only winning on a synthetic worst-case map size
no real file here approaches) — so there was no clear win to justify a
native-extension toolchain this project doesn't otherwise need.

Not the real Phase 1+ isometric renderer. This is the throwaway-but-honest
prototype the "real isometric Z-height terrain rendering" plan's Phase 0
calls for: implement the same representative per-tile kernel `render.py`'s
`render_tile` already does (crop the cached terrain texture, shade by
elevation, blit) — just blitted at an isometric screen position instead of
a flat grid position — once in pure Python/numpy and once in Cython, and
benchmark the two against each other and against the existing flat
renderer on real example files. The goal is a real, measured µs/tile
number to replace this plan's "(prototype, unverified)" 23-29µs/tile
estimate, and a real backend decision, before any of Phase 1's actual
`iso_geometry.py`/diamond-warp/skirt-face work starts.

Deliberately does **not** do diamond warping, skirt faces, or real
elevation-aware canvas bounds — those are Phase 1's job (`iso_geometry.py`,
verified by `tools/verify_iso_geometry.py`). Phase 0's kernel is
intentionally the simplest faithful proxy: same crop+shade cost, same
block size, different (isometric) destination offset per tile — enough to
measure whether Python-loop/numpy-per-call overhead dominates and whether
a native backend is worth the phases of work that follow.

Files:
- `iso_bench_numpy.py` — pure Python/numpy version of the per-tile iso
  blit kernel plus a full-map driver.
- `iso_kernel.pyx` — Cython port of the same per-tile blit, statically
  typed over memoryviews (`nogil` inner loop). The per-tile Python-level
  loop (attribute access, texture cache lookup) stays in Python on both
  sides — only the actual crop/shade/clip/blit numeric work moves into
  Cython, matching how a real Cython port of `render_tile` would look.
- `setup.py` — builds `iso_kernel.pyx` into a native extension via
  `cythonize`. Build with (from this directory, using the project's own
  venv):

  ```
  ../../.venv/bin/python setup.py build_ext --inplace
  ```

  This produces a `.so` importable as `iso_kernel`. No CI/wheel-building
  infrastructure exists for this yet — see the plan's Phase 0 section for
  why that's an explicitly separate, later follow-up.

Driven by `tools/bench_iso_backend.py` at the project root — see that
script for the actual benchmark numbers.
