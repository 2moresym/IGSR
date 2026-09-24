# IGSR — HD 4000 profiling notes (stage 7)

Target: Intel HD Graphics 4000 (Ivy Bridge GT2) via Mesa crocus, 4 GB RAM.
Everything below was measured on that exact machine
(`OpenGL 4.2 (Core Profile) Mesa 26.1.2`, 207 extensions). Re-run the
commands here after any shader or pipeline change.

## 1. Compute availability — answered empirically

- `GL_ARB_compute_shader` **is advertised** on crocus 4.2, and compute
  shaders **work** — but only as `#version 420 core` + `#extension
  GL_ARB_compute_shader : require`. Plain `#version 430` is rejected
  (`GLSL 4.30 is not supported`).
- The backend encodes this in `backend::gl::compute_prelude()`; compute
  shader sources carry no `#version` line. If a future Mesa changes the
  answer, `./target/debug/igsr-testbed --selftest` prints exactly which
  prelude compiled (look for the `convert.comp probe` lines).
- Checklist:
  - [ ] `--selftest` → `convert.comp probe (#version 420 core...): ok`
  - [ ] default run → `backend: OpenGL (glow) path=Compute`
  - [ ] `--force-fallback` (or live key `F`) → `path=FragmentFallback`,
        image still correct (compare `--dump` outputs of both paths)

## 2. Numbers (stage 8A: measured with timer queries, no more guessing)

`GL_ARB_timer_query` **is supported** on crocus 4.2. Each pass is wrapped in
`TIME_ELAPSED` queries (`PassTimers` in `pipeline.rs`: 3 in-flight slots
per pass, results consumed 1–2 frames late, EMA overlay, `G` toggles the
readout). Fragment path, 677x760 display:

| render scale | convert | upscale | total |
| ------------ | ------- | ------- | ----- |
| 0.50 (338x380) | 0.39ms | 3.23ms | 3.58ms |
| 0.75 (507x570) | 0.90ms | 3.57ms | 4.29ms |
| 1.00 (677x760) | 1.60ms | 3.53ms | 5.02ms |

Reads as: convert scales with render pixels (real work), upscale is flat
(display-res work at fixed window size), totals self-sum. Trustworthy.

Compute path on this driver is **not measurable per-pass**: the first timed
query per frame reads ~6.5ms while activate/upscale read exactly 0.00,
even given whole frames to themselves. The 6.5ms barely moves when render
pixels quadruple (6.42 → 6.69ms), so it is a fixed per-frame cost — most
likely a full-pipeline drain triggered by the first compute dispatch after
fragment work — not the pass cost. Related finding: nested TIME_ELAPSED
queries mis-attribute on crocus (the outer query steals the inner's time),
so each query gets its own frame (rotating Convert → Upscale → Activate →
Total). Follow-ups if compute numbers are ever needed: `TIMESTAMP`
query-counter pairs (different driver path), or `intel_gpu_top` coarse
signal. Until then, compare paths with the fragment timings + `--dump`
image agreement, not the compute query readouts.

Fps is still vsync-capped (~59); the ms numbers above are the real signal.

## 3. Per-pass GPU timings — instrumented (stage 8A)

Covered by §2 above. Remaining gap: trustworthy *compute* per-pass numbers
(see the drain finding). The hooks live in `pipeline.rs::execute`
(`PassTimers::begin/end` around each pass); re-verify with `--selftest`
adjacent runs after any change. `intel_gpu_top` remains the coarse
cross-check while the testbed runs.

## 4. Correctness checks that already run here

- `cargo test --workspace` — 19 tests: Halton values/ranges, UBO packing,
  CPU reprojection vs translation, `detect_compute`/`gl_version`/
  `compute_prelude` matrix, mat4 inverse round-trip, keybind logic.
- `--selftest` — compiles every shader, runs convert + the full
  convert→upscale chain on the real driver, asserts readbacks
  (`(0,0,0,1)` convert porte-manteau, `(0.2,0.4,0.6)` flat-field chain).
- `--view <mode> --dump <file.ppm>` — headless captures of all five
  views; fragment vs compute dumps should agree pixel-near-identically
  (they do as of stage 6 — keep comparing after shader edits).
- Motion view reads **render-pixels/frame**; expect ~jitter-delta
  magnitudes (±1px) on static geometry. If the whole sky saturates,
  suspect the MRT clears in `draw()` (`clear_buffer_f32_slice` for
  attachments 1/2) — that exact bug happened in stage 6.

## 5. Memory discipline (4 GB box)

Approximate GPU footprint at 1280x720 display, 0.5 render (640x360),
RGBA16F history x2 + RGBA8/RG32F/R32F scene + RGBA16F convert x2 +
RG16F luma x2: ~25 MB. Comfortable. It scales with the square of the
scale slider — at scale 1.0 + 1080p display, re-check with
`intel_gpu_top` / dmesg for eviction pressure. `Pipeline::resize`
deletes superseded textures (no leak on live scale changes), verified by
repeated `+`/`-` runs holding steady fps.

## 6. Useful commands

```
cargo test --workspace
./target/debug/igsr-testbed --selftest
./target/debug/igsr-testbed --dump /tmp/frag.ppm
./target/debug/igsr-testbed --three-pass --dump /tmp/comp.ppm
./target/debug/igsr-testbed --view motion --dump /tmp/m.ppm
intel_gpu_top   # in another terminal while the testbed runs
```

Keys in the window: `V` cycle views, `M` motion, `H` luma-history,
`+`/`-` live scale, `R` history reset (camera-cut path), `F`
compute/fragment toggle, `T` 2/3-pass toggle, `G` gpu-times overlay.
Headless flags: `--scale`, `--view`, `--dump`, `--selftest`,
`--debug-events`, `--force-fallback`, `--three-pass`.
