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

## 2. Numbers so far (and why they don't conclude anything yet)

At 677x760 window, scale 0.5 (338x380 → 677x760), vsync on:

| path | fps |
| ---- | --- |
| fragment 2-pass | ~58 |
| compute 3-pass | ~59 |

Both sit at the vsync ceiling — the swap interval (`SwapInterval::Wait(1)`
in `init_gl`) caps the loop, so these numbers prove *correctness under
load*, not relative speed. To actually compare paths on this box:

1. Temporarily switch to `SwapInterval::DontWait` (one line in
   `apps/igsr-testbed/src/main.rs`), re-run both paths, compare the
   printed fps; and/or grow the window (render cost scales with pixels —
   try 1280x720 and 1920x1080 display with `-`/`+` scale sweeps).
2. Watch `intel_gpu_top` (package `intel-gpu-tools`) while running: 3D
   busy %, and whether the compute path changes the render-vs-compute
   balance crocus reports.

## 3. Per-pass GPU timings — not yet instrumented (known gap)

The fps overlay is CPU frame time only. The hooks for real timings:

- Wrap each pass in `pipeline.rs::execute` (`convert`, `activate`,
  `upscale` sections) with `GL_TIME_ELAPSED` queries
  (`gen_queries`, `begin_query`/`end_query`, read back 2 frames late to
  avoid stalls). crocus supports `GL_ARB_timer_query` on IVB — verify
  with the extension list first.
- Until then: use `--dump` frame captures + `intel_gpu_top` as the
  coarse signal, and the selftest readbacks as the correctness signal.

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
compute/fragment toggle, `T` 2/3-pass toggle.
