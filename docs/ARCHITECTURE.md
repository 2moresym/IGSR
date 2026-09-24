# IGSR — Architecture (stage 1)

Own from-scratch temporal upscaler for Intel HD 4000 / Mesa crocus.
Rust for orchestration, C for hot-path math/dispatch sizing.

## Layout

- `crates/igsr-core` — C static lib (`src/igsr.{h,c}`, `jitter.c`,
  `passes/`). Never calls GPU APIs; GPU work goes through a
  function-pointer table filled by Rust (see `IgsrDispatchTable`).
- `crates/igsr-sys` — raw FFI. Unsafe, thin, no policy.
- `crates/igsr` — safe wrapper: `IgsrConfig`, `IgsrContext` (RAII),
  `GpuBackend` trait, `backend::gl` (runtime compute detection +
  forced-fallback knob), `backend::vireo` (feature-gated stub).
- `crates/igsr-shaders` — our own GLSL only. Reference shaders under
  `reference/sgsr2` are never compiled in.
- `apps/igsr-testbed` — standalone window. Stage 1: animated triangle +
  plumbing printout. Later: procedural scene, low-res render, IGSR
  upscale, UI/debug views.
- `reference/sgsr2` — read-only study copy of SGSR2 v2.
- `docs/ALGORITHM_NOTES.md` — stage 2: what each reference pass really does.

## Backend rule (HD 4000)

Query compute support at runtime (`detect_compute` in
`crates/igsr/src/backend/gl.rs`). Prefer compute when present, fall back
to fragment automatically; `--force-fallback` (`-f`) in the testbed forces
the fragment path for testing. Keep allocations conservative (4 GB RAM).

## Build order

1. ✅ Workspace + triangle (stage 1; triangle since replaced by the live scene).
2. ✅ `docs/ALGORITHM_NOTES.md` from the real shaders (stage 2).
3. ✅ Pass 1 (convert) reimplemented + wired (stage 3).
4. ✅ Pass 2 (CS + FS) + upscale/output + activate (stage 4).
5. ✅ Jitter + history ping-pong + camera jitter (stage 5).
6. ✅ UI controls + debug views — keybinds, no egui (stage 6).
7. ✅ HD 4000 profiling note (stage 7: `docs/PROFILING_HD4000.md`).
