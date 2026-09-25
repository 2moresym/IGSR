//! Own reimplemented GLSL for IGSR. Every shader here is written from
//! scratch to fill the *role* identified in docs/ALGORITHM_NOTES.md —
//! never copied or line-translated from reference/sgsr2.
//!
//! Stage 1: only the testbed's debug triangle. Pass shaders land in
//! stages 3–4.

pub const TRIANGLE_VERT: &str = include_str!("../shaders/triangle.vert");
pub const TRIANGLE_FRAG: &str = include_str!("../shaders/triangle.frag");

pub const CONVERT_FRAG: &str = include_str!("../shaders/convert.frag");
pub const CONVERT_COMP: &str = include_str!("../shaders/convert.comp");

// Stage 4: reimplemented upscale (own code) + 3-pass activate (own code).
// Upscale: one algorithm, two entry points — fragment (single RGB output,
// history == previous output) and compute (explicit history/confidence
// buffer). Activate: compute-only quality stage (temporal clip + luma
// deltas), no `#version` line (backend prepends via compute_prelude),
// same as CONVERT_COMP.
pub const UPSCALE_FRAG: &str = include_str!("../shaders/upscale.frag");
pub const UPSCALE_COMP: &str = include_str!("../shaders/upscale.comp");
pub const ACTIVATE_COMP: &str = include_str!("../shaders/activate.comp");
// Stage 9: sharpen post-pass (own RCAS implementation). Fragment writes one
// RGB target; compute uses the prelude convention (no `#version` line).
pub const SHARPEN_FRAG: &str = include_str!("../shaders/sharpen.frag");
pub const SHARPEN_COMP: &str = include_str!("../shaders/sharpen.comp");
pub const FULLSCREEN_VERT: &str = include_str!("../shaders/fullscreen.vert");
