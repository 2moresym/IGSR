//! Own reimplemented GLSL for IGSR. Every shader here is written from
//! scratch to fill the *role* identified in docs/ALGORITHM_NOTES.md —
//! never copied or line-translated from reference/sgsr2.
//!
//! Stage 1: only the testbed's debug triangle. Pass shaders land in
//! stages 3–4.

pub const TRIANGLE_VERT: &str = include_str!("../shaders/triangle.vert");
pub const TRIANGLE_FRAG: &str = include_str!("../shaders/triangle.frag");

// Stage 3–4 placeholders so the module layout is stable from day one.
// Each returns the stub source checked into shaders/ (a `#error`-style
// comment until the real reimplementation lands).
pub const CONVERT_COMP_STUB: &str = include_str!("../shaders/convert.comp.stub");
pub const UPSCALE_COMP_STUB: &str = include_str!("../shaders/upscale.comp.stub");
pub const CONVERT_FRAG_STUB: &str = include_str!("../shaders/convert.frag.stub");
pub const UPSCALE_FRAG_STUB: &str = include_str!("../shaders/upscale.frag.stub");
pub const FULLSCREEN_VERT: &str = include_str!("../shaders/fullscreen.vert");
