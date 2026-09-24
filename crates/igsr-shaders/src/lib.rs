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

// Stage 4 placeholders: reimplemented upscale (own code) lands next.
pub const UPSCALE_COMP_STUB: &str = include_str!("../shaders/upscale.comp.stub");
pub const UPSCALE_FRAG_STUB: &str = include_str!("../shaders/upscale.frag.stub");
pub const FULLSCREEN_VERT: &str = include_str!("../shaders/fullscreen.vert");
