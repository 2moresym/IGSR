//! Pass 2 — activate (render resolution, 3-pass quality only, compute-only).
//!
//! Own algorithm in `shaders/activate.comp` (stage 4; temporal depth test
//! added stage 8C). Refines convert's `data` into a second tap of the same
//! ping-pong pair and tracks the luma history in its own ping-pong pair.
//!
//! Compute-only by design (matches the studied structure); the framework
//! disables it when no compute path is available, which is also why the
//! pre-refactor `three_warned` runtime fallback is no longer needed.

use super::{DebugViewSpec, Pass, PassCtx, PassConfig, PassIo, Variants, Variant};
use crate::pipeline::{ResourceKind, ResourceSpec, SizeDomain};

pub struct Activate;

impl Activate {
    pub fn new() -> Activate {
        Activate
    }
}

const IO: PassIo = PassIo {
    reads: &["data", "color", "luma"],
    writes: &["data", "luma"],
    final_output: None,
};

impl Pass for Activate {
    fn name(&self) -> &'static str {
        "activate"
    }
    fn io(&self) -> PassIo {
        IO
    }
    fn variants(&self) -> Variants {
        Variants::COMPUTE_ONLY
    }
    fn sources(&self, variant: Variant) -> Option<(&'static str, &'static str)> {
        match variant {
            Variant::Compute => Some(("", igsr_shaders::ACTIVATE_COMP)),
            Variant::Fragment => None,
        }
    }
    fn enabled(&self, cfg: &PassConfig) -> bool {
        cfg.three_pass && cfg.use_compute
    }
    fn resources(&self) -> &'static [ResourceSpec] {
        &[LUMA_SPEC]
    }
    fn debug_views(&self) -> &'static [DebugViewSpec] {
        // Order matters: these append to the pipeline's view list, which
        // is the cycle order (motion, luma, clip) — same as pre-refactor.
        &[
            DebugViewSpec {
                label: "luma",
                source: "luma",
                frag: LUMA_VIEW,
            },
            DebugViewSpec {
                label: "clip",
                source: "data",
                frag: CLIP_VIEW,
            },
        ]
    }
    fn dispatch(&mut self, ctx: &mut PassCtx) {
        let p = ctx.params;
        ctx.bind_tex(0, ctx.reads[0]); // data (convert output)
        ctx.bind_tex(1, ctx.reads[1]); // color
        ctx.bind_tex(2, ctx.reads[2]); // luma history (read tap)
        ctx.set_tex_unit("u_data", 0);
        ctx.set_tex_unit("u_color", 1);
        ctx.set_tex_unit("u_luma_prev", 2);
        ctx.set2f("u_render_size", p.render_size[0], p.render_size[1]);
        ctx.set2f("u_render_rcp", p.render_size_rcp[0], p.render_size_rcp[1]);
        ctx.set1f("u_reset", p.reset as f32);
        ctx.set1f("u_fov_hor", p.camera_fov_hor);
        let (rw, rh) = ctx.render_size;
        ctx.execute((rw + 7) / 8, (rh + 7) / 8, 1);
    }
}

/// `luma`: render-res RG16F ping-pong, written only by activate.
const LUMA_SPEC: ResourceSpec = ResourceSpec {
    name: "luma",
    kind: ResourceKind::PingPong { taps: 2 },
    internal: glow::RG16F as i32,
    upload_format: glow::RG,
    upload_type: glow::HALF_FLOAT,
    domain: SizeDomain::Render,
    filter: glow::NEAREST,
};

// Debug view programs (visualizers; identical GLSL to the pre-refactor
// versions — only *where they live* changed).
const LUMA_VIEW: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() {
    vec2 l = texture(u_tex, v_uv).rg;
    float d = clamp(l.y * 8.0, -0.5, 0.5);
    o_col = vec4(l.x + d, l.x - abs(d) * 0.5, l.x - d, 1.0);
}
";

const CLIP_VIEW: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() {
    vec4 d = texture(u_tex, v_uv);
    o_col = vec4(d.w, d.z, 0.0, 1.0);
}
";
