//! Pass 3 — upscale (display resolution).
//!
//! Own algorithm in `shaders/upscale.{frag,comp}` (stage 4): Lanczos
//! upsample + statistics box, history clamp, temporal blend. Dual-path.
//!
//! Reads `data` (convert's output, or activate's when the activate pass
//! ran and flipped the pair) plus the history ping-pong, writes the
//! history write-tap. In the compute variant it also writes `scene_out`
//! (the sharpened-copy buffer used for the display view); the fragment
//! variant needs only the single history target.
//!
//! The pre/post-RCAS comparison view (`pre_sharpen_tex`) is the history
//! write-tap, which is exactly the unsharpened frame — the framework
//! exposes it by resource name, so no special-casing lives here.

use super::{DebugViewSpec, Pass, PassCtx, PassIo, Variants, Variant};
use crate::pipeline::{ResourceKind, ResourceSpec, SizeDomain};

pub struct Upscale;

impl Upscale {
    pub fn new() -> Upscale {
        Upscale
    }
}

const IO: PassIo = PassIo {
    reads: &["color", "data", "history"],
    writes: &["history", "scene_out"],
    final_output: None,
};

impl Pass for Upscale {
    fn name(&self) -> &'static str {
        "upscale"
    }
    fn io(&self) -> PassIo {
        IO
    }
    fn variants(&self) -> Variants {
        Variants::BOTH
    }
    fn sources(&self, variant: Variant) -> Option<(&'static str, &'static str)> {
        match variant {
            Variant::Compute => Some(("", igsr_shaders::UPSCALE_COMP)),
            Variant::Fragment => Some((igsr_shaders::FULLSCREEN_VERT, igsr_shaders::UPSCALE_FRAG)),
        }
    }
    fn resources(&self) -> &'static [ResourceSpec] {
        &[HISTORY_SPEC, SCENE_OUT_SPEC]
    }
    fn debug_views(&self) -> &'static [DebugViewSpec] {
        &[DebugViewSpec {
            label: "motion",
            source: "data",
            frag: MOTION_VIEW,
        }]
    }
    fn dispatch(&mut self, ctx: &mut PassCtx) {
        let p = ctx.params;
        ctx.bind_tex(0, ctx.reads[0]); // color (render-res scene)
        ctx.bind_tex(1, ctx.reads[1]); // data (motion/disocc)
        ctx.bind_tex(2, ctx.reads[2]); // history (read tap)
        ctx.set_tex_unit("u_color", 0);
        ctx.set_tex_unit("u_data", 1);
        ctx.set_tex_unit("u_history", 2);
        ctx.set2f("u_render_size", p.render_size[0], p.render_size[1]);
        ctx.set2f("u_render_rcp", p.render_size_rcp[0], p.render_size_rcp[1]);
        ctx.set2f("u_display_size", p.display_size[0], p.display_size[1]);
        ctx.set2f("u_display_rcp", p.display_size_rcp[0], p.display_size_rcp[1]);
        ctx.set2f("u_jitter", p.jitter[0], p.jitter[1]);
        ctx.set1f("u_reset", p.reset as f32);
        ctx.set1f("u_min_lerp", p.min_lerp_contrib);
        ctx.set1f("u_full_taps", if p.same_camera_frames >= 2 { 1.0 } else { 0.0 });
        let (dw, dh) = ctx.display_size;
        ctx.execute((dw + 7) / 8, (dh + 7) / 8, 1);
    }
}

/// History color: display-res RGBA16F ping-pong. Owned by upscale; the
/// unsharpened frame lives here, which is what the before/after view reads.
const HISTORY_SPEC: ResourceSpec = ResourceSpec {
    name: "history",
    kind: ResourceKind::PingPong { taps: 2 },
    internal: glow::RGBA16F as i32,
    upload_format: glow::RGBA,
    upload_type: glow::HALF_FLOAT,
    domain: SizeDomain::Display,
    filter: glow::LINEAR,
};

/// Second compute image (scene color) written only by the compute variant.
const SCENE_OUT_SPEC: ResourceSpec = ResourceSpec {
    name: "scene_out",
    kind: ResourceKind::Scratch,
    internal: glow::RGBA16F as i32,
    upload_format: glow::RGBA,
    upload_type: glow::HALF_FLOAT,
    domain: SizeDomain::Display,
    filter: glow::NEAREST,
};

const MOTION_VIEW: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
uniform vec2 u_render_size;
void main() {
    vec4 d = texture(u_tex, v_uv);
    vec2 px = d.xy * u_render_size * 0.25;
    o_col = vec4(clamp(px * 0.5 + 0.5, 0.0, 1.0), d.z, 1.0);
}
";
