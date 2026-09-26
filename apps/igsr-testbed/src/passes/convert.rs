//! Pass 1 — convert (render resolution).
//!
//! Own algorithm in `shaders/convert.{frag,comp}` (stage 3). This file
//! only declares data flow, supplies shader sources, and sets uniforms —
//! the framework owns FBO/image binding, variant choice, and timing.
//!
//! Outputs `data` (ping-pong). Upscale (and activate, in 3-pass) read it,
//! so activating or skipping the activate pass is purely a dataflow
//! consequence of the ping-pong flip, not a branch here.

use super::{Pass, PassCtx, PassIo, Variants, Variant};
use crate::pipeline::ResourceSpec;

pub struct Convert;

impl Convert {
    pub fn new() -> Convert {
        Convert
    }
}

const IO: PassIo = PassIo {
    reads: &["depth", "velocity"],
    writes: &["data"],
    final_output: None,
};

impl Pass for Convert {
    fn name(&self) -> &'static str {
        "convert"
    }
    fn io(&self) -> PassIo {
        IO
    }
    fn variants(&self) -> Variants {
        Variants::BOTH
    }
    fn sources(&self, variant: Variant) -> Option<(&'static str, &'static str)> {
        match variant {
            // Compute body only; the framework prepends the version prelude.
            Variant::Compute => Some(("", igsr_shaders::CONVERT_COMP)),
            Variant::Fragment => Some((igsr_shaders::FULLSCREEN_VERT, igsr_shaders::CONVERT_FRAG)),
        }
    }
    fn resources(&self) -> &'static [ResourceSpec] {
        // Also declares the app-provided externals: convert is the first
        // consumer, and the framework dedups by name for later passes.
        &[DEPTH_SPEC, VELOCITY_SPEC, COLOR_SPEC, DATA_SPEC]
    }
    fn dispatch(&mut self, ctx: &mut PassCtx) {
        let p = ctx.params;
        ctx.bind_tex(0, ctx.reads[0]); // depth
        ctx.bind_tex(1, ctx.reads[1]); // velocity
        ctx.set_tex_unit("u_depth", 0);
        ctx.set_tex_unit("u_velocity", 1);
        ctx.set2f("u_render_size", p.render_size[0], p.render_size[1]);
        ctx.set2f("u_render_rcp", p.render_size_rcp[0], p.render_size_rcp[1]);
        ctx.set_mat4_rowmajor("u_clip_to_prev", &p.clip_to_prev_clip);
        ctx.set1f("u_fov_hor", p.camera_fov_hor);
        let (rw, rh) = ctx.render_size;
        ctx.execute((rw + 7) / 8, (rh + 7) / 8, 1);
    }
}

// App-provided externals (the testbed scene pass writes these; the
// framework registers their real handles each frame via set_external).
const DEPTH_SPEC: ResourceSpec = ResourceSpec {
    name: "depth",
    kind: crate::pipeline::ResourceKind::External,
    internal: glow::R32F as i32,
    upload_format: glow::RED,
    upload_type: glow::FLOAT,
    domain: crate::pipeline::SizeDomain::Render,
    filter: glow::NEAREST,
};
const VELOCITY_SPEC: ResourceSpec = ResourceSpec {
    name: "velocity",
    kind: crate::pipeline::ResourceKind::External,
    internal: glow::RG32F as i32,
    upload_format: glow::RG,
    upload_type: glow::FLOAT,
    domain: crate::pipeline::SizeDomain::Render,
    filter: glow::NEAREST,
};
const COLOR_SPEC: ResourceSpec = ResourceSpec {
    name: "color",
    kind: crate::pipeline::ResourceKind::External,
    internal: glow::RGBA8 as i32,
    upload_format: glow::RGBA,
    upload_type: glow::UNSIGNED_BYTE,
    domain: crate::pipeline::SizeDomain::Render,
    filter: glow::NEAREST,
};

/// `data`: render-res RGBA16F, ping-pong so activate (when present) can
/// refine convert's output without a second named resource.
const DATA_SPEC: ResourceSpec = ResourceSpec {
    name: "data",
    kind: crate::pipeline::ResourceKind::PingPong { taps: 2 },
    internal: glow::RGBA16F as i32,
    upload_format: glow::RGBA,
    upload_type: glow::HALF_FLOAT,
    domain: crate::pipeline::SizeDomain::Render,
    filter: glow::NEAREST,
};
