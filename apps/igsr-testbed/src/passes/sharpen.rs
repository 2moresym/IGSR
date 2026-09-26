//! Pass 4 — sharpen (display resolution, stage 9 RCAS post-process).
//!
//! Own algorithm in `shaders/sharpen.{frag,comp}`. Strict post-process:
//! reads the upscale output, writes a new display target. The history
//! pair is owned by `Upscale` and holds the *unsharpened* frame, which is
//! what the before/after toggle shows — this pass declares no ping-pong
//! writes, so the framework can never accidentally sharpen history.

use super::{Pass, PassCtx, PassIo, Variants, Variant};
use crate::pipeline::{ResourceKind, ResourceSpec, SizeDomain};

pub struct Sharpen;

impl Sharpen {
    pub fn new() -> Sharpen {
        Sharpen
    }
}

const IO: PassIo = PassIo {
    reads: &["history"],
    writes: &["sharp"],
    // Terminal pass: the framework returns this as the frame output.
    final_output: Some("sharp"),
};

impl Pass for Sharpen {
    fn name(&self) -> &'static str {
        "sharpen"
    }
    fn io(&self) -> PassIo {
        IO
    }
    fn variants(&self) -> Variants {
        Variants::BOTH
    }
    fn sources(&self, variant: Variant) -> Option<(&'static str, &'static str)> {
        match variant {
            Variant::Compute => Some(("", igsr_shaders::SHARPEN_COMP)),
            Variant::Fragment => Some((igsr_shaders::FULLSCREEN_VERT, igsr_shaders::SHARPEN_FRAG)),
        }
    }
    fn resources(&self) -> &'static [ResourceSpec] {
        &[SHARP_SPEC]
    }
    fn dispatch(&mut self, ctx: &mut PassCtx) {
        let p = ctx.params;
        ctx.bind_tex(0, ctx.reads[0]); // upscale output (history write-tap)
        ctx.set_tex_unit("u_image", 0);
        ctx.set2f("u_display_size", p.display_size[0], p.display_size[1]);
        ctx.set2f("u_display_rcp", p.display_size_rcp[0], p.display_size_rcp[1]);
        // Sharpness is pipeline-owned, not part of the frame UBO.
        ctx.set1f("u_sharp", ctx.sharpness);
        let (dw, dh) = ctx.display_size;
        ctx.execute((dw + 7) / 8, (dh + 7) / 8, 1);
    }
}

/// Final display-res output. Scratch (no cross-frame role) — sharpening
/// never feeds back into history.
const SHARP_SPEC: ResourceSpec = ResourceSpec {
    name: "sharp",
    kind: ResourceKind::Scratch,
    internal: glow::RGBA16F as i32,
    upload_format: glow::RGBA,
    upload_type: glow::HALF_FLOAT,
    domain: SizeDomain::Display,
    filter: glow::NEAREST,
};
