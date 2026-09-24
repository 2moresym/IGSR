//! RAII context over the C `IgsrContext`.

use crate::config::{FrameInputs, IgsrConfig};
use std::ptr::NonNull;

/// Safe owner of the C context. `Send` but not `Sync` (GL contexts are
/// thread-affine); created/destroyed on the render thread.
pub struct IgsrContext {
    ptr: NonNull<igsr_sys::IgsrContextOpaque>,
    config: IgsrConfig,
}

// SAFETY: context is thread-affine by convention; Send allows moving it to
// the render thread, matching how GL contexts are handled.
unsafe impl Send for IgsrContext {}

impl IgsrContext {
    pub fn new(config: IgsrConfig) -> Result<Self, &'static str> {
        let ffi = config.to_ffi();
        // SAFETY: ffi is a valid stack struct; C copies it or fails with NULL.
        let ptr = unsafe { igsr_sys::igsr_create(&ffi) };
        let ptr = NonNull::new(ptr).ok_or("igsr_create failed: invalid sizes?")?;
        Ok(Self { ptr, config })
    }

    pub fn config(&self) -> &IgsrConfig {
        &self.config
    }

    pub fn resize(
        &mut self,
        render_w: u32,
        render_h: u32,
        display_w: u32,
        display_h: u32,
    ) -> Result<(), &'static str> {
        // SAFETY: ptr is live (owned by self).
        let rc = unsafe {
            igsr_sys::igsr_resize(self.ptr.as_ptr(), render_w, render_h, display_w, display_h)
        };
        if rc == 0 {
            self.config.render_w = render_w;
            self.config.render_h = render_h;
            self.config.display_w = display_w;
            self.config.display_h = display_h;
            Ok(())
        } else {
            Err("igsr_resize failed")
        }
    }

    pub fn pass_count(&self) -> u32 {
        // SAFETY: ptr is live.
        unsafe { igsr_sys::igsr_pass_count(self.ptr.as_ptr()) }
    }

    pub fn frame_index(&self) -> u64 {
        // SAFETY: ptr is live.
        unsafe { igsr_sys::igsr_frame_index(self.ptr.as_ptr()) }
    }

    /// Current-frame jitter in [-0.5, 0.5] (Halton 2,3 from C core).
    pub fn jitter(&self) -> [f32; 2] {
        igsr_sys::calc_jitter(self.frame_index())
    }

    pub fn advance_frame(&mut self) {
        // SAFETY: ptr is live and &mut guarantees exclusivity.
        unsafe { igsr_sys::igsr_advance_frame(self.ptr.as_ptr()) };
    }

    pub fn render_dispatch(&self, local: u32) -> [u32; 3] {
        let mut out = [0u32; 3];
        // SAFETY: out is a valid 3-u32 buffer.
        unsafe { igsr_sys::igsr_render_dispatch(self.ptr.as_ptr(), local, out.as_mut_ptr()) };
        out
    }

    pub fn display_dispatch(&self, local: u32) -> [u32; 3] {
        let mut out = [0u32; 3];
        // SAFETY: out is a valid 3-u32 buffer.
        unsafe {
            igsr_sys::igsr_display_dispatch(self.ptr.as_ptr(), local, out.as_mut_ptr())
        };
        out
    }

    /// Pack the shader uniform block for this frame. Pure function of the
    /// config + inputs (computed in the C core so GL and Vireo share it).
    pub fn frame_params(&self, inputs: &FrameInputs) -> igsr_sys::IgsrParamsFfi {
        let ffi = inputs.to_ffi();
        let mut out = igsr_sys::IgsrParamsFfi {
            render_size: [0.0; 2],
            display_size: [0.0; 2],
            render_size_rcp: [0.0; 2],
            display_size_rcp: [0.0; 2],
            jitter: [0.0; 2],
            clip_to_prev_clip: [0.0; 16],
            pre_exposure: 0.0,
            camera_fov_hor: 0.0,
            camera_near: 0.0,
            min_lerp_contrib: 0.0,
            same_camera_frames: 0,
            reset: 0,
        };
        // SAFETY: ptr/ffi/out are all live for this call; C only reads/writes them.
        unsafe {
            igsr_sys::igsr_fill_params(self.ptr.as_ptr(), &ffi, &mut out);
        }
        out
    }

    /// Stage 1 stub. Full `.upscale(inputs) -> Texture` lands in stages 3–4
    /// once the reimplemented GLSL + GL dispatch exist.
    pub fn upscale_stub(&self) -> &'static str {
        "upscale() not yet implemented (stages 3-4)"
    }
}

impl Drop for IgsrContext {
    fn drop(&mut self) {
        // SAFETY: ptr was returned by igsr_create and not yet freed.
        unsafe { igsr_sys::igsr_destroy(self.ptr.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QualityMode;

    fn test_ctx() -> IgsrContext {
        let cfg = IgsrConfig::from_display(1280, 720, QualityMode::Balanced);
        IgsrContext::new(cfg).unwrap()
    }

    #[test]
    fn params_pack_sizes_and_rcps() {
        let ctx = test_ctx();
        let p = ctx.frame_params(&FrameInputs::reset_frame());
        assert_eq!(p.render_size, [755.0, 424.0]);
        assert_eq!(p.display_size, [1280.0, 720.0]);
        assert!((p.render_size_rcp[0] - 1.0 / 755.0).abs() < 1e-7);
        assert!((p.render_size_rcp[1] - 1.0 / 424.0).abs() < 1e-7);
        assert!((p.display_size_rcp[0] - 1.0 / 1280.0).abs() < 1e-9);
        assert_eq!(p.reset, 1);
    }

    #[test]
    fn params_carry_frame_inputs() {
        let ctx = test_ctx();
        let mut fi = FrameInputs::reset_frame();
        fi.jitter = [0.25, -0.125];
        fi.pre_exposure = 2.0;
        fi.reset = false;
        let p = ctx.frame_params(&fi);
        assert_eq!(p.jitter, [0.25, -0.125]);
        assert_eq!(p.pre_exposure, 2.0);
        assert_eq!(p.reset, 0);
    }

    #[test]
    fn jitter_stays_in_half_pixel() {
        for f in 1..=64u64 {
            let j = igsr_sys::calc_jitter(f);
            assert!(j[0] >= -0.5 && j[0] < 0.5, "f{f} x={}", j[0]);
            assert!(j[1] >= -0.5 && j[1] < 0.5, "f{f} y={}", j[1]);
        }
        // Known Halton(2,3) values: f1=(0,-1/6), f2=(-1/4,+1/6).
        let j1 = igsr_sys::calc_jitter(1);
        assert!((j1[0] - 0.0).abs() < 1e-6 && (j1[1] + 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn reproject_identity_is_zero_motion() {
        let ident = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let m = igsr_sys::reproject_motion([0.3, -0.5], 0.7, &ident);
        assert!(m[0].abs() < 1e-6 && m[1].abs() < 1e-6);
    }

    #[test]
    fn reproject_translation_matches() {
        // prev = curr shifted by (-0.1, +0.2): motion should read (+0.1, -0.2).
        let shift = [
            1.0, 0.0, 0.0, -0.1, //
            0.0, 1.0, 0.0, 0.2, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let m = igsr_sys::reproject_motion([0.0, 0.0], 0.5, &shift);
        assert!((m[0] - 0.1).abs() < 1e-6, "x={}", m[0]);
        assert!((m[1] + 0.2).abs() < 1e-6, "y={}", m[1]);
    }
}
