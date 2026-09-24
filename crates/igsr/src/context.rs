//! RAII context over the C `IgsrContext`.

use crate::config::IgsrConfig;
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
