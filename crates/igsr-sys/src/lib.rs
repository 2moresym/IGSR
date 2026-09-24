//! Raw unsafe FFI over `igsr-core` (C). No safety invariants beyond the
//! C contract: pointers must be valid, sizes non-zero. Prefer the safe
//! wrapper in the `igsr` crate.

use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgsrConfigFfi {
    pub render_w: u32,
    pub render_h: u32,
    pub display_w: u32,
    pub display_h: u32,
    pub prefer_compute: i32,
    pub three_pass: i32,
    pub reset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgsrParamsFfi {
    pub render_size: [f32; 2],
    pub display_size: [f32; 2],
    pub render_size_rcp: [f32; 2],
    pub display_size_rcp: [f32; 2],
    pub jitter: [f32; 2],
    pub clip_to_prev_clip: [f32; 16],
    pub pre_exposure: f32,
    pub camera_fov_hor: f32,
    pub camera_near: f32,
    pub min_lerp_contrib: f32,
    pub same_camera_frames: u32,
    pub reset: u32,
}

pub enum IgsrContextOpaque {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgsrFrameInputsFfi {
    pub jitter: [f32; 2],
    pub clip_to_prev: [f32; 16],
    pub pre_exposure: f32,
    pub camera_fov_hor: f32,
    pub camera_near: f32,
    pub min_lerp_contrib: f32,
    pub same_camera_frames: u32,
    pub reset: u32,
}

// Explicit static link: the archive is built by the `igsr-core` crate's
// build script (cc), whose `-L native=...` search path Cargo already
// propagates to final links. The attribute records the native dep in this
// rlib's metadata so downstream crates (igsr, testbed) resolve the symbols.
#[link(name = "igsr_core", kind = "static")]
extern "C" {
    pub fn igsr_create(cfg: *const IgsrConfigFfi) -> *mut IgsrContextOpaque;
    pub fn igsr_destroy(ctx: *mut IgsrContextOpaque);
    pub fn igsr_resize(
        ctx: *mut IgsrContextOpaque,
        render_w: u32,
        render_h: u32,
        display_w: u32,
        display_h: u32,
    ) -> c_int;
    pub fn igsr_calc_jitter(frame_index: u64, out_jitter: *mut f32);
    pub fn igsr_pass_count(ctx: *const IgsrContextOpaque) -> u32;
    pub fn igsr_render_dispatch(
        ctx: *const IgsrContextOpaque,
        local: u32,
        out_xyz: *mut u32,
    );
    pub fn igsr_display_dispatch(
        ctx: *const IgsrContextOpaque,
        local: u32,
        out_xyz: *mut u32,
    );
    pub fn igsr_frame_index(ctx: *const IgsrContextOpaque) -> u64;
    pub fn igsr_advance_frame(ctx: *mut IgsrContextOpaque);
    pub fn igsr_version_string() -> *const c_char;
    pub fn igsr_fill_params(
        ctx: *const IgsrContextOpaque,
        inputs: *const IgsrFrameInputsFfi,
        out: *mut IgsrParamsFfi,
    );
    pub fn igsr_reproject_motion(
        clip_xy: *const f32,
        depth: f32,
        m: *const f32,
        out_motion: *mut f32,
    );
}

/// Safe helper: run Halton jitter for a frame without a context.
pub fn calc_jitter(frame_index: u64) -> [f32; 2] {
    let mut out = [0.0f32; 2];
    // SAFETY: out is a valid 2-float buffer; C writes exactly 2 floats.
    unsafe { igsr_calc_jitter(frame_index, out.as_mut_ptr()) };
    out
}

/// Safe helper: CPU depth-reprojection motion (mirrors the convert shaders).
pub fn reproject_motion(clip_xy: [f32; 2], depth: f32, m: &[f32; 16]) -> [f32; 2] {
    let mut out = [0.0f32; 2];
    // SAFETY: all pointers are to valid stack arrays of the right size.
    unsafe {
        igsr_reproject_motion(
            clip_xy.as_ptr(),
            depth,
            m.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    out
}
