//! IGSR safe wrapper — the public API.
//!
//! Stage 1: context lifecycle, config, backend trait + GL detection stub.
//! Upscale dispatch lands in stages 3–4.

pub mod backend;
pub mod config;
pub mod context;

pub use backend::{ComputePath, GpuBackend};
pub use config::{IgsrConfig, QualityMode};
pub use context::IgsrContext;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Core (C) version string, e.g. "igsr-core 0.1.0 (stage1)".
pub fn core_version() -> String {
    // SAFETY: C returns a static NUL-terminated string.
    unsafe {
        let p = igsr_sys::igsr_version_string();
        assert!(!p.is_null());
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
