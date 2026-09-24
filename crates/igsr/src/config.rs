//! Config: quality modes, resolutions, flags.

/// Upscale quality presets. Ratios mirror common dynamic-res steps and stay
/// conservative for 4 GB RAM (no preset exceeds 1080p output in stage 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMode {
    /// ~50% per axis (0.5x scale). Fastest.
    Performance,
    /// ~59% per axis.
    Balanced,
    /// ~67% per axis.
    Quality,
    /// Custom render scale in (0, 1].
    Custom(u8),
}

impl QualityMode {
    pub fn render_scale(self) -> f32 {
        match self {
            QualityMode::Performance => 0.5,
            QualityMode::Balanced => 0.59,
            QualityMode::Quality => 0.67,
            QualityMode::Custom(pct) => (pct.clamp(25, 100) as f32) / 100.0,
        }
    }
}

/// Safe config. Converts to the C `IgsrConfigFfi` at the FFI boundary.
#[derive(Debug, Clone)]
pub struct IgsrConfig {
    pub render_w: u32,
    pub render_h: u32,
    pub display_w: u32,
    pub display_h: u32,
    pub prefer_compute: bool,
    pub three_pass: bool,
}

impl IgsrConfig {
    pub fn new(render_w: u32, render_h: u32, display_w: u32, display_h: u32) -> Self {
        Self {
            render_w,
            render_h,
            display_w,
            display_h,
            prefer_compute: true,
            three_pass: false,
        }
    }

    /// Derive render size from display size + quality mode.
    pub fn from_display(display_w: u32, display_h: u32, mode: QualityMode) -> Self {
        let s = mode.render_scale();
        let rw = ((display_w as f32 * s) as u32).max(1);
        let rh = ((display_h as f32 * s) as u32).max(1);
        Self::new(rw, rh, display_w, display_h)
    }

    pub(crate) fn to_ffi(&self) -> igsr_sys::IgsrConfigFfi {
        igsr_sys::IgsrConfigFfi {
            render_w: self.render_w,
            render_h: self.render_h,
            display_w: self.display_w,
            display_h: self.display_h,
            prefer_compute: self.prefer_compute as i32,
            three_pass: self.three_pass as i32,
            reset: 1,
        }
    }
}
