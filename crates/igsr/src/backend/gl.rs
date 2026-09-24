//! OpenGL/GLES backend — works today. Own code; no reference code.
//!
//! Stage 1: extension-string sniffing for `GL_ARB_compute_shader`
//! (or GL >= 4.3 / GLES >= 3.1). Real program compile + dispatch lands
//! in stages 3–4.

use super::{ComputePath, GpuBackend};

/// GL backend handle. `compute_available` comes from the runtime query
/// in [`detect_compute`] (or forced off for fallback testing).
#[derive(Debug, Clone, Copy)]
pub struct GlBackend {
    pub compute_available: bool,
    /// Force the fragment path even when compute exists (profiling knob).
    pub force_fallback: bool,
}

impl GlBackend {
    pub fn new(compute_available: bool) -> Self {
        Self {
            compute_available,
            force_fallback: false,
        }
    }

    pub fn with_forced_fallback(mut self) -> Self {
        self.force_fallback = true;
        self
    }
}

impl GpuBackend for GlBackend {
    fn name(&self) -> &'static str {
        "OpenGL (glow)"
    }

    fn compute_path(&self) -> ComputePath {
        if self.compute_available && !self.force_fallback {
            ComputePath::Compute
        } else {
            ComputePath::FragmentFallback
        }
    }
}

/// Parse a GL_VERSION + GL_EXTENSIONS pair. Pure function so it can be
/// unit-tested without a GL context.
///
/// Rules:
/// - Desktop GL version >= 4.3 implies compute.
/// - Otherwise the extension string must contain `GL_ARB_compute_shader`.
/// - GLES 3.1+ implies compute (checked via version string prefix "OpenGL ES 3.1").
pub fn detect_compute(version: &str, extensions: &str) -> bool {
    let v = version.trim();
    if let Some(es) = v.strip_prefix("OpenGL ES") {
        // e.g. "OpenGL ES 3.2 Mesa ..." or "OpenGL ES 3.10 ..."
        let mut nums = es
            .split_whitespace()
            .next()
            .unwrap_or("")
            .split('.')
            .filter_map(|p| p.parse::<u32>().ok());
        if let (Some(maj), Some(min)) = (nums.next(), nums.next()) {
            if maj > 3 || (maj == 3 && min >= 1) {
                return true;
            }
        }
        return extensions.split_whitespace().any(|e| e == "GL_ARB_compute_shader");
    }
    // Desktop GL: first two integers in the version string.
    let mut nums = v
        .split(|c: char| !(c.is_ascii_digit()))
        .filter(|s| !s.is_empty())
        .filter_map(|p| p.parse::<u32>().ok());
    if let (Some(maj), Some(min)) = (nums.next(), nums.next()) {
        if maj > 4 || (maj == 4 && min >= 3) {
            return true;
        }
    }
    extensions.split_whitespace().any(|e| e == "GL_ARB_compute_shader")
}

#[cfg(test)]
mod tests {
    use super::detect_compute;

    #[test]
    fn gl43_implies_compute() {
        assert!(detect_compute("4.5 (Core Profile) Mesa 26.1.2", ""));
    }

    #[test]
    fn gl42_with_ext_has_compute() {
        assert!(detect_compute(
            "4.2 (Compatibility Profile) Mesa 26.1.2",
            "GL_ARB_compute_shader GL_ARB_vertex_shader"
        ));
    }

    #[test]
    fn gl42_without_ext_no_compute() {
        assert!(!detect_compute("4.2 (Compatibility Profile) Mesa", "GL_ARB_vertex_shader"));
    }

    #[test]
    fn gles31_implies_compute() {
        assert!(detect_compute("OpenGL ES 3.1 Mesa", ""));
    }
}
