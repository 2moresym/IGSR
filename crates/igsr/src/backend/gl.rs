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

/// Parse the leading `major.minor` from a GL_VERSION string.
/// Handles "4.2 (...)", "OpenGL ES 3.1 ...", etc. Used to gate IGSR passes
/// (convert needs GLSL 4.20 for textureGather) independently of compute.
pub fn gl_version(version: &str) -> (u32, u32) {
    let v = version.trim().strip_prefix("OpenGL ES").unwrap_or(version.trim());
    let mut nums = v
        .split(|c: char| !(c.is_ascii_digit()))
        .filter(|s| !s.is_empty())
        .filter_map(|p| p.parse::<u32>().ok());
    (nums.next().unwrap_or(0), nums.next().unwrap_or(0))
}

/// Version/extension prelude the backend prepends to every compute shader
/// body. Compute sources in `igsr-shaders` carry NO `#version` line; the
/// backend selects it at runtime:
///
/// - GL >= 4.3 → `#version 430 core`.
/// - GL 4.x + `GL_ARB_compute_shader` (HD 4000 / crocus: 4.2) →
///   `#version 420 core` + the extension directive (verified working on
///   Mesa 26.1.2 crocus in the stage-3 selftest).
/// - Otherwise → `None`: no compute path, use the fragment fallback.
pub fn compute_prelude(major: u32, minor: u32, compute_advertised: bool) -> Option<&'static str> {
    if major > 4 || (major == 4 && minor >= 3) {
        Some("#version 430 core\n")
    } else if major == 4 && compute_advertised {
        Some("#version 420 core\n#extension GL_ARB_compute_shader : require\n")
    } else {
        None
    }
}

/// Compile + link a raster program. Returns the program or the driver log.
/// Callers must hold a current GL context; errors are strings, never panics,
/// so the testbed selftest can report them cleanly on crocus.
pub fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
) -> Result<glow::NativeProgram, String> {
    use glow::HasContext as _;
    // SAFETY: caller guarantees a current context; all handles are fresh.
    unsafe {
        let vs = gl.create_shader(glow::VERTEX_SHADER).map_err(|e| e.to_string())?;
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            gl.delete_shader(vs);
            return Err(format!("vertex: {log}"));
        }
        let fs = gl.create_shader(glow::FRAGMENT_SHADER).map_err(|e| e.to_string())?;
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            return Err(format!("fragment: {log}"));
        }
        let prog = gl.create_program().map_err(|e| e.to_string())?;
        gl.attach_shader(prog, vs);
        gl.attach_shader(prog, fs);
        gl.link_program(prog);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            return Err(format!("link: {log}"));
        }
        Ok(prog)
    }
}

/// Compile a compute program (probe only in stage 3: tells us whether our
/// `#version 430` compute shaders build on the 4.2 crocus context).
pub fn compile_compute(gl: &glow::Context, cs_src: &str) -> Result<glow::NativeProgram, String> {
    use glow::HasContext as _;
    // SAFETY: same contract as compile_program.
    unsafe {
        let cs = gl.create_shader(glow::COMPUTE_SHADER).map_err(|e| e.to_string())?;
        gl.shader_source(cs, cs_src);
        gl.compile_shader(cs);
        if !gl.get_shader_compile_status(cs) {
            let log = gl.get_shader_info_log(cs);
            gl.delete_shader(cs);
            return Err(format!("compute: {log}"));
        }
        let prog = gl.create_program().map_err(|e| e.to_string())?;
        gl.attach_shader(prog, cs);
        gl.link_program(prog);
        gl.delete_shader(cs);
        if !gl.get_program_link_status(prog) {
            let log = gl.get_program_info_log(prog);
            gl.delete_program(prog);
            return Err(format!("link: {log}"));
        }
        Ok(prog)
    }
}

#[cfg(test)]
mod tests {
    use super::{detect_compute, gl_version};

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

    #[test]
    fn version_parses() {
        assert_eq!(gl_version("4.2 (Core Profile) Mesa 26.1.2"), (4, 2));
        assert_eq!(gl_version("OpenGL ES 3.1 Mesa"), (3, 1));
        assert_eq!(gl_version("3.3"), (3, 3));
    }

    #[test]
    fn prelude_picks_per_driver() {
        use super::compute_prelude;
        assert!(compute_prelude(4, 5, false).unwrap().contains("430"));
        assert!(compute_prelude(4, 2, true).unwrap().contains("420"));
        assert!(compute_prelude(4, 2, true).unwrap().contains("GL_ARB_compute_shader"));
        assert!(compute_prelude(4, 2, false).is_none());
        assert!(compute_prelude(3, 3, false).is_none());
    }
}
