//! Stage 11 pipeline framework — the generic `Pass` abstraction.
//!
//! A pass declares what it reads and writes and supplies its shader
//! sources; the framework (`super::Pipeline`) owns ordering, resource
//! resolution, ping-pong flips, the compute/fragment decision, uniform
//! caching, timer queries, and debug-view collection. Passes never bind
//! FBOs, never pick a dispatch path, never touch timers.
//!
//! Adding a pass (a future FSR2.2 disocclusion stage, for example) means
//! adding one file in this directory and one line to the default pipeline
//! construction. No existing file needs editing.

pub mod activate;
pub mod convert;
pub mod sharpen;
pub mod upscale;

use std::collections::HashMap;
use glow::HasContext as _;

/// Which dispatch path a pass runs. Selection is the framework's job
/// (compute availability + `variants()`); passes never branch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Compute,
    Fragment,
}

/// Which variants a pass implements. Default is both (the dual-path
/// convention from convert/upscale); a compute-only pass (activate)
/// overrides. The framework reports a variant the pass doesn't implement
/// instead of silently falling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variants {
    pub compute: bool,
    pub fragment: bool,
}

impl Variants {
    pub const BOTH: Variants = Variants { compute: true, fragment: true };
    pub const COMPUTE_ONLY: Variants = Variants { compute: true, fragment: false };
}

/// Logical resource name. Resolution to a GL handle happens once per frame
/// in the resource table; passes refer to resources by name only.
pub type ResourceRef = &'static str;

/// Declared data flow for one pass.
pub struct PassIo {
    /// Logical resources sampled this frame (resolved before dispatch).
    pub reads: &'static [ResourceRef],
    /// Logical resources written this frame. A write to a ping-pong
    /// resource causes the framework to flip that pair after dispatch.
    pub writes: &'static [ResourceRef],
    /// Name of the final output this pass produces, if it's terminal.
    /// The framework's `run` returns the last such handle.
    pub final_output: Option<ResourceRef>,
}

/// A debug view a pass offers. The framework compiles `frag` and resolves
/// `source` against the resource table at display time, so the view tracks
/// ping-pong taps automatically. `label` appears in the view cycle and
/// the `--view` list — no `main.rs` edit needed for a new one.
pub struct DebugViewSpec {
    pub label: &'static str,
    pub source: ResourceRef,
    pub frag: &'static str,
}

/// Per-dispatch context. Deliberately narrow: resolved resources, frame
/// params, and a uniform helper. Passes cannot reach the resource table
/// or the pipeline.
pub struct PassCtx<'a> {
    pub gl: &'a glow::Context,
    /// Frame uniforms packed by the C core.
    pub params: &'a igsr_sys::IgsrParamsFfi,
    /// Resolved read handles, indexed like `io().reads`.
    pub reads: Vec<glow::NativeTexture>,
    /// Resolved write handles, indexed like `io().writes`.
    pub writes: Vec<glow::NativeTexture>,
    /// Display size (for size-dependent dispatch math).
    pub display_size: (u32, u32),
    /// Render size.
    pub render_size: (u32, u32),
    /// Uniform-location cache, shared across the frame.
    locs: &'a mut HashMap<(usize, String), Option<glow::NativeUniformLocation>>,
    active_prog: Option<glow::NativeProgram>,
    active_key: usize,
    /// Framework-chosen variant. The pass never inspects it — it calls
    /// `execute` and the framework issues the matching draw/dispatch.
    variant: Variant,
    /// Fullscreen VAO for the fragment draw path (framework-owned).
    blit_vao: Option<glow::NativeVertexArray>,
    /// Pipeline-owned sharpness (sharpen pass reads it; not in the UBO).
    pub sharpness: f32,
}

impl<'a> PassCtx<'a> {
    pub(crate) fn new(
        gl: &'a glow::Context,
        params: &'a igsr_sys::IgsrParamsFfi,
        locs: &'a mut HashMap<(usize, String), Option<glow::NativeUniformLocation>>,
    ) -> Self {
        Self {
            gl,
            params,
            reads: Vec::new(),
            writes: Vec::new(),
            display_size: (0, 0),
            render_size: (0, 0),
            locs,
            active_prog: None,
            active_key: 0,
            variant: Variant::Fragment,
            blit_vao: None,
            sharpness: 0.3,
        }
    }

    /// Issue this pass's work. The framework has already bound the render
    /// target (FBO for fragment, image units for compute) and picked the
    /// variant, so this is the only "do the work" call a pass needs.
    pub fn execute(&mut self, gx: u32, gy: u32, gz: u32) {
        let compute = self.variant == Variant::Compute;
        unsafe {
            if compute {
                self.gl.dispatch_compute(gx, gy, gz);
                self.gl.memory_barrier(
                    glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
                );
            } else {
                if let Some(vao) = self.blit_vao {
                    self.gl.bind_vertex_array(Some(vao));
                    self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                    self.gl.bind_vertex_array(None);
                }
            }
        }
    }

    /// Point the uniform helper at the program about to be dispatched.
    /// `key` namespaces the cache (pass index × variant).
    pub(crate) fn set_active(
        &mut self,
        prog: Option<glow::NativeProgram>,
        key: usize,
        variant: Variant,
        blit_vao: Option<glow::NativeVertexArray>,
    ) {
        self.active_prog = prog;
        self.active_key = key;
        self.variant = variant;
        self.blit_vao = blit_vao;
    }

    /// Cached uniform location lookup (stage 10 noted the per-frame string
    /// lookups; caching is free here because a location is a property of
    /// the program, not of the frame).
    pub fn uni(&mut self, name: &str) -> Option<glow::NativeUniformLocation> {
        let Some(prog) = self.active_prog else { return None };
        let key = (self.active_key, name.to_string());
        if !self.locs.contains_key(&key) {
            let loc = unsafe { self.gl.get_uniform_location(prog, name) };
            self.locs.insert(key.clone(), loc);
        }
        self.locs.get(&key).cloned().flatten()
    }

    pub fn set1f(&mut self, name: &str, v: f32) {
        if let Some(l) = self.uni(name) {
            unsafe { self.gl.uniform_1_f32(Some(&l), v) }
        }
    }
    pub fn set1i(&mut self, name: &str, v: i32) {
        if let Some(l) = self.uni(name) {
            unsafe { self.gl.uniform_1_i32(Some(&l), v) }
        }
    }
    pub fn set2f(&mut self, name: &str, x: f32, y: f32) {
        if let Some(l) = self.uni(name) {
            unsafe { self.gl.uniform_2_f32(Some(&l), x, y) }
        }
    }
    /// Row-major source is transposed to column-major here (the UBO is
    /// row-major; GLSL `mat4` wants column-major).
    pub fn set_mat4_rowmajor(&mut self, name: &str, rowmajor: &[f32; 16]) {
        if let Some(l) = self.uni(name) {
            let col = crate::mat4::transpose(rowmajor);
            unsafe { self.gl.uniform_matrix_4_f32_slice(Some(&l), false, &col) }
        }
    }

    pub fn bind_tex(&mut self, unit: u32, tex: glow::NativeTexture) {
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + unit);
            self.gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        }
    }
    pub fn set_tex_unit(&mut self, name: &str, unit: u32) {
        self.set1i(name, unit as i32);
    }
}

/// One unit of GPU work.
///
/// `dispatch` does shader work only: bind declared reads to units, set
/// declared uniforms, issue the draw/dispatch. Everything else
/// (FBO/image binding, variant choice, timing, state flips) is the
/// framework's.
pub trait Pass {
    /// Stable unique identifier (timer label, view namespace).
    fn name(&self) -> &'static str;

    fn io(&self) -> PassIo;

    fn variants(&self) -> Variants {
        Variants::BOTH
    }

    /// Shader sources for a variant. Compute: body only (the framework
    /// prepends the version prelude). Fragment: (vertex, fragment).
    fn sources(&self, variant: Variant) -> Option<(&'static str, &'static str)>;

    /// Whether this pass runs in the current configuration. Default
    /// always; activate overrides (3-pass only).
    fn enabled(&self, _cfg: &PassConfig) -> bool {
        true
    }

    /// Resources this pass owns/needs. The framework allocates the union
    /// of all passes' declarations, deduped by name.
    fn resources(&self) -> &'static [crate::pipeline::ResourceSpec] {
        &[]
    }

    /// Debug views this pass offers (default: none). Collected even when
    /// the pass is disabled, so e.g. the 2-pass build still shows a
    /// luma/clip view (reading whatever the resource holds) — matching
    /// pre-refactor behavior.
    fn debug_views(&self) -> &'static [DebugViewSpec] {
        &[]
    }

    /// Do the work. `ctx` is pre-populated with resolved handles.
    fn dispatch(&mut self, ctx: &mut PassCtx);
}

/// Pipeline-wide configuration passed to `enabled`/`declare`.
#[derive(Debug, Clone, Copy)]
pub struct PassConfig {
    pub use_compute: bool,
    pub three_pass: bool,
    pub sharpness: f32,
}

impl PassConfig {
    pub fn new(use_compute: bool, three_pass: bool) -> Self {
        Self { use_compute, three_pass, sharpness: 0.3 }
    }
}
