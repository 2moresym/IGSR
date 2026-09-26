//! Pipeline framework (stage 11) — the orchestrator that replaces the
//! hand-written per-pass blocks in the old `execute()`.
//!
//! Responsibilities, all generic:
//! - compile every pass's compute/fragment programs + view programs
//! - allocate and own every resource passes declare (ping-pong, scratch,
//!   external), reallocating on resize
//! - per frame: pick each pass's variant, resolve declared reads/writes,
//!   bind the FBO or image units, dispatch, flip ping-pong pairs
//! - per-pass timer queries via name registration (not a hardcoded enum)
//! - collect debug views offered by passes, resolved against the resource
//!   table at display time
//!
//! The algorithm (shaders, blend math, reset semantics) is untouched:
//! this file only re-expresses control flow. Regression bar is pixel-equal
//! captures vs the pre-refactor build.

use glow::HasContext as _;
use std::collections::HashMap;

use super::passes::{Pass, PassConfig, PassCtx, PassIo, Variant};

#[derive(Debug, Clone, Copy)]
pub enum SizeDomain {
    Render,
    Display,
}

#[derive(Debug, Clone, Copy)]
pub enum ResourceKind {
    /// Produced outside the framework (testbed scene pass). Registered
    /// per frame via `set_external`.
    External,
    /// Write-only per frame; no cross-frame role.
    Scratch,
    /// Stateful pair. Flips after any pass writes it. Ownership is
    /// expressed by a pass writing the name — the table never knows or
    /// cares which pass is stateful.
    PingPong { taps: usize },
}

/// A resource a pass declares. The framework allocates exactly one of
/// each unique name across the whole pipeline.
#[derive(Debug, Clone, Copy)]
pub struct ResourceSpec {
    pub name: &'static str,
    pub kind: ResourceKind,
    pub internal: i32,
    pub upload_format: u32,
    pub upload_type: u32,
    pub domain: SizeDomain,
    pub filter: u32,
}

struct ResourceEntry {
    spec: ResourceSpec,
    tex: Vec<glow::NativeTexture>,
    fbo: Vec<glow::NativeFramebuffer>,
    /// Ping-pong read tap; scratch always 0.
    read: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    pub render: (u32, u32),
    pub display: (u32, u32),
}

impl Sizes {
    pub fn dims(&self, d: SizeDomain) -> (u32, u32) {
        match d {
            SizeDomain::Render => self.render,
            SizeDomain::Display => self.display,
        }
    }
}

// ---- Shared texture helper (used by framework + passes) ----

pub fn make_tex(
    gl: &glow::Context,
    internal: i32,
    w: i32,
    h: i32,
    upload_format: u32,
    upload_type: u32,
    upload: Option<&[u8]>,
    filter: u32,
) -> glow::NativeTexture {
    unsafe {
        let t = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.tex_image_2d(
            glow::TEXTURE_2D, 0, internal, w, h, 0, upload_format, upload_type,
            glow::PixelUnpackData::Slice(upload),
        );
        t
    }
}

/// A view offered by a pass: a label, the logical resource it displays,
/// and the fragment shader that visualizes it.
pub struct ViewEntry {
    pub label: &'static str,
    pub source: &'static str,
    pub program: glow::NativeProgram,
}

pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
    res: Vec<ResourceEntry>,
    /// Programs per pass: [compute, fragment]. `None` = didn't compile.
    progs: Vec<[Option<glow::NativeProgram>; 2]>,
    /// Whether a pass is compiled/run at all this configuration.
    enabled: Vec<bool>,
    variant: Vec<Variant>,
    locs: HashMap<(usize, String), Option<glow::NativeUniformLocation>>,
    views: Vec<ViewEntry>,
    timers: PassTimers,
    sizes: Sizes,
    cfg: PassConfig,
    pub needs_reset: bool,
    // External scene targets owned by the testbed, bridged into the table.
    scene_fbo: glow::NativeFramebuffer,
    scene_rb: Option<glow::Renderbuffer>,
    scene_color: glow::NativeTexture,
    scene_vel: glow::NativeTexture,
    scene_depth: glow::NativeTexture,
    blit_prog: glow::NativeProgram,
    blit_vao: glow::NativeVertexArray,
    _blit_vbo: glow::NativeBuffer,
    timer_supported: bool,
    timer_frame: u64,
}

// ---- Per-pass timer queries, generalized by name registration ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassId {
    Convert = 0,
    Activate = 1,
    Upscale = 2,
    Total = 3,
    Sharp = 4,
    Scene = 5,
}

const N_TIMED: usize = 6;

pub struct PassTimers {
    supported: bool,
    slots: [[Option<glow::NativeQuery>; 3]; N_TIMED],
    cursor: [usize; N_TIMED],
    consumed: [[bool; 3]; N_TIMED],
    ms: [f32; N_TIMED],
    ema_ms: [f32; N_TIMED],
    valid: [bool; N_TIMED],
    /// Bit p set = pass p skipped this frame (one active query per frame;
    /// nested TIME_ELAPSED queries mis-report on crocus — stage 8A).
    mask: u8,
}

impl PassTimers {
    pub unsafe fn new(gl: &glow::Context, supported: bool) -> PassTimers {
        let mut t = PassTimers {
            supported,
            slots: [[None, None, None]; N_TIMED],
            cursor: [0; N_TIMED],
            consumed: [[true, true, true]; N_TIMED],
            ms: [0.0; N_TIMED],
            ema_ms: [0.0; N_TIMED],
            valid: [false; N_TIMED],
            mask: 0,
        };
        if !supported {
            return t;
        }
        unsafe {
            for p in 0..N_TIMED {
                for s in 0..3 {
                    match gl.create_query() {
                        Ok(q) => t.slots[p][s] = Some(q),
                        Err(_) => {
                            t.supported = false;
                            return t;
                        }
                    }
                }
            }
        }
        t
    }
    pub unsafe fn begin(&mut self, gl: &glow::Context, pass: PassId) {
        if !self.supported || (self.mask & (1 << pass as u8)) != 0 {
            return;
        }
        unsafe {
            let p = pass as usize;
            let s = self.cursor[p];
            self.cursor[p] = (s + 1) % 3;
            self.consumed[p][s] = false;
            if let Some(q) = self.slots[p][s] {
                gl.begin_query(glow::TIME_ELAPSED, q);
            }
        }
    }
    pub unsafe fn end(&self, gl: &glow::Context, pass: PassId) {
        if !self.supported || (self.mask & (1 << pass as u8)) != 0 {
            return;
        }
        unsafe {
            gl.end_query(glow::TIME_ELAPSED);
        }
    }
    pub unsafe fn poll(&mut self, gl: &glow::Context) {
        if !self.supported {
            return;
        }
        unsafe {
            for p in 0..N_TIMED {
                for s in 0..3 {
                    if self.consumed[p][s] {
                        continue;
                    }
                    if let Some(q) = self.slots[p][s] {
                        if gl.get_query_parameter_u32(q, glow::QUERY_RESULT_AVAILABLE) != 0 {
                            let ns = gl.get_query_parameter_u32(q, glow::QUERY_RESULT);
                            let ms = ns as f32 / 1e6;
                            self.ms[p] = ms;
                            self.ema_ms[p] = if self.valid[p] {
                                self.ema_ms[p] * 0.95 + ms * 0.05
                            } else {
                                ms
                            };
                            self.valid[p] = true;
                            self.consumed[p][s] = true;
                        }
                    }
                }
            }
        }
    }
    pub fn ema(&self, pass: PassId) -> Option<f32> {
        let v = self.valid[pass as usize];
        v.then_some(self.ema_ms[pass as usize])
    }
    pub fn report(&self) -> String {
        format_report(
            self.supported,
            self.ema(PassId::Convert),
            self.ema(PassId::Activate),
            self.ema(PassId::Upscale),
            self.ema(PassId::Total),
            self.ema(PassId::Sharp),
            self.ema(PassId::Scene),
        )
    }
    pub fn timer_armed(&self, pass: PassId) -> bool {
        self.supported && (self.mask & (1 << pass as u8)) == 0
    }
    pub unsafe fn timer_begin(&mut self, gl: &glow::Context, pass: PassId) {
        unsafe { self.begin(gl, pass) }
    }
    pub unsafe fn timer_end(&self, gl: &glow::Context, pass: PassId) {
        unsafe { self.end(gl, pass) }
    }
}

fn format_report(
    supported: bool,
    convert_ms: Option<f32>,
    activate_ms: Option<f32>,
    upscale_ms: Option<f32>,
    total_ms: Option<f32>,
    sharp_ms: Option<f32>,
    scene_ms: Option<f32>,
) -> String {
    if !supported {
        return "gpu=[timer queries unsupported]".into();
    }
    let mut s = String::from("gpu=[");
    s.push_str(&format!("convert {} ", fmt_ms(convert_ms)));
    s.push_str(&format!("activate {} ", fmt_ms(activate_ms)));
    s.push_str(&format!("upscale {} ", fmt_ms(upscale_ms)));
    s.push_str(&format!("sharp {} ", fmt_ms(sharp_ms)));
    s.push_str(&format!("scene {} ", fmt_ms(scene_ms)));
    s.push_str(&format!("total {}]", fmt_ms(total_ms)));
    s
}

fn fmt_ms(v: Option<f32>) -> String {
    match v {
        Some(ms) => format!("{ms:.2}ms"),
        None => "--".into(),
    }
}

impl Pipeline {
    pub unsafe fn new(
        gl: &glow::Context,
        gl_major: u32,
        gl_minor: u32,
        backend_compute: bool,
        three_pass: bool,
        timer_supported: bool,
        rw: u32,
        rh: u32,
        dw: u32,
        dh: u32,
    ) -> Result<Pipeline, String> {
        unsafe {
            // The default pipeline: today's four passes, in order. Adding a
            // pass later = one `passes::` file + one line here; nothing else
            // in the framework changes.
            let passes: Vec<Box<dyn Pass>> = vec![
                Box::new(super::passes::convert::Convert::new()),
                Box::new(super::passes::activate::Activate::new()),
                Box::new(super::passes::upscale::Upscale::new()),
                Box::new(super::passes::sharpen::Sharpen::new()),
            ];
            let mut p = Pipeline {
                passes,
                res: Vec::new(),
                progs: Vec::new(),
                enabled: Vec::new(),
                variant: Vec::new(),
                locs: HashMap::new(),
                views: Vec::new(),
                timers: PassTimers::new(gl, timer_supported),
                sizes: Sizes { render: (rw, rh), display: (dw, dh) },
                cfg: PassConfig::new(backend_compute, three_pass),
                needs_reset: true,
                scene_fbo: gl.create_framebuffer().unwrap(),
                scene_rb: None,
                scene_color: gl.create_texture().unwrap(),
                scene_vel: gl.create_texture().unwrap(),
                scene_depth: gl.create_texture().unwrap(),
                blit_prog: gl.create_program().unwrap(),
                blit_vao: gl.create_vertex_array().unwrap(),
                _blit_vbo: gl.create_buffer().unwrap(),
                timer_supported,
                timer_frame: 0,
            };
            p.compile_pass_programs(gl, gl_major, gl_minor, backend_compute);
            p.compile_view_programs(gl);
            p.build_blit(gl);
            p.resize(gl, rw, rh, dw, dh);
            p.resolve_variants();
            Ok(p)
        }
    }

    /// Compile both variants of every pass (compute body gets the version
    /// prelude). A variant that fails leaves `None`; the variant chooser
    /// falls back to the other one (the stage-1 contract).
    unsafe fn compile_pass_programs(
        &mut self,
        gl: &glow::Context,
        _maj: u32,
        _min: u32,
        backend_compute: bool,
    ) {
        {
            let prelude = igsr::backend::gl::compute_prelude(_maj, _min, backend_compute);
            for pass in &self.passes {
                let mut pair = [None, None];
                let variants = pass.variants();
                if variants.compute {
                    if let Some(pre) = prelude {
                        if let Some((_, body)) = pass.sources(Variant::Compute) {
                            let src = format!("{pre}{body}");
                            pair[0] = igsr::backend::gl::compile_compute(gl, &src).ok();
                        }
                    }
                }
                if variants.fragment {
                    if let Some((vs, fs)) = pass.sources(Variant::Fragment) {
                        pair[1] = igsr::backend::gl::compile_program(gl, vs, fs).ok();
                    }
                }
                self.progs.push(pair);
                self.enabled.push(false);
                self.variant.push(Variant::Fragment);
            }
        }
    }

    unsafe fn compile_view_programs(&mut self, gl: &glow::Context) {
        {
            for pass in &self.passes {
                for vs in pass.debug_views() {
                    match igsr::backend::gl::compile_program(
                        gl,
                        igsr_shaders::FULLSCREEN_VERT,
                        vs.frag,
                    ) {
                        Ok(program) => {
                            self.views.push(ViewEntry { label: vs.label, source: vs.source, program })
                        }
                        Err(e) => {
                            eprintln!("[pipeline] view {} failed to compile: {e}", vs.label);
                        }
                    }
                }
            }
        }
    }

    unsafe fn build_blit(&mut self, gl: &glow::Context) {
        unsafe {
            const BLIT_FRAG: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() { o_col = vec4(texture(u_tex, v_uv).rgb, 1.0); }
";
            let p = igsr::backend::gl::compile_program(
                gl,
                igsr_shaders::FULLSCREEN_VERT,
                BLIT_FRAG,
            )
            .expect("blit program");
            let verts: [f32; 12] = [-1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 2.0];
            let vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(self.blit_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                std::slice::from_raw_parts(verts.as_ptr() as *const u8, verts.len() * 4),
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 4 * 4, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 4 * 4, 2 * 4);
            gl.enable_vertex_attrib_array(1);
            gl.bind_vertex_array(None);
            self._blit_vbo = vbo;
            self.blit_prog = p;
        }
    }

    /// Allocate/refresh every declared resource + external scene targets.
    pub unsafe fn resize(&mut self, gl: &glow::Context, rw: u32, rh: u32, dw: u32, dh: u32) {
        unsafe {
            self.sizes = Sizes { render: (rw, rh), display: (dw, dh) };
            let sharpness = self.cfg.sharpness;
            self.cfg = PassConfig::new(self.cfg.use_compute, self.cfg.three_pass);
            self.cfg.sharpness = sharpness;
            self.build_scene_targets(gl, rw, rh);
            self.build_resources(gl);
            self.needs_reset = true;
            self.timer_frame = 0;
            self.resolve_variants();
        }
    }

    unsafe fn build_scene_targets(&mut self, gl: &glow::Context, rw: u32, rh: u32) {
        unsafe {
            let none: Option<&[u8]> = None;
            self.scene_color = make_tex(
                gl, glow::RGBA8 as i32, rw as i32, rh as i32,
                glow::RGBA, glow::UNSIGNED_BYTE, none, glow::NEAREST,
            );
            self.scene_vel = make_tex(
                gl, glow::RG32F as i32, rw as i32, rh as i32,
                glow::RG, glow::FLOAT, none, glow::NEAREST,
            );
            self.scene_depth = make_tex(
                gl, glow::R32F as i32, rw as i32, rh as i32,
                glow::RED, glow::FLOAT, none, glow::NEAREST,
            );
            let rb = gl.create_renderbuffer().unwrap();
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, rw as i32, rh as i32);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            if let Some(old) = self.scene_rb.replace(rb) {
                gl.delete_renderbuffer(old);
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scene_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.scene_color), 0);
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1, glow::TEXTURE_2D, Some(self.scene_vel), 0);
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT2, glow::TEXTURE_2D, Some(self.scene_depth), 0);
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rb));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1, glow::COLOR_ATTACHMENT2]);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE, "scene FBO incomplete");
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// (Re)allocate the union of all pass resource declarations.
    unsafe fn build_resources(&mut self, gl: &glow::Context) {
        unsafe {
            for e in &self.res {
                for t in &e.tex {
                    gl.delete_texture(*t);
                }
                for f in &e.fbo {
                    gl.delete_framebuffer(*f);
                }
            }
            self.res.clear();
            // Collect unique specs across all passes.
            for pass in &self.passes {
                for spec in pass.resources() {
                    if self.res.iter().any(|e| e.spec.name == spec.name) {
                        continue;
                    }
                    let (w, h) = self.sizes.dims(spec.domain);
                    let taps = match spec.kind {
                        ResourceKind::External | ResourceKind::Scratch => 1,
                        ResourceKind::PingPong { taps } => taps,
                    };
                    let mut tex = Vec::with_capacity(taps);
                    let mut fbo = Vec::with_capacity(taps);
                    let (aw, ah) = match spec.kind {
                        // Externals are replaced wholesale by set_external
                        // each frame; allocate a 1x1 slot, not full-size.
                        ResourceKind::External => (1, 1),
                        _ => (w as i32, h as i32),
                    };
                    for _ in 0..taps {
                        let t = make_tex(
                            gl, spec.internal, aw, ah,
                            spec.upload_format, spec.upload_type, None, spec.filter,
                        );
                        tex.push(t);
                        let f = gl.create_framebuffer().unwrap();
                        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(f));
                        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(t), 0);
                        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                        assert_eq!(
                            gl.check_framebuffer_status(glow::FRAMEBUFFER),
                            glow::FRAMEBUFFER_COMPLETE
                        );
                        fbo.push(f);
                    }
                    self.res.push(ResourceEntry { spec: *spec, tex, fbo, read: 0 });
                }
            }
            // Clear the render-res debug-visible pairs (data/luma) to black,
            // exactly as the pre-refactor resize did, so 2-pass views read
            // defined memory rather than garbage.
            for name in ["luma"] {
                if let Some(e) = self.res.iter_mut().find(|e| e.spec.name == name) {
                    for f in e.fbo.iter() {
                        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(*f));
                        gl.clear_color(0.0, 0.0, 0.0, 0.0);
                        gl.clear(glow::COLOR_BUFFER_BIT);
                    }
                }
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    /// Register a texture the testbed scene produced this frame.
    pub fn set_external(&mut self, name: &str, tex: glow::NativeTexture) {
        if let Some(e) = self.res.iter_mut().find(|e| e.spec.name == name) {
            if e.tex.is_empty() {
                e.tex.push(tex);
            } else {
                e.tex[0] = tex;
            }
        }
    }

    fn res_idx(&self, name: &str) -> Option<usize> {
        self.res.iter().position(|e| e.spec.name == name)
    }

    /// Read handle for a name (ping-pong current tap).
    fn read_tex(&self, name: &str) -> Option<glow::NativeTexture> {
        let e = self.res.get(self.res_idx(name)?)?;
        e.tex.get(e.read).copied()
    }

    /// Write tap index + handle for a name (ping-pong next tap, or 0).
    fn write_tex(&self, name: &str) -> Option<(usize, glow::NativeTexture)> {
        let e = self.res.get(self.res_idx(name)?)?;
        let idx = match e.spec.kind {
            ResourceKind::PingPong { taps } => (e.read + 1) % taps,
            _ => 0,
        };
        e.tex.get(idx).copied().map(|t| (idx, t))
    }

    /// Resolve a debug view's logical resource to its current read handle.
    pub fn view_tex(&self, label: &str) -> Option<glow::NativeTexture> {
        let v = self.views.iter().find(|v| v.label == label)?;
        self.read_tex(v.source)
    }
    /// The visualizer program a pass offered for `label`.
    pub fn view_program(&self, label: &str) -> Option<glow::NativeProgram> {
        self.views.iter().find(|v| v.label == label).map(|v| v.program)
    }
    /// The unsharpened frame (upscale output) for the before/after toggle.
    pub fn pre_sharpen_tex(&self) -> glow::NativeTexture {
        self.read_tex("history").unwrap_or(self.scene_color)
    }

    /// Start a frame's timer rotation. Must be called before the scene
    /// pass (which also times itself) — see `PassTimers::begin` for why
    /// only one query may be active per frame on this driver.
    pub fn begin_frame(&mut self) {
        const ALL: u8 = 0b111111;
        self.timers.mask = match self.timer_frame % 6 {
            0 => ALL & !(1 << PassId::Convert as u8),
            1 => ALL & !(1 << PassId::Upscale as u8),
            2 => ALL & !(1 << PassId::Activate as u8),
            3 => ALL & !(1 << PassId::Total as u8),
            4 => ALL & !(1 << PassId::Sharp as u8),
            _ => ALL & !(1 << PassId::Scene as u8),
        };
        self.timer_frame += 1;
    }

    /// Decide, per pass, whether it runs and on which variant. Called on
    /// resize / config change (cheap; not per frame).
    fn resolve_variants(&mut self) {
        let cfg = self.cfg;
        for (i, pass) in self.passes.iter().enumerate() {
            let enabled_by_cfg = pass.enabled(&cfg);
            let [comp, frag] = self.progs[i];
            let has = comp.is_some() || frag.is_some();
            self.enabled[i] = enabled_by_cfg && has;
            // Prefer compute when available; fall back to fragment. A
            // compute-only pass with no compute program is disabled.
            self.variant[i] = if comp.is_some() { Variant::Compute } else { Variant::Fragment };
        }
    }

    pub fn set_compute(&mut self, on: bool) {
        self.cfg.use_compute = on;
        self.resolve_variants();
    }
    pub fn set_three_pass(&mut self, on: bool) {
        self.cfg.three_pass = on;
        self.resolve_variants();
    }

    /// Run the whole pipeline for one frame. Returns the final output
    /// texture (the last pass's `final_output`).
    pub unsafe fn run(
        &mut self,
        gl: &glow::Context,
        params: &igsr_sys::IgsrParamsFfi,
    ) -> glow::NativeTexture {
        unsafe {
            // Scene targets are external inputs to convert/activate/upscale.
            let sc = self.scene_color;
            let sv = self.scene_vel;
            let sd = self.scene_depth;
            self.set_external("color", sc);
            self.set_external("velocity", sv);
            self.set_external("depth", sd);

            self.timers.poll(gl);
            self.timers.begin(gl, PassId::Total);

            let mut final_tex = sc;
            for i in 0..self.passes.len() {
                if !self.enabled[i] {
                    continue;
                }
                let io: PassIo = self.passes[i].io();
                let variant = self.variant[i];
                let idx = match variant {
                    Variant::Compute => 0,
                    Variant::Fragment => 1,
                };
                let prog = self.progs[i][idx];

                // Resolve reads (before this pass's writes).
                let mut reads = Vec::with_capacity(io.reads.len());
                for r in io.reads {
                    reads.push(self.read_tex(r).unwrap_or(sc));
                }
                // Resolve write targets.
                let mut writes = Vec::with_capacity(io.writes.len());
                for wname in io.writes {
                    writes.push(self.write_tex(wname));
                }

                // Bind the render target for this variant.
                match variant {
                    Variant::Compute => {
                        // Bind each declared write as an image unit, in
                        // declaration order (the shaders' binding = index).
                        for (i, wname) in io.writes.iter().enumerate() {
                            if let Some((_, tex)) = writes.get(i).copied().flatten() {
                                let fmt = self
                                    .res
                                    .get(self.res_idx(wname).unwrap())
                                    .map(|e| e.spec.internal as u32)
                                    .unwrap_or(glow::RGBA16F);
                                gl.bind_image_texture(
                                    i as u32, Some(tex), 0, false, 0,
                                    glow::WRITE_ONLY, fmt,
                                );
                            }
                        }
                    }
                    Variant::Fragment => {
                        // Viewport must match the target's domain (render vs
                        // display) — the old per-pass code set it inline.
                        if let Some(wname) = io.writes.first() {
                            if let Some(e) = self.res.get(self.res_idx(wname).unwrap()) {
                                let tap = match e.spec.kind {
                                    ResourceKind::PingPong { taps } => (e.read + 1) % taps,
                                    _ => 0,
                                };
                                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(e.fbo[tap]));
                                let (w, h) = self.sizes.dims(e.spec.domain);
                                gl.viewport(0, 0, w as i32, h as i32);
                            }
                        }
                    }
                }

                let timer_pass = match self.passes[i].name() {
                    "convert" => Some(PassId::Convert),
                    "activate" => Some(PassId::Activate),
                    "upscale" => Some(PassId::Upscale),
                    "sharpen" => Some(PassId::Sharp),
                    _ => None,
                };
                if let Some(tp) = timer_pass {
                    self.timers.begin(gl, tp);
                }

                // Bind the pass's program before any uniform upload or
                // dispatch — the old inline code did this per pass.
                gl.use_program(prog);

                let mut ctx = PassCtx::new(gl, params, &mut self.locs);
                ctx.set_active(prog, i * 2 + idx, variant, Some(self.blit_vao));
                ctx.sharpness = self.cfg.sharpness;
                ctx.reads = reads;
                ctx.writes = writes.iter().map(|w| w.map(|(_, t)| t).unwrap_or(sc)).collect();
                ctx.display_size = self.sizes.display;
                ctx.render_size = self.sizes.render;
                self.passes[i].dispatch(&mut ctx);
                gl.use_program(None);

                if let Some(tp) = timer_pass {
                    self.timers.end(gl, tp);
                }

                // Ping-pong flips for any declared write.
                for wname in io.writes {
                    if let Some(e) = self.res.iter_mut().find(|e| e.spec.name == *wname) {
                        if let ResourceKind::PingPong { taps } = e.spec.kind {
                            e.read = (e.read + 1) % taps;
                        }
                    }
                }

                if let Some(fo) = io.final_output {
                    if let Some(t) = self.read_tex(fo) {
                        final_tex = t;
                    }
                }
            }

            self.timers.end(gl, PassId::Total);
            self.needs_reset = false;
            final_tex
        }
    }

    // ---- public accessors preserved for main.rs ----
    pub fn scene_fbo(&self) -> glow::NativeFramebuffer {
        self.scene_fbo
    }
    pub fn render_size(&self) -> (u32, u32) {
        self.sizes.render
    }
    pub fn display_size(&self) -> (u32, u32) {
        self.sizes.display
    }
    pub fn use_compute(&self) -> bool {
        self.cfg.use_compute
    }
    pub fn three_pass(&self) -> bool {
        self.cfg.three_pass
    }
    pub fn sharpness(&self) -> f32 {
        self.cfg.sharpness
    }
    pub fn set_sharpness(&mut self, s: f32) {
        self.cfg.sharpness = s.clamp(0.0, 1.0);
    }
    pub fn can_compute(&self) -> bool {
        self.progs
            .iter()
            .enumerate()
            .all(|(i, p)| p[0].is_some() || !self.passes[i].enabled(&self.cfg))
    }
    pub fn timers_report(&self) -> String {
        self.timers.report()
    }
    pub fn timers_supported(&self) -> bool {
        self.timer_supported
    }
    pub fn timer_armed(&self, pass: PassId) -> bool {
        self.timers.timer_armed(pass)
    }
    pub unsafe fn timer_begin(&mut self, gl: &glow::Context, pass: PassId) {
        unsafe { self.timers.timer_begin(gl, pass) }
    }
    pub unsafe fn timer_end(&self, gl: &glow::Context, pass: PassId) {
        unsafe { self.timers.timer_end(gl, pass) }
    }
    pub fn blit_prog(&self) -> glow::NativeProgram {
        self.blit_prog
    }
    pub fn scene_color_tex(&self) -> glow::NativeTexture {
        self.scene_color
    }
    /// Viewport-clipped blit with an explicit program (split-screen).
    pub unsafe fn blit_region(
        &self,
        gl: &glow::Context,
        tex: glow::NativeTexture,
        prog: glow::NativeProgram,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
    ) {
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(x, y, w, h);
            gl.disable(glow::DEPTH_TEST);
            gl.use_program(Some(prog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if let Some(l) = gl.get_uniform_location(prog, "u_tex") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            // The motion view needs render size to express motion in
            // render-pixels; the blit/other views don't declare it.
            if let Some(l) = gl.get_uniform_location(prog, "u_render_size") {
                let (rw, rh) = self.sizes.render;
                gl.uniform_2_f32(Some(&l), rw as f32, rh as f32);
            }
            gl.bind_vertex_array(Some(self.blit_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
    pub unsafe fn blit_to_screen(
        &self,
        gl: &glow::Context,
        tex: glow::NativeTexture,
        w: i32,
        h: i32,
    ) {
        unsafe { self.blit_region(gl, tex, self.blit_prog, 0, 0, w, h) }
    }
}

#[cfg(test)]
mod timer_tests {
    use super::format_report;

    #[test]
    fn report_formats() {
        assert_eq!(
            format_report(false, None, None, None, None, None, None),
            "gpu=[timer queries unsupported]"
        );
        assert_eq!(
            format_report(true, Some(0.424), None, Some(1.096), Some(1.62), Some(0.31), Some(0.5)),
            "gpu=[convert 0.42ms activate -- upscale 1.10ms sharp 0.31ms scene 0.50ms total 1.62ms]"
        );
        assert_eq!(
            format_report(true, Some(0.424), Some(0.1), None, None, None, None),
            "gpu=[convert 0.42ms activate 0.10ms upscale -- sharp -- scene -- total --]"
        );
    }
}
