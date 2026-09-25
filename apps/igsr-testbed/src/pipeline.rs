//! Live IGSR pipeline (own code): owns every FBO/texture, runs
//! convert → [activate] → upscale each frame with history ping-pong, and
//! blits the result to the window. Fragment path always available; compute
//! path used when the backend reports compute support (verbatim the stage-1
//! fallback contract). 3-pass activate needs compute (it is compute-only);
//! with `--three-pass` on a fragment-only backend we log once and run 2-pass.

use glow::HasContext as _;
use igsr::backend::gl as gl_backend;
use igsr_sys::IgsrParamsFfi;

const BLIT_FRAG: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() {
    o_col = vec4(texture(u_tex, v_uv).rgb, 1.0);
}
";

// Debug view: convert motion/disocclusion buffer. Motion is shown in
// render-resolution pixels per frame (±4px maps red/green around
// mid-grey); disocclusion goes to blue. Static + attached pixels read
// (0.5,0.5,0).
const MOTION_FRAG: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
uniform vec2 u_render_size;
void main() {
    vec4 d = texture(u_tex, v_uv);
    vec2 px = d.xy * u_render_size * 0.25;
    o_col = vec4(clamp(px * 0.5 + 0.5, 0.0, 1.0), d.z, 1.0);
}
";

// Debug view: luma history (R = luma grey, G = signed delta as red/blue).
// All black in 2-pass mode (no luma tracking) — that itself is the signal.
const LUMA_FRAG: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() {
    vec2 l = texture(u_tex, v_uv).rg;
    float d = clamp(l.y * 8.0, -0.5, 0.5);
    o_col = vec4(l.x + d, l.x - abs(d) * 0.5, l.x - d, 1.0);
}
";

// Debug view: activate output (combined disocclusion clip as green, luma
// edge flag as red). Reads the same buffer the motion view reads, so in
// 3-pass mode Motion shows (motion, combined clip) while this shows
// (edge, clip). Black in 2-pass mode (activate never runs).
const CLIP_FRAG: &str = "#version 420 core
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_col;
uniform sampler2D u_tex;
void main() {
    vec4 d = texture(u_tex, v_uv);
    o_col = vec4(d.w, d.z, 0.0, 1.0);
}
";

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

// ---- Per-pass GPU timers (stage 8A) ----

/// Passes we time. Order matches the report and the ms/valid/ema arrays.
/// `Total` brackets the whole execute (the only trustworthy number when the
/// driver mis-reports individual compute passes — see PROFILING_HD4000.md).
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

/// GL_ARB_timer_query instrumentation with 3 in-flight slots per pass, so
/// results are consumed 1–2 frames late and never stall the pipeline.
/// `u32` nanosecond reads are used deliberately: per-pass times are
/// millisecond-scale (no wrap risk below ~4s), which avoids glow's awkward
/// pointer-based u64 getter.
pub struct PassTimers {
    supported: bool,
    slots: [[Option<glow::NativeQuery>; 3]; N_TIMED],
    cursor: [usize; N_TIMED],
    consumed: [[bool; 3]; N_TIMED],
    ms: [f32; N_TIMED],
    ema_ms: [f32; N_TIMED],
    valid: [bool; N_TIMED],
    /// Bit p set = pass p is skipped this frame. The driver gets at most
    /// one active query per frame (see PROFILING_HD4000.md §2).
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

    pub fn supported(&self) -> bool {
        self.supported
    }

    /// Start timing `pass`. No-op when masked (another pass owns this frame).
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

    /// Stop timing. Same mask rule as begin (a skipped begin must pair
    /// with a skipped end — the pair always matches by construction).
    pub unsafe fn end(&self, gl: &glow::Context, pass: PassId) {
        if !self.supported || (self.mask & (1 << pass as u8)) != 0 {
            return;
        }
        unsafe {
            gl.end_query(glow::TIME_ELAPSED);
        }
    }

    /// Harvest available results without blocking. Call once per frame
    /// (start of execute); updates last + EMA milliseconds per pass.
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

    /// One-line overlay fragment, e.g. `gpu=[convert 0.42ms activate -- upscale 1.10ms
    /// total 1.60ms]`. Activate is shown only once it has produced a sample
    /// (3-pass runs).
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

    /// Whether `pass` owns this frame's query slot (rotation scheme).
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

/// Pure report formatting (unit-tested without GL).
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
    // Always shown (2-pass runs read `--`): an absent activate must be
    // visibly absent, not silently missing (stage 10 step 1).
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

fn tex2d(
    gl: &glow::Context,
    internal: i32,
    w: i32,
    h: i32,
    format: u32,
    ty: u32,
    data: glow::PixelUnpackData,
    filter: u32,
) -> glow::NativeTexture {
    unsafe {
        let t = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.tex_image_2d(glow::TEXTURE_2D, 0, internal, w, h, 0, format, ty, data);
        t
    }
}

pub struct Pipeline {
    pub use_compute: bool,
    pub three_pass: bool,
    three_warned: bool,
    rw: u32,
    rh: u32,
    dw: u32,
    dh: u32,
    pub needs_reset: bool,
    // Scene targets.
    scene_fbo: glow::NativeFramebuffer,
    scene_rb: Option<glow::Renderbuffer>,
    scene_color: glow::NativeTexture,
    scene_vel: glow::NativeTexture,
    scene_depth: glow::NativeTexture,
    // Convert targets.
    convert_fbo: glow::NativeFramebuffer,
    data_tex: glow::NativeTexture,
    act_fbo: glow::NativeFramebuffer,
    data2_tex: glow::NativeTexture,
    luma_tex: [glow::NativeTexture; 2],
    luma_read: usize,
    // History ping-pong (display res).
    hist_fbo: [glow::NativeFramebuffer; 2],
    hist_tex: [glow::NativeTexture; 2],
    hist_read: usize,
    // Convert/activate output consumed by the last upscale (debug views).
    last_data: Option<glow::NativeTexture>,
    // Compute-path scene copy (display res; also feeds stage-6 debug views).
    scene_out_tex: glow::NativeTexture,
    timers: PassTimers,
    timer_frame: u64,
    // Sharpen post-pass (stage 9): strict post-process on the upscale
    // output. History keeps the UNSHARPENED frame (sharpening history
    // would feed amplified detail back into temporal accumulation).
    sharp_fbo: glow::NativeFramebuffer,
    sharp_tex: glow::NativeTexture,
    sharpen_frag: glow::NativeProgram,
    sharpen_comp: Option<glow::NativeProgram>,
    pub sharpness: f32,
    // Programs.
    convert_frag: glow::NativeProgram,
    upscale_frag: glow::NativeProgram,
    blit_prog: glow::NativeProgram,
    motion_prog: glow::NativeProgram,
    luma_prog: glow::NativeProgram,
    clip_prog: glow::NativeProgram,
    convert_comp: Option<glow::NativeProgram>,
    upscale_comp: Option<glow::NativeProgram>,
    activate_comp: Option<glow::NativeProgram>,
    blit_vao: glow::NativeVertexArray,
    _blit_vbo: glow::NativeBuffer,
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
            let convert_frag =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, igsr_shaders::CONVERT_FRAG)?;
            let upscale_frag =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, igsr_shaders::UPSCALE_FRAG)?;
            let blit_prog =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, BLIT_FRAG)?;
            let motion_prog =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, MOTION_FRAG)?;
            let luma_prog =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, LUMA_FRAG)?;
            let clip_prog =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, CLIP_FRAG)?;
            let sharpen_frag =
                gl_backend::compile_program(gl, igsr_shaders::FULLSCREEN_VERT, igsr_shaders::SHARPEN_FRAG)?;

            // Compute programs (prelude + body). Any failure → fragment path.
            let prelude = gl_backend::compute_prelude(gl_major, gl_minor, backend_compute);
            let mut use_compute = backend_compute;
            let mut convert_comp = None;
            let mut upscale_comp = None;
            let mut activate_comp = None;
            let mut sharpen_comp = None;
            if let Some(pre) = prelude {
                let cc = |body: &str| {
                    let src = format!("{pre}{body}");
                    gl_backend::compile_compute(gl, &src).ok()
                };
                convert_comp = cc(igsr_shaders::CONVERT_COMP);
                upscale_comp = cc(igsr_shaders::UPSCALE_COMP);
                activate_comp = cc(igsr_shaders::ACTIVATE_COMP);
                sharpen_comp = cc(igsr_shaders::SHARPEN_COMP);
                if convert_comp.is_none() || upscale_comp.is_none() {
                    eprintln!("[pipeline] compute compile failed; using fragment path");
                    use_compute = false;
                }
            } else {
                use_compute = false;
            }

            // Shared fullscreen triangle VAO.
            let verts: [f32; 12] = [-1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 2.0];
            let blit_vao = gl.create_vertex_array().unwrap();
            let blit_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(blit_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(blit_vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, f32_bytes(&verts), glow::STATIC_DRAW);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 4 * 4, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 4 * 4, 2 * 4);
            gl.enable_vertex_attrib_array(1);
            gl.bind_vertex_array(None);

            let mut p = Pipeline {
                use_compute,
                three_pass,
                three_warned: false,
                rw: 0,
                rh: 0,
                dw: 0,
                dh: 0,
                needs_reset: true,
                scene_fbo: gl.create_framebuffer().unwrap(),
                scene_rb: None,
                scene_color: gl.create_texture().unwrap(),
                scene_vel: gl.create_texture().unwrap(),
                scene_depth: gl.create_texture().unwrap(),
                convert_fbo: gl.create_framebuffer().unwrap(),
                data_tex: gl.create_texture().unwrap(),
                act_fbo: gl.create_framebuffer().unwrap(),
                data2_tex: gl.create_texture().unwrap(),
                luma_tex: [gl.create_texture().unwrap(), gl.create_texture().unwrap()],
                luma_read: 0,
                hist_fbo: [gl.create_framebuffer().unwrap(), gl.create_framebuffer().unwrap()],
                hist_tex: [gl.create_texture().unwrap(), gl.create_texture().unwrap()],
                hist_read: 0,
                last_data: None,
                scene_out_tex: gl.create_texture().unwrap(),
                timers: PassTimers::new(gl, timer_supported),
                timer_frame: 0,
                convert_frag,
                upscale_frag,
                blit_prog,
                motion_prog,
                luma_prog,
                clip_prog,
                convert_comp,
                upscale_comp,
                activate_comp,
                sharp_fbo: gl.create_framebuffer().unwrap(),
                sharp_tex: gl.create_texture().unwrap(),
                sharpen_frag,
                sharpen_comp,
                sharpness: 0.3,
                blit_vao,
                _blit_vbo: blit_vbo,
            };
            p.resize(gl, rw, rh, dw, dh);
            Ok(p)
        }
    }

    /// (Re)create all targets. Marks history for reset (camera-cut path).
    /// Old textures/renderbuffers are deleted first so live scale changes
    /// don't leak VRAM on a 4 GB machine.
    pub unsafe fn resize(&mut self, gl: &glow::Context, rw: u32, rh: u32, dw: u32, dh: u32) {
        unsafe {
            for t in [
                self.scene_color, self.scene_vel, self.scene_depth,
                self.data_tex, self.data2_tex, self.scene_out_tex,
                self.sharp_tex,
                self.luma_tex[0], self.luma_tex[1],
                self.hist_tex[0], self.hist_tex[1],
            ] {
                gl.delete_texture(t);
            }
            if let Some(rb) = self.scene_rb.take() {
                gl.delete_renderbuffer(rb);
            }
            self.rw = rw;
            self.rh = rh;
            self.dw = dw;
            self.dh = dh;
            let (rw, rh, dw, dh) = (rw as i32, rh as i32, dw as i32, dh as i32);
            // NOTE: PixelUnpackData is not Copy; construct a fresh None per call.
            let no_data = || glow::PixelUnpackData::Slice(None);

            self.scene_color = tex2d(gl, glow::RGBA8 as i32, rw, rh, glow::RGBA, glow::UNSIGNED_BYTE, no_data(), glow::NEAREST);
            self.scene_vel = tex2d(gl, glow::RG32F as i32, rw, rh, glow::RG, glow::FLOAT, no_data(), glow::NEAREST);
            self.scene_depth = tex2d(gl, glow::R32F as i32, rw, rh, glow::RED, glow::FLOAT, no_data(), glow::NEAREST);
            let rb = gl.create_renderbuffer().unwrap();
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
            gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, rw, rh);
            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            self.scene_rb = Some(rb);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.scene_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.scene_color), 0);
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1, glow::TEXTURE_2D, Some(self.scene_vel), 0);
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT2, glow::TEXTURE_2D, Some(self.scene_depth), 0);
            gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(rb));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0, glow::COLOR_ATTACHMENT1, glow::COLOR_ATTACHMENT2]);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);

            self.data_tex = tex2d(gl, glow::RGBA16F as i32, rw, rh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::NEAREST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.convert_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.data_tex), 0);
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);

            self.data2_tex = tex2d(gl, glow::RGBA16F as i32, rw, rh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::NEAREST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.act_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.data2_tex), 0);
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);

            for i in 0..2 {
                self.luma_tex[i] = tex2d(gl, glow::RG16F as i32, rw, rh, glow::RG, glow::HALF_FLOAT, no_data(), glow::NEAREST);
                self.hist_tex[i] = tex2d(gl, glow::RGBA16F as i32, dw, dh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::LINEAR);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.hist_fbo[i]));
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.hist_tex[i]), 0);
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
                assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);
                // Clear histories so the first frame blends from black, not garbage.
                gl.clear_color(0.0, 0.0, 0.0, 0.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }
            // Sharpen target (display res). No history role — never cleared
            // for blending; every frame overwrites it fully.
            self.sharp_tex = tex2d(gl, glow::RGBA16F as i32, dw, dh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::NEAREST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.sharp_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(self.sharp_tex), 0);
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.luma_read = 0;
            self.hist_read = 0;
            self.last_data = None;
            self.scene_out_tex =
                tex2d(gl, glow::RGBA16F as i32, dw, dh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::NEAREST);
            // Clear debug-visible buffers that no pass writes in 2-pass mode
            // (activate output + luma history), so the clip/luma views read
            // black instead of uninitialized memory. Luma has no FBO of its
            // own; clear it through act_fbo's second attachment.
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.act_fbo));
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            for i in 0..2 {
                gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1, glow::TEXTURE_2D, Some(self.luma_tex[i]), 0);
                gl.draw_buffers(&[glow::COLOR_ATTACHMENT1]);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT1, glow::TEXTURE_2D, None, 0);
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.needs_reset = true;
        }
    }

    pub fn scene_fbo(&self) -> glow::NativeFramebuffer {
        self.scene_fbo
    }
    pub fn render_size(&self) -> (u32, u32) {
        (self.rw, self.rh)
    }

    fn uni(gl: &glow::Context, prog: glow::NativeProgram, name: &str) -> Option<glow::NativeUniformLocation> {
        unsafe { gl.get_uniform_location(prog, name) }
    }

    fn set_convert_uniforms(gl: &glow::Context, prog: glow::NativeProgram, p: &IgsrParamsFfi) {
        unsafe {
            if let Some(l) = Self::uni(gl, prog, "u_render_size") {
                gl.uniform_2_f32(Some(&l), p.render_size[0], p.render_size[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_render_rcp") {
                gl.uniform_2_f32(Some(&l), p.render_size_rcp[0], p.render_size_rcp[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_clip_to_prev") {
                // Params store row-major; GLSL wants column-major → transpose.
                gl.uniform_matrix_4_f32_slice(Some(&l), true, &p.clip_to_prev_clip);
            }
            if let Some(l) = Self::uni(gl, prog, "u_fov_hor") {
                gl.uniform_1_f32(Some(&l), p.camera_fov_hor);
            }
        }
    }

    fn set_upscale_uniforms(gl: &glow::Context, prog: glow::NativeProgram, p: &IgsrParamsFfi) {
        unsafe {
            if let Some(l) = Self::uni(gl, prog, "u_render_size") {
                gl.uniform_2_f32(Some(&l), p.render_size[0], p.render_size[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_render_rcp") {
                gl.uniform_2_f32(Some(&l), p.render_size_rcp[0], p.render_size_rcp[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_display_size") {
                gl.uniform_2_f32(Some(&l), p.display_size[0], p.display_size[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_display_rcp") {
                gl.uniform_2_f32(Some(&l), p.display_size_rcp[0], p.display_size_rcp[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_jitter") {
                gl.uniform_2_f32(Some(&l), p.jitter[0], p.jitter[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_reset") {
                gl.uniform_1_f32(Some(&l), p.reset as f32);
            }
            if let Some(l) = Self::uni(gl, prog, "u_min_lerp") {
                gl.uniform_1_f32(Some(&l), p.min_lerp_contrib);
            }
            if let Some(l) = Self::uni(gl, prog, "u_full_taps") {
                gl.uniform_1_f32(Some(&l), if p.same_camera_frames >= 2 { 1.0 } else { 0.0 });
            }
        }
    }

    fn set_sharpen_uniforms(
        gl: &glow::Context,
        prog: glow::NativeProgram,
        p: &IgsrParamsFfi,
        sharpness: f32,
    ) {
        unsafe {
            if let Some(l) = Self::uni(gl, prog, "u_display_size") {
                gl.uniform_2_f32(Some(&l), p.display_size[0], p.display_size[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_display_rcp") {
                gl.uniform_2_f32(Some(&l), p.display_size_rcp[0], p.display_size_rcp[1]);
            }
            if let Some(l) = Self::uni(gl, prog, "u_sharp") {
                gl.uniform_1_f32(Some(&l), sharpness.clamp(0.0, 1.0));
            }
        }
    }

    fn bind_tex(gl: &glow::Context, unit: u32, tex: glow::NativeTexture) {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        }
    }

    fn barrier(gl: &glow::Context) {
        unsafe {
            gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
        }
    }

    fn draw_full(gl: &glow::Context, vao: glow::NativeVertexArray) {
        unsafe {
            gl.bind_vertex_array(Some(vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
    }

    /// Run convert → [activate] → upscale. Returns the display-res output
    /// texture (a history buffer, valid until the next execute).
    pub unsafe fn execute(&mut self, gl: &glow::Context, p: &IgsrParamsFfi) -> glow::NativeTexture {
        unsafe {
            self.timers.poll(gl);
            // One timed pass per frame, rotating Convert → Upscale →
            // Activate → Total. Back-to-back TIME_ELAPSED queries mis-report
            // on crocus (nested queries starve the inner ones); giving each
            // query a whole frame keeps every reading trustworthy.
            // EMA converges ~4x slower — acceptable for an overlay number.
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
            self.timers.begin(gl, PassId::Total);
            let three = self.three_pass && self.use_compute && self.activate_comp.is_some();
            if self.three_pass && !three && !self.three_warned {
                self.three_warned = true;
                eprintln!("[pipeline] 3-pass needs compute; running 2-pass");
            }
            let gx = (self.rw + 7) / 8;
            let gy = (self.rh + 7) / 8;
            let dx = (self.dw + 7) / 8;
            let dy = (self.dh + 7) / 8;

            // ---- Convert ----
            if self.use_compute {
                let prog = self.convert_comp.unwrap();
                self.timers.begin(gl, PassId::Convert);
                gl.use_program(Some(prog));
                Self::bind_tex(gl, 0, self.scene_depth);
                Self::bind_tex(gl, 1, self.scene_vel);
                if let Some(l) = Self::uni(gl, prog, "u_depth") {
                    gl.uniform_1_i32(Some(&l), 0);
                }
                if let Some(l) = Self::uni(gl, prog, "u_velocity") {
                    gl.uniform_1_i32(Some(&l), 1);
                }
                Self::set_convert_uniforms(gl, prog, p);
                gl.bind_image_texture(0, Some(self.data_tex), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.dispatch_compute(gx, gy, 1);
                Self::barrier(gl);
                self.timers.end(gl, PassId::Convert);
                gl.use_program(None);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.convert_fbo));
                gl.viewport(0, 0, self.rw as i32, self.rh as i32);
                self.timers.begin(gl, PassId::Convert);
                gl.use_program(Some(self.convert_frag));
                Self::bind_tex(gl, 0, self.scene_depth);
                Self::bind_tex(gl, 1, self.scene_vel);
                if let Some(l) = Self::uni(gl, self.convert_frag, "u_depth") {
                    gl.uniform_1_i32(Some(&l), 0);
                }
                if let Some(l) = Self::uni(gl, self.convert_frag, "u_velocity") {
                    gl.uniform_1_i32(Some(&l), 1);
                }
                Self::set_convert_uniforms(gl, self.convert_frag, p);
                Self::draw_full(gl, self.blit_vao);
                self.timers.end(gl, PassId::Convert);
                gl.use_program(None);
            }

            // ---- Activate (3-pass, compute-only) ----
            let data_for_upscale = if three {
                let prog = self.activate_comp.unwrap();
                self.timers.begin(gl, PassId::Activate);
                gl.use_program(Some(prog));
                Self::bind_tex(gl, 0, self.data_tex);
                Self::bind_tex(gl, 1, self.scene_color);
                Self::bind_tex(gl, 2, self.luma_tex[self.luma_read]);
                for (n, u) in [("u_data", 0), ("u_color", 1), ("u_luma_prev", 2)] {
                    if let Some(l) = Self::uni(gl, prog, n) {
                        gl.uniform_1_i32(Some(&l), u);
                    }
                }
                if let Some(l) = Self::uni(gl, prog, "u_render_size") {
                    gl.uniform_2_f32(Some(&l), p.render_size[0], p.render_size[1]);
                }
                if let Some(l) = Self::uni(gl, prog, "u_render_rcp") {
                    gl.uniform_2_f32(Some(&l), p.render_size_rcp[0], p.render_size_rcp[1]);
                }
                if let Some(l) = Self::uni(gl, prog, "u_reset") {
                    gl.uniform_1_f32(Some(&l), p.reset as f32);
                }
                if let Some(l) = Self::uni(gl, prog, "u_fov_hor") {
                    gl.uniform_1_f32(Some(&l), p.camera_fov_hor);
                }
                let lw = 1 - self.luma_read;
                gl.bind_image_texture(0, Some(self.data2_tex), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.bind_image_texture(1, Some(self.luma_tex[lw]), 0, false, 0, glow::WRITE_ONLY, glow::RG16F);
                gl.dispatch_compute(gx, gy, 1);
                Self::barrier(gl);
                self.timers.end(gl, PassId::Activate);
                gl.use_program(None);
                self.luma_read = lw;
                self.data2_tex
            } else {
                self.data_tex
            };

            // ---- Upscale ----
            let hw = 1 - self.hist_read;
            if self.use_compute {
                let prog = self.upscale_comp.unwrap();
                self.timers.begin(gl, PassId::Upscale);
                gl.use_program(Some(prog));
                Self::bind_tex(gl, 0, self.scene_color);
                Self::bind_tex(gl, 1, data_for_upscale);
                Self::bind_tex(gl, 2, self.hist_tex[self.hist_read]);
                for (n, u) in [("u_color", 0), ("u_data", 1), ("u_history", 2)] {
                    if let Some(l) = Self::uni(gl, prog, n) {
                        gl.uniform_1_i32(Some(&l), u);
                    }
                }
                Self::set_upscale_uniforms(gl, prog, p);
                gl.bind_image_texture(0, Some(self.hist_tex[hw]), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.bind_image_texture(1, Some(self.scene_out_tex), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.dispatch_compute(dx, dy, 1);
                Self::barrier(gl);
                self.timers.end(gl, PassId::Upscale);
                gl.use_program(None);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.hist_fbo[hw]));
                gl.viewport(0, 0, self.dw as i32, self.dh as i32);
                self.timers.begin(gl, PassId::Upscale);
                gl.use_program(Some(self.upscale_frag));
                Self::bind_tex(gl, 0, self.scene_color);
                Self::bind_tex(gl, 1, data_for_upscale);
                Self::bind_tex(gl, 2, self.hist_tex[self.hist_read]);
                for (n, u) in [("u_color", 0), ("u_data", 1), ("u_history", 2)] {
                    if let Some(l) = Self::uni(gl, self.upscale_frag, n) {
                        gl.uniform_1_i32(Some(&l), u);
                    }
                }
                Self::set_upscale_uniforms(gl, self.upscale_frag, p);
                Self::draw_full(gl, self.blit_vao);
                self.timers.end(gl, PassId::Upscale);
                gl.use_program(None);
            }
            self.hist_read = hw;
            self.needs_reset = false;
            self.last_data = Some(data_for_upscale);

            // ---- Sharpen (stage 9, strict post-process) ----
            // Reads the upscale output, writes the new final frame. History
            // keeps the unsharpened pixels by design (see struct docs).
            let pre = self.hist_tex[self.hist_read];
            let use_cs = self.use_compute && self.sharpen_comp.is_some();
            if use_cs {
                let prog = self.sharpen_comp.unwrap();
                self.timers.begin(gl, PassId::Sharp);
                gl.use_program(Some(prog));
                Self::bind_tex(gl, 0, pre);
                if let Some(l) = Self::uni(gl, prog, "u_image") {
                    gl.uniform_1_i32(Some(&l), 0);
                }
                Self::set_sharpen_uniforms(gl, prog, p, self.sharpness);
                gl.bind_image_texture(0, Some(self.sharp_tex), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.dispatch_compute(dx, dy, 1);
                Self::barrier(gl);
                self.timers.end(gl, PassId::Sharp);
                gl.use_program(None);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.sharp_fbo));
                gl.viewport(0, 0, self.dw as i32, self.dh as i32);
                self.timers.begin(gl, PassId::Sharp);
                gl.use_program(Some(self.sharpen_frag));
                Self::bind_tex(gl, 0, pre);
                if let Some(l) = Self::uni(gl, self.sharpen_frag, "u_image") {
                    gl.uniform_1_i32(Some(&l), 0);
                }
                Self::set_sharpen_uniforms(gl, self.sharpen_frag, p, self.sharpness);
                Self::draw_full(gl, self.blit_vao);
                self.timers.end(gl, PassId::Sharp);
                gl.use_program(None);
            }

            self.timers.end(gl, PassId::Total);
            self.sharp_tex
        }
    }

    /// The upscale output BEFORE sharpening (for the B before/after toggle).
    /// Valid after execute(); history-identical (sharpen never feeds back).
    pub fn pre_sharpen_tex(&self) -> glow::NativeTexture {
        self.hist_tex[self.hist_read]
    }

    /// Blit a display-res texture to the window.
    pub unsafe fn blit_to_screen(
        &self,
        gl: &glow::Context,
        tex: glow::NativeTexture,
        win_w: i32,
        win_h: i32,
    ) {
        unsafe {
            self.blit_region(gl, tex, self.blit_prog, 0, 0, win_w, win_h);
        }
    }

    /// Viewport-clipped blit with an explicit program (split-screen views).
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
            Self::bind_tex(gl, 0, tex);
            if let Some(l) = Self::uni(gl, prog, "u_tex") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            // Only the motion debug program declares this; others skip it.
            if let Some(l) = Self::uni(gl, prog, "u_render_size") {
                gl.uniform_2_f32(Some(&l), self.rw as f32, self.rh as f32);
            }
            Self::draw_full(gl, self.blit_vao);
            gl.use_program(None);
        }
    }

    // ---- Stage-6 debug-view accessors ----
    pub fn blit_prog(&self) -> glow::NativeProgram {
        self.blit_prog
    }
    pub fn motion_prog(&self) -> glow::NativeProgram {
        self.motion_prog
    }
    pub fn luma_prog(&self) -> glow::NativeProgram {
        self.luma_prog
    }
    pub fn clip_prog(&self) -> glow::NativeProgram {
        self.clip_prog
    }
    pub fn scene_color_tex(&self) -> glow::NativeTexture {
        self.scene_color
    }
    pub fn data_tex_debug(&self) -> Option<glow::NativeTexture> {
        self.last_data
    }
    pub fn luma_tex_debug(&self) -> glow::NativeTexture {
        self.luma_tex[self.luma_read]
    }
    pub fn display_size(&self) -> (u32, u32) {
        (self.dw, self.dh)
    }
    /// Compute dispatch is usable only if both compute programs compiled.
    pub fn can_compute(&self) -> bool {
        self.convert_comp.is_some() && self.upscale_comp.is_some()
    }
    /// Current per-pass GPU timings overlay fragment (stage 8A).
    pub fn timers_report(&self) -> String {
        self.timers.report()
    }
    /// Scene pass lives outside execute; main wraps it via these when its
    /// rotation slot is armed (stage 10).
    pub fn timer_armed(&self, pass: PassId) -> bool {
        self.timers.timer_armed(pass)
    }
    pub unsafe fn timer_begin(&mut self, gl: &glow::Context, pass: PassId) {
        unsafe { self.timers.timer_begin(gl, pass) }
    }
    pub unsafe fn timer_end(&self, gl: &glow::Context, pass: PassId) {
        unsafe { self.timers.timer_end(gl, pass) }
    }
    pub fn timers_supported(&self) -> bool {
        self.timers.supported()
    }
}
