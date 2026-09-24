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

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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
    // Compute-path scene copy (display res; also feeds stage-6 debug views).
    scene_out_tex: glow::NativeTexture,
    // Programs.
    convert_frag: glow::NativeProgram,
    upscale_frag: glow::NativeProgram,
    blit_prog: glow::NativeProgram,
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

            // Compute programs (prelude + body). Any failure → fragment path.
            let prelude = gl_backend::compute_prelude(gl_major, gl_minor, backend_compute);
            let mut use_compute = backend_compute;
            let mut convert_comp = None;
            let mut upscale_comp = None;
            let mut activate_comp = None;
            if let Some(pre) = prelude {
                let cc = |body: &str| {
                    let src = format!("{pre}{body}");
                    gl_backend::compile_compute(gl, &src).ok()
                };
                convert_comp = cc(igsr_shaders::CONVERT_COMP);
                upscale_comp = cc(igsr_shaders::UPSCALE_COMP);
                activate_comp = cc(igsr_shaders::ACTIVATE_COMP);
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
                scene_out_tex: gl.create_texture().unwrap(),
                convert_frag,
                upscale_frag,
                blit_prog,
                convert_comp,
                upscale_comp,
                activate_comp,
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
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.luma_read = 0;
            self.hist_read = 0;
            self.scene_out_tex =
                tex2d(gl, glow::RGBA16F as i32, dw, dh, glow::RGBA, glow::HALF_FLOAT, no_data(), glow::NEAREST);
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
                gl.use_program(None);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.convert_fbo));
                gl.viewport(0, 0, self.rw as i32, self.rh as i32);
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
                gl.use_program(None);
            }

            // ---- Activate (3-pass, compute-only) ----
            let data_for_upscale = if three {
                let prog = self.activate_comp.unwrap();
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
                let lw = 1 - self.luma_read;
                gl.bind_image_texture(0, Some(self.data2_tex), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.bind_image_texture(1, Some(self.luma_tex[lw]), 0, false, 0, glow::WRITE_ONLY, glow::RGBA16F);
                gl.dispatch_compute(gx, gy, 1);
                Self::barrier(gl);
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
                gl.use_program(None);
            } else {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.hist_fbo[hw]));
                gl.viewport(0, 0, self.dw as i32, self.dh as i32);
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
                gl.use_program(None);
            }
            self.hist_read = hw;
            self.needs_reset = false;
            self.hist_tex[self.hist_read]
        }
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
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, win_w, win_h);
            gl.disable(glow::DEPTH_TEST);
            gl.use_program(Some(self.blit_prog));
            Self::bind_tex(gl, 0, tex);
            if let Some(l) = Self::uni(gl, self.blit_prog, "u_tex") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            Self::draw_full(gl, self.blit_vao);
            gl.use_program(None);
        }
    }
}
