//! Stage-3 GPU selftest (`--selftest`): compiles the reimplemented convert
//! shaders and runs one convert dispatch against procedural inputs on the
//! real driver (crocus here), then checks the readback. No window
//! interaction needed; the harness exits 0/1 with a printed report.

use super::App;
use glow::HasContext as _;
use igsr::backend::gl as gl_backend;

const W: i32 = 64;
const H: i32 = 64;

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 slice reinterpreted as bytes for upload; same size/alignment.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl App {
    pub fn run_selftest(&mut self) -> Result<Vec<String>, Vec<String>> {
        let mut log: Vec<String> = Vec::new();
        let mut fail: Vec<String> = Vec::new();
        let (maj, min) = self.gl_version;
        // Convert needs GLSL 4.20 (textureGather). Below that, IGSR passes
        // cannot run — report instead of producing garbage.
        if maj < 4 {
            fail.push(format!(
                "GL {maj}.{min} < 4.0: convert requires GLSL 4.20 (textureGather)"
            ));
            return Err(fail);
        }
        log.push(format!("GL {maj}.{min} >= 4.0: convert baseline met"));
        let gl = self.gl.as_ref().unwrap();

        // 1. Fragment convert must compile (runs everywhere).
        let prog = match gl_backend::compile_program(
            gl,
            igsr_shaders::FULLSCREEN_VERT,
            igsr_shaders::CONVERT_FRAG,
        ) {
            Ok(p) => {
                log.push("convert.frag compiles: ok".into());
                p
            }
            Err(e) => {
                fail.push(format!("convert.frag compile FAILED: {e}"));
                return Err(fail);
            }
        };

        // 2. Compute convert is a probe: report, don't fail. It dogfoods the
        //    real backend path (compute_prelude + shader body, no #version in
        //    the source file). The fragment path above stays the fallback.
        {
            let (maj, min) = self.gl_version;
            let body = igsr_shaders::CONVERT_COMP;
            let mut candidates: Vec<&str> = Vec::new();
            if let Some(p) = gl_backend::compute_prelude(maj, min, self.compute_advertised) {
                candidates.push(p);
            }
            // Also try the prelude we did NOT select, so the report shows
            // exactly what this driver accepts.
            for alt in [
                "#version 430 core\n",
                "#version 420 core\n#extension GL_ARB_compute_shader : require\n",
            ] {
                if !candidates.contains(&alt) {
                    candidates.push(alt);
                }
            }
            let mut probed = false;
            for prelude in candidates {
                let src = format!("{prelude}{body}");
                let which = prelude.lines().next().unwrap_or("?");
                match gl_backend::compile_compute(gl, &src) {
                    Ok(p) => {
                        log.push(format!("convert.comp probe ({which}...): ok"));
                        unsafe { gl.delete_program(p) };
                        probed = true;
                    }
                    Err(e) => {
                        let first = e.lines().next().unwrap_or("?");
                        log.push(format!("convert.comp probe ({which}...): {first}"));
                    }
                }
            }
            if !probed {
                log.push("convert.comp probe: no compute prelude worked (fragment fallback covers this)".into());
            }
        }

        // 3. Procedural inputs: flat sky depth (1.0), zero velocity (static).
        let (depth_tex, vel_tex, out_tex, fbo, vao) = unsafe {
            let depth_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            let depths = vec![1.0f32; (W * H) as usize];
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::R32F as i32, W, H, 0,
                glow::RED, glow::FLOAT, glow::PixelUnpackData::Slice(Some(f32_bytes(&depths))),
            );

            let vel_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(vel_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            let vels = vec![0.0f32; (W * H * 2) as usize];
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RG32F as i32, W, H, 0,
                glow::RG, glow::FLOAT, glow::PixelUnpackData::Slice(Some(f32_bytes(&vels))),
            );

            let out_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(out_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA16F as i32, W, H, 0,
                glow::RGBA, glow::HALF_FLOAT, glow::PixelUnpackData::Slice(None),
            );

            let fbo = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(out_tex), 0,
            );
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                fail.push(format!("convert FBO incomplete: 0x{status:x}"));
                return Err(fail);
            }

            // Fullscreen triangle with UVs (pos.xy, uv).
            let verts: [f32; 12] = [-1.0, -1.0, 0.0, 0.0, 3.0, -1.0, 2.0, 0.0, -1.0, 3.0, 0.0, 2.0];
            let vao = gl.create_vertex_array().unwrap();
            let vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, f32_bytes(&verts), glow::STATIC_DRAW);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 4 * 4, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 4 * 4, 2 * 4);
            gl.enable_vertex_attrib_array(1);
            (depth_tex, vel_tex, out_tex, fbo, vao)
        };
        log.push("convert FBO (RGBA16F, 64x64): complete".into());

        // 4. Draw with identity reprojection: expect motion=0, disocc=0, depth=1.
        let ident = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        unsafe {
            gl.viewport(0, 0, W, H);
            gl.use_program(Some(prog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(vel_tex));
            let loc = |n| gl.get_uniform_location(prog, n);
            if let Some(l) = loc("u_depth") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            if let Some(l) = loc("u_velocity") {
                gl.uniform_1_i32(Some(&l), 1);
            }
            if let Some(l) = loc("u_render_size") {
                gl.uniform_2_f32(Some(&l), W as f32, H as f32);
            }
            if let Some(l) = loc("u_render_rcp") {
                gl.uniform_2_f32(Some(&l), 1.0 / W as f32, 1.0 / H as f32);
            }
            if let Some(l) = loc("u_clip_to_prev") {
                gl.uniform_matrix_4_f32_slice(Some(&l), false, &ident);
            }
            if let Some(l) = loc("u_fov_hor") {
                gl.uniform_1_f32(Some(&l), 1.0);
            }
            gl.bind_vertex_array(Some(vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
        let err = unsafe { gl.get_error() };
        if err != glow::NO_ERROR {
            fail.push(format!("GL error after convert draw: 0x{err:x}"));
            return Err(fail);
        }
        log.push("convert draw: no GL errors".into());

        // 5. Read back center pixel: expect (0, 0, 0, 1) ± epsilon.
        let mut px = [0u8; 16];
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fbo));
            gl.read_pixels(W / 2, H / 2, 1, 1, glow::RGBA, glow::FLOAT, glow::PixelPackData::Slice(Some(&mut px)));
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        let v: [f32; 4] = [f32::from_le_bytes(px[0..4].try_into().unwrap()),
            f32::from_le_bytes(px[4..8].try_into().unwrap()),
            f32::from_le_bytes(px[8..12].try_into().unwrap()),
            f32::from_le_bytes(px[12..16].try_into().unwrap())];
        if v[0].abs() < 1e-3 && v[1].abs() < 1e-3 && v[2].abs() < 1e-3 && (v[3] - 1.0).abs() < 1e-3 {
            log.push(format!(
                "convert readback center=({:.4}, {:.4}, {:.4}, {:.4}): ok",
                v[0], v[1], v[2], v[3]
            ));
        } else {
            fail.push(format!(
                "convert readback center=({v:?}): expected ~(0,0,0,1)"
            ));
            return Err(fail);
        }

        // 6. Upscale fragment must compile (the HD 4000 default path).
        let uprog = match gl_backend::compile_program(
            gl,
            igsr_shaders::FULLSCREEN_VERT,
            igsr_shaders::UPSCALE_FRAG,
        ) {
            Ok(p) => {
                log.push("upscale.frag compiles: ok".into());
                p
            }
            Err(e) => {
                fail.push(format!("upscale.frag compile FAILED: {e}"));
                return Err(fail);
            }
        };

        // 7. Compute probes (log only). Sharpen included: same prelude rule.
        {
            let (maj, min) = self.gl_version;
            for (name, body) in [
                ("upscale.comp", igsr_shaders::UPSCALE_COMP),
                ("activate.comp", igsr_shaders::ACTIVATE_COMP),
                ("sharpen.comp", igsr_shaders::SHARPEN_COMP),
            ] {
                let mut done = false;
                if let Some(pre) = gl_backend::compute_prelude(maj, min, self.compute_advertised) {
                    let src = format!("{pre}{body}");
                    match gl_backend::compile_compute(gl, &src) {
                        Ok(p) => {
                            log.push(format!("{name} probe (selected prelude): ok"));
                            unsafe { gl.delete_program(p) };
                            done = true;
                        }
                        Err(e) => {
                            let f = e.lines().next().unwrap_or("?");
                            log.push(format!("{name} probe (selected prelude): {f}"));
                        }
                    }
                }
                if !done {
                    log.push(format!("{name} probe: no working prelude (fragment path covers upscale; activate waits for stage 5)"));
                }
            }
        }

        // 8. Full chain: convert output + flat color + empty history at
        //    reset=1 must reproduce the flat color (alpha -> 1, Lanczos of a
        //    constant field is the constant).
        const DW: i32 = 96;
        const DH: i32 = 96;
        let (color_tex, hist_tex, disp_tex, disp_fbo) = unsafe {
            let color_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(color_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            let mut flat: Vec<u8> = Vec::with_capacity((W * H * 4) as usize);
            for _ in 0..W * H {
                flat.extend_from_slice(&[51, 102, 153, 255]); // (0.2,0.4,0.6)
            }
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA8 as i32, W, H, 0,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&flat)),
            );

            let hist_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(hist_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            let grey = vec![0.5f32; (DW * DH * 4) as usize];
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA16F as i32, DW, DH, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(f32_bytes(&grey))),
            );

            let disp_tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(disp_tex));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA16F as i32, DW, DH, 0,
                glow::RGBA, glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(None),
            );
            let disp_fbo = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(disp_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(disp_tex), 0,
            );
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                fail.push("upscale FBO incomplete".into());
                return Err(fail);
            }
            (color_tex, hist_tex, disp_tex, disp_fbo)
        };
        unsafe {
            gl.viewport(0, 0, DW, DH);
            gl.use_program(Some(uprog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(color_tex));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(out_tex)); // convert output
            gl.active_texture(glow::TEXTURE2);
            gl.bind_texture(glow::TEXTURE_2D, Some(hist_tex));
            let loc = |n| gl.get_uniform_location(uprog, n);
            if let Some(l) = loc("u_color") {
                gl.uniform_1_i32(Some(&l), 0);
            }
            if let Some(l) = loc("u_data") {
                gl.uniform_1_i32(Some(&l), 1);
            }
            if let Some(l) = loc("u_history") {
                gl.uniform_1_i32(Some(&l), 2);
            }
            if let Some(l) = loc("u_render_size") {
                gl.uniform_2_f32(Some(&l), W as f32, H as f32);
            }
            if let Some(l) = loc("u_render_rcp") {
                gl.uniform_2_f32(Some(&l), 1.0 / W as f32, 1.0 / H as f32);
            }
            if let Some(l) = loc("u_display_size") {
                gl.uniform_2_f32(Some(&l), DW as f32, DH as f32);
            }
            if let Some(l) = loc("u_display_rcp") {
                gl.uniform_2_f32(Some(&l), 1.0 / DW as f32, 1.0 / DH as f32);
            }
            if let Some(l) = loc("u_jitter") {
                gl.uniform_2_f32(Some(&l), 0.0, 0.0);
            }
            if let Some(l) = loc("u_reset") {
                gl.uniform_1_f32(Some(&l), 1.0);
            }
            if let Some(l) = loc("u_min_lerp") {
                gl.uniform_1_f32(Some(&l), 0.2);
            }
            if let Some(l) = loc("u_full_taps") {
                gl.uniform_1_f32(Some(&l), 0.0);
            }
            gl.bind_vertex_array(Some(vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
        let err = unsafe { gl.get_error() };
        if err != glow::NO_ERROR {
            fail.push(format!("GL error after upscale draw: 0x{err:x}"));
            return Err(fail);
        }
        let mut up = [0u8; 16];
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(disp_fbo));
            gl.read_pixels(DW / 2, DH / 2, 1, 1, glow::RGBA, glow::FLOAT, glow::PixelPackData::Slice(Some(&mut up)));
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        let c = [
            f32::from_le_bytes(up[0..4].try_into().unwrap()),
            f32::from_le_bytes(up[4..8].try_into().unwrap()),
            f32::from_le_bytes(up[8..12].try_into().unwrap()),
        ];
        if (c[0] - 0.2).abs() < 0.02 && (c[1] - 0.4).abs() < 0.02 && (c[2] - 0.6).abs() < 0.02 {
            log.push(format!(
                "upscale chain readback center=({:.4}, {:.4}, {:.4}): ok",
                c[0], c[1], c[2]
            ));
        } else {
            fail.push(format!("upscale chain readback=({c:?}): expected ~(0.2,0.4,0.6)"));
            return Err(fail);
        }

        // 9. Sharpen fragment must compile (post-process, both paths use it
        //    unless compute is available — see probe above).
        let sharp_prog = match gl_backend::compile_program(
            gl,
            igsr_shaders::FULLSCREEN_VERT,
            igsr_shaders::SHARPEN_FRAG,
        ) {
            Ok(p) => {
                log.push("sharpen.frag compiles: ok".into());
                p
            }
            Err(e) => {
                fail.push(format!("sharpen.frag compile FAILED: {e}"));
                return Err(fail);
            }
        };

        // Helper: run sharpen.frag over `src` (DW x DH) at sharpness `sh`
        // into a scratch RGBA16F target, read back pixel (rx, ry).
        let run_sharp = |gl: &glow::Context,
                         sharp_prog: glow::NativeProgram,
                         vao: glow::NativeVertexArray,
                         src: glow::NativeTexture,
                         sh: f32,
                         w: i32,
                         h: i32,
                         rx: i32,
                         ry: i32,
                         scratch_fbo: glow::NativeFramebuffer|
         -> [f32; 4] {
            unsafe {
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(scratch_fbo));
                gl.viewport(0, 0, w, h);
                gl.use_program(Some(sharp_prog));
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(src));
                if let Some(l) = gl.get_uniform_location(sharp_prog, "u_image") {
                    gl.uniform_1_i32(Some(&l), 0);
                }
                if let Some(l) = gl.get_uniform_location(sharp_prog, "u_display_size") {
                    gl.uniform_2_f32(Some(&l), w as f32, h as f32);
                }
                if let Some(l) = gl.get_uniform_location(sharp_prog, "u_display_rcp") {
                    gl.uniform_2_f32(Some(&l), 1.0 / w as f32, 1.0 / h as f32);
                }
                if let Some(l) = gl.get_uniform_location(sharp_prog, "u_sharp") {
                    gl.uniform_1_f32(Some(&l), sh);
                }
                gl.bind_vertex_array(Some(vao));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);
                gl.bind_vertex_array(None);
                gl.use_program(None);
                let mut px = [0u8; 16];
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(scratch_fbo));
                gl.read_pixels(rx, ry, 1, 1, glow::RGBA, glow::FLOAT, glow::PixelPackData::Slice(Some(&mut px)));
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
                [
                    f32::from_le_bytes(px[0..4].try_into().unwrap()),
                    f32::from_le_bytes(px[4..8].try_into().unwrap()),
                    f32::from_le_bytes(px[8..12].try_into().unwrap()),
                    f32::from_le_bytes(px[12..16].try_into().unwrap()),
                ]
            }
        };

        // Scratch target for sharpen runs.
        let (scratch_tex, scratch_fbo) = unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA16F as i32, DW, DH, 0,
                glow::RGBA, glow::HALF_FLOAT,
                glow::PixelUnpackData::Slice(None),
            );
            let f = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(f));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(t), 0);
            assert_eq!(gl.check_framebuffer_status(glow::FRAMEBUFFER), glow::FRAMEBUFFER_COMPLETE);
            (t, f)
        };
        let _ = scratch_tex;

        // 10. sharp=0 over the upscale output must reproduce it bit-closely
        //     (passthrough: lobe 0 -> output = center tap, fp32-exact).
        //     disp_tex holds the step-8 chain result; `c` is its center.
        let s0 = run_sharp(gl, sharp_prog, vao, disp_tex, 0.0, DW, DH, DW / 2, DH / 2, scratch_fbo);
        let same = (s0[0] - c[0]).abs() < 1e-6
            && (s0[1] - c[1]).abs() < 1e-6
            && (s0[2] - c[2]).abs() < 1e-6;
        if same {
            log.push(format!(
                "sharpen(0) passthrough=({:.4}, {:.4}, {:.4}): ok",
                s0[0], s0[1], s0[2]
            ));
        } else {
            fail.push(format!("sharpen(0) passthrough=({s0:?}): expected ({c:?})"));
            return Err(fail);
        }

        // 11b. The step field is DW x DH directly: vertical edge at
        //     x = DW/2 (0.2 | 0.8 grey), black at (0,0), white at corner.
        let step_tex = unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            let mut px = vec![0u8; (DW * DH * 4) as usize];
            for y in 0..DH {
                for x in 0..DW {
                    let i = ((y * DW + x) * 4) as usize;
                    let g = if x < DW / 2 { 51u8 } else { 204u8 }; // 0.2 / 0.8
                    px[i..i + 4].copy_from_slice(&[g, g, g, 255]);
                }
            }
            px[0..4].copy_from_slice(&[0, 0, 0, 255]);
            let last = (((DH - 1) * DW + (DW - 1)) * 4) as usize;
            px[last..last + 4].copy_from_slice(&[255, 255, 255, 255]);
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA8 as i32, DW, DH, 0,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&px)),
            );
            t
        };
        let lx = DW / 2 - 1;
        let rx = DW / 2;
        let ey = DH / 2;
        let pre_l = run_sharp(gl, sharp_prog, vao, step_tex, 0.0, DW, DH, lx, ey, scratch_fbo);
        let pre_r = run_sharp(gl, sharp_prog, vao, step_tex, 0.0, DW, DH, rx, ey, scratch_fbo);
        let post_l = run_sharp(gl, sharp_prog, vao, step_tex, 1.0, DW, DH, lx, ey, scratch_fbo);
        let post_r = run_sharp(gl, sharp_prog, vao, step_tex, 1.0, DW, DH, rx, ey, scratch_fbo);
        let pre_gap = (pre_r[0] - pre_l[0]).abs();
        let post_gap = (post_r[0] - post_l[0]).abs();
        // sharp=0 reproduces the step exactly (sanity on the harness).
        let step_ok = (pre_l[0] - 0.2).abs() < 0.01 && (pre_r[0] - 0.8).abs() < 0.01;
        // sharp=1 must widen the edge: dark side darker, bright side brighter.
        let boost_ok = post_gap > pre_gap + 0.02 && post_l[0] < pre_l[0] && post_r[0] > pre_r[0];
        if step_ok && boost_ok {
            log.push(format!(
                "sharpen(1) edge gap {pre_gap:.3} -> {post_gap:.3}: ok"
            ));
        } else {
            fail.push(format!(
                "sharpen edge: pre=({pre_l:?},{pre_r:?}) post=({post_l:?},{post_r:?})"
            ));
            return Err(fail);
        }
        // 11c. Pure black / pure white survive (0/0 limiter NaN tolerance).
        let blk = run_sharp(gl, sharp_prog, vao, step_tex, 1.0, DW, DH, 0, 0, scratch_fbo);
        let wht = run_sharp(gl, sharp_prog, vao, step_tex, 1.0, DW, DH, DW - 1, DH - 1, scratch_fbo);
        if blk[0].abs() < 1e-3 && blk[1].abs() < 1e-3 && blk[2].abs() < 1e-3
            && (wht[0] - 1.0).abs() < 1e-3 && (wht[1] - 1.0).abs() < 1e-3 && (wht[2] - 1.0).abs() < 1e-3
        {
            log.push("sharpen(1) black->black white->white: ok".into());
        } else {
            fail.push(format!("sharpen extremes: black={blk:?} white={wht:?}"));
            return Err(fail);
        }

        unsafe {
            gl.delete_program(prog);
            gl.delete_program(uprog);
            gl.delete_program(sharp_prog);
        }
        let _ = (out_tex, self.force_fallback);
        if fail.is_empty() { Ok(log) } else { Err(fail) }
    }
}
