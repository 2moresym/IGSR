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

        unsafe {
            gl.delete_program(prog);
        }
        let _ = (out_tex, self.force_fallback);
        if fail.is_empty() { Ok(log) } else { Err(fail) }
    }
}
