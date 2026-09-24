//! Procedural test scene (own code): one spinning cube, one orbiting
//! moon-cube, one static floor — rendered at render resolution with color,
//! per-pixel velocity (current vs previous clip), and linear depth. No
//! external assets. Gives the upscaler real motion vectors to chew on.

use super::mat4::{self, Mat4};
use glow::HasContext as _;

const SCENE_VERT: &str = "#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_col;
uniform mat4 u_mvp;
uniform mat4 u_mvp_prev;
uniform mat4 u_mv;
out vec3 v_col;
out vec4 v_cur;
out vec4 v_prev;
out float v_viewz;
void main() {
    v_col = a_col;
    vec4 p = vec4(a_pos, 1.0);
    v_cur = u_mvp * p;
    v_prev = u_mvp_prev * p;
    v_viewz = (u_mv * p).z;
    gl_Position = v_cur;
}
";

const SCENE_FRAG: &str = "#version 330 core
in vec3 v_col;
in vec4 v_cur;
in vec4 v_prev;
in float v_viewz;
layout(location = 0) out vec4 o_color;
layout(location = 1) out vec4 o_vel;
layout(location = 2) out vec4 o_depth;
uniform float u_far;
void main() {
    o_color = vec4(v_col, 1.0);
    vec2 cur = v_cur.xy / v_cur.w;
    vec2 prv = v_prev.xy / v_prev.w;
    o_vel = vec4(cur - prv, 0.0, 0.0);
    o_depth = vec4(clamp(-v_viewz / u_far, 0.0, 1.0), 0.0, 0.0, 0.0);
}
";

struct Mesh {
    vao: glow::NativeVertexArray,
    _vbo: glow::NativeBuffer,
    _ebo: glow::NativeBuffer,
    count: i32,
}

impl Mesh {
    unsafe fn new(
        gl: &glow::Context,
        verts: &[f32], // x,y,z,r,g,b
        idx: &[u32],
    ) -> Mesh {
        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            let vbo = gl.create_buffer().unwrap();
            let ebo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                f32_bytes(verts),
                glow::STATIC_DRAW,
            );
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                u32_bytes(idx),
                glow::STATIC_DRAW,
            );
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 6 * 4, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 6 * 4, 3 * 4);
            gl.enable_vertex_attrib_array(1);
            gl.bind_vertex_array(None);
            Mesh { vao, _vbo: vbo, _ebo: ebo, count: idx.len() as i32 }
        }
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

pub struct Scene {
    prog: glow::NativeProgram,
    cube: Mesh,
    floor: Mesh,
    // Previous-frame matrices for velocity.
    prev_vp: Mat4,
    prev_models: [Mat4; 3],
    initialized: bool,
}

impl Scene {
    pub unsafe fn new(gl: &glow::Context) -> Result<Scene, String> {
        unsafe {
            let prog = gl_backend_compile(gl)?;
            // Unit cube, per-face colors, centered on origin.
            let c = [
                [1.0, 0.25, 0.25], // +x red
                [0.25, 1.0, 0.25], // -x green
                [0.25, 0.25, 1.0], // +y blue
                [1.0, 1.0, 0.25], // -y yellow
                [1.0, 0.25, 1.0], // +z magenta
                [0.25, 1.0, 1.0], // -z cyan
            ];
            let p = [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ];
            // Faces as quads (a,b,c,d) with one color each.
            let faces: [(usize, usize, usize, usize, usize); 6] = [
                (1, 2, 6, 5, 0), // +x
                (0, 4, 7, 3, 1), // -x
                (3, 7, 6, 2, 2), // +y
                (0, 1, 5, 4, 3), // -y
                (4, 5, 6, 7, 4), // +z
                (0, 3, 2, 1, 5), // -z
            ];
            let mut verts: Vec<f32> = Vec::new();
            let mut idx: Vec<u32> = Vec::new();
            for (a, b, cc, d, ci) in faces {
                let base = (verts.len() / 6) as u32;
                for &vi in &[a, b, cc, d] {
                    verts.extend_from_slice(&[p[vi][0], p[vi][1], p[vi][2], c[ci][0], c[ci][1], c[ci][2]]);
                }
                idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
            let cube = Mesh::new(gl, &verts, &idx);

            // Floor: 10x10 quad at y = -0.8, neutral grey with a subtle tint.
            let fverts: [f32; 24] = [
                -5.0, -0.8, -5.0, 0.32, 0.33, 0.36, //
                5.0, -0.8, -5.0, 0.32, 0.33, 0.36, //
                5.0, -0.8, 5.0, 0.36, 0.37, 0.40, //
                -5.0, -0.8, 5.0, 0.36, 0.37, 0.40,
            ];
            let fidx: [u32; 6] = [0, 1, 2, 0, 2, 3];
            let floor = Mesh::new(gl, &fverts, &fidx);

            Ok(Scene {
                prog,
                cube,
                floor,
                prev_vp: mat4::identity(),
                prev_models: [mat4::identity(), mat4::identity(), mat4::identity()],
                initialized: false,
            })
        }
    }

    /// Render the scene for `angle` into the already-bound scene FBO.
    /// Returns (vp_curr, vp_prev) for the frame's clip_to_prev matrix.
    pub unsafe fn render(
        &mut self,
        gl: &glow::Context,
        angle: f32,
        view: &Mat4,
        proj: &Mat4, // already jittered
        far: f32,
    ) -> (Mat4, Mat4) {
        unsafe {
            let vp = mat4::mul(proj, view);
            // Object models: spinning cube, orbiting moon, static floor.
            let m0 = mat4::mul(
                &mat4::translate(0.0, 0.3, 0.0),
                &mat4::mul(&mat4::rot_y(angle), &mat4::translate(0.0, 0.0, 0.0)),
            );
            let mx = angle.cos() * 1.6;
            let mz = angle.sin() * 1.6;
            let m1 = mat4::mul(
                &mat4::translate(mx, 0.9, mz),
                &mat4::mul(&mat4::rot_y(-angle * 1.7), &mat4::translate(0.0, 0.0, 0.0)),
            );
            // Moon is half-size: bake scale into rotation matrix manually.
            let m1 = scale_mat(&m1, 0.45);
            let m2 = mat4::identity();
            let models = [m0, m1, m2];

            gl.use_program(Some(self.prog));
            let loc = |n: &str| gl.get_uniform_location(self.prog, n);
            if let Some(l) = loc("u_far") {
                gl.uniform_1_f32(Some(&l), far);
            }
            // Explicit (mesh, model) pairs. NOTE (stage 8C): this used to be
            // two parallel arrays cut by one index list, which silently drew
            // the cube mesh with the floor transform and vice versa — the
            // "hero cube" never spun and the floor secretly rotated. It
            // looked right in color because a spinning two-tone quad is
            // nearly indistinguishable from a static floor; only the motion
            // debug view exposed it (diverging velocity field on static
            // geometry). Never index meshes and models separately again.
            let draws = [
                (&self.floor, &models[2], &self.prev_models[2]),
                (&self.cube, &models[0], &self.prev_models[0]),
                (&self.cube, &models[1], &self.prev_models[1]),
            ];
            for (mesh, model, prev_model) in draws {
                let mvp = mat4::mul(&vp, model);
                let mv = mat4::mul(view, model);
                let pmvp = if self.initialized {
                    mat4::mul(&self.prev_vp, prev_model)
                } else {
                    mvp
                };
                if let Some(l) = loc("u_mvp") {
                    gl.uniform_matrix_4_f32_slice(Some(&l), false, &mvp);
                }
                if let Some(l) = loc("u_mvp_prev") {
                    gl.uniform_matrix_4_f32_slice(Some(&l), false, &pmvp);
                }
                if let Some(l) = loc("u_mv") {
                    gl.uniform_matrix_4_f32_slice(Some(&l), false, &mv);
                }
                gl.bind_vertex_array(Some(mesh.vao));
                gl.draw_elements(glow::TRIANGLES, mesh.count, glow::UNSIGNED_INT, 0);
            }
            gl.bind_vertex_array(None);
            gl.use_program(None);

            let vp_prev = if self.initialized { self.prev_vp } else { vp };
            self.prev_vp = vp;
            self.prev_models = models;
            self.initialized = true;
            (vp, vp_prev)
        }
    }
}

fn scale_mat(m: &Mat4, s: f32) -> Mat4 {
    let mut o = *m;
    for c in 0..3 {
        for r in 0..3 {
            o[c * 4 + r] *= s;
        }
    }
    o
}

fn gl_backend_compile(gl: &glow::Context) -> Result<glow::NativeProgram, String> {
    igsr::backend::gl::compile_program(gl, SCENE_VERT, SCENE_FRAG)
}
