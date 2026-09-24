//! igsr-testbed — standalone test window (stage 1).
//!
//! Stage 1 scope: open one window, render an animated color triangle with
//! plain OpenGL, and print the IGSR core/backend plumbing info. No game, no
//! Vireo/Lake, no external assets. Later stages add: low-res scene render,
//! motion vectors, jittered camera, IGSR passes, UI/debug views.

mod selftest;

use glow::HasContext as _;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::{GlConfig, GlDisplay, GlSurface, NotCurrentGlContext};
use glutin::surface::{SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use std::ffi::CString;
use std::num::NonZeroU32;
use winit::raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::Window;

use igsr::backend::gl::{detect_compute, GlBackend};
use igsr::backend::GpuBackend;

struct App {
    force_fallback: bool,
    selftest: bool,
    selftest_done: bool,
    window: Option<Window>,
    config: Option<glutin::config::Config>,
    surface: Option<glutin::surface::Surface<WindowSurface>>,
    context: Option<PossiblyCurrentContext>,
    gl: Option<glow::Context>,
    gl_version: (u32, u32),
    compute_advertised: bool,
    program: Option<glow::NativeProgram>,
    vao: Option<glow::NativeVertexArray>,
    _vbo: Option<glow::NativeBuffer>,
    start: std::time::Instant,
    frames: u64,
}

impl App {
    fn new(force_fallback: bool, selftest: bool) -> Self {
        Self {
            force_fallback,
            selftest,
            selftest_done: false,
            window: None,
            config: None,
            surface: None,
            context: None,
            gl: None,
            gl_version: (0, 0),
            compute_advertised: false,
            program: None,
            vao: None,
            _vbo: None,
            start: std::time::Instant::now(),
            frames: 0,
        }
    }

    fn init_gl(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_title("IGSR testbed (stage 1: triangle)");
        let template = ConfigTemplateBuilder::new().with_api(glutin::config::Api::OPENGL);

        let (window, config) = DisplayBuilder::new()
            .with_window_attributes(Some(attrs))
            .build(
                event_loop,
                template,
                |configs| {
                    configs
                        .reduce(|best, c| {
                            if c.num_samples() > best.num_samples() {
                                c
                            } else {
                                best
                            }
                        })
                        .expect("no GL configs")
                },
            )
            .expect("DisplayBuilder::build failed");
        let window = window.expect("no window returned");
        let raw_handle = window
            .window_handle()
            .expect("no window handle")
            .as_raw();

        // IGSR passes need GLSL 4.20 (textureGather), so prefer a 4.2 core
        // context and only fall back to 3.3 (triangle-only, no upscaler) if
        // the driver refuses. HD 4000 / crocus does 4.2.
        let mut not_current = None;
        for (maj, min) in [(4u8, 2u8), (3, 3)] {
            let try_attrs = ContextAttributesBuilder::new().with_context_api(
                ContextApi::OpenGl(Some(Version::new(maj, min))),
            );
            // SAFETY: raw handle is live for the life of the window.
            match unsafe {
                config
                    .display()
                    .create_context(&config, &try_attrs.build(Some(raw_handle)))
            } {
                Ok(ctx) => {
                    not_current = Some(ctx);
                    break;
                }
                Err(e) => eprintln!("[testbed] GL {maj}.{min} context failed: {e:?}"),
            }
        }
        let not_current = not_current.expect("no GL context (tried 4.2, 3.3)");
        let surface_attrs = window
            .build_surface_attributes(SurfaceAttributesBuilder::new())
            .expect("build surface attrs failed");
        // SAFETY: surface attrs came from this window/display pair.
        let surface = unsafe {
            config
                .display()
                .create_window_surface(&config, &surface_attrs)
                .expect("create window surface failed")
        };
        let context = not_current.make_current(&surface).expect("make_current failed");
        if let Err(e) = surface.set_swap_interval(
            &context,
            SwapInterval::Wait(NonZeroU32::new(1).unwrap()),
        ) {
            eprintln!("[testbed] swap interval err (non-fatal): {e:?}");
        }

        // Load GL via glow.
        let display = config.display();
        // SAFETY: proc addresses are valid after make_current.
        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                let c = CString::new(s).unwrap();
                display.get_proc_address(&c)
            })
        };

        // Query version/extensions for the compute-fallback decision.
        // NOTE: core profiles removed glGetString(GL_EXTENSIONS); enumerate
        // via glGetStringi instead (glow: get_parameter_indexed_string).
        let version: String = unsafe { gl.get_parameter_string(glow::VERSION) };
        let n_ext: i32 = unsafe { gl.get_parameter_i32(glow::NUM_EXTENSIONS) };
        let mut ext_list = Vec::with_capacity(n_ext.max(0) as usize);
        for i in 0..n_ext.max(0) as u32 {
            ext_list.push(unsafe { gl.get_parameter_indexed_string(glow::EXTENSIONS, i) });
        }
        let extensions = ext_list.join(" ");
        let n_ext = extensions.split_whitespace().count();
        let compute = detect_compute(&version, &extensions);
        self.gl_version = igsr::backend::gl::gl_version(&version);
        self.compute_advertised = compute;
        let backend = if self.force_fallback {
            GlBackend::new(compute).with_forced_fallback()
        } else {
            GlBackend::new(compute)
        };
        eprintln!("[testbed] GL version: {version}");
        eprintln!("[testbed] GL extensions: {n_ext} (compute advertised: {compute})");
        eprintln!(
            "[testbed] backend: {} path={:?}{}",
            backend.name(),
            backend.compute_path(),
            if self.force_fallback { " (forced fallback)" } else { "" }
        );

        // Compile the debug triangle (own shaders, not reference code).
        let program = unsafe { compile_program(&gl, igsr_shaders::TRIANGLE_VERT, igsr_shaders::TRIANGLE_FRAG) };

        // Interleaved triangle: x, y, r, g, b.
        #[rustfmt::skip]
        let verts: [f32; 15] = [
             0.0,  0.6,  1.0, 0.2, 0.2,
            -0.6, -0.5,  0.2, 1.0, 0.2,
             0.6, -0.5,  0.2, 0.2, 1.0,
        ];
        let (vao, vbo) = unsafe {
            let vao = gl.create_vertex_array().expect("vao");
            let vbo = gl.create_buffer().expect("vbo");
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let bytes: &[u8] = std::slice::from_raw_parts(
                verts.as_ptr() as *const u8,
                verts.len() * std::mem::size_of::<f32>(),
            );
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 5 * 4, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 5 * 4, 2 * 4);
            gl.enable_vertex_attrib_array(1);
            gl.bind_vertex_array(None);
            (vao, vbo)
        };

        self.window = Some(window);
        self.config = Some(config);
        self.surface = Some(surface);
        self.context = Some(context);
        self.gl = Some(gl);
        self.program = Some(program);
        self.vao = Some(vao);
        self._vbo = Some(vbo);
    }

    fn draw(&mut self) {
        let (gl, surface, context, window) = match (
            self.gl.as_ref(),
            self.surface.as_ref(),
            self.context.as_ref(),
            self.window.as_ref(),
        ) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return,
        };
        let size = window.inner_size();
        unsafe {
            gl.viewport(0, 0, size.width as i32, size.height as i32);
            // Animated clear color so a static screenshot still proves frames advance.
            let t = self.start.elapsed().as_secs_f32();
            let pulse = 0.5 + 0.5 * (t * 1.5).sin();
            gl.clear_color(0.05 + 0.1 * pulse, 0.07, 0.10 + 0.15 * (1.0 - pulse), 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(self.program);
            gl.bind_vertex_array(self.vao);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
        if let Err(e) = surface.swap_buffers(context) {
            eprintln!("[testbed] swap_buffers err: {e:?}");
        }
        self.frames += 1;
        if self.frames % 600 == 0 {
            let fps = self.frames as f32 / self.start.elapsed().as_secs_f32();
            eprintln!("[testbed] ~{fps:.1} fps over {} frames", self.frames);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.init_gl(event_loop);
        }
        if self.selftest && !self.selftest_done {
            self.selftest_done = true;
            match self.run_selftest() {
                Ok(report) => {
                    for line in report {
                        eprintln!("[selftest] {line}");
                    }
                    eprintln!("[selftest] RESULT: PASS");
                }
                Err(report) => {
                    for line in report {
                        eprintln!("[selftest] {line}");
                    }
                    eprintln!("[selftest] RESULT: FAIL");
                    std::process::exit(1);
                }
            }
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width != 0 && size.height != 0 {
                    if let (Some(surface), Some(context)) = (self.surface.as_ref(), self.context.as_ref()) {
                        surface.resize(
                            context,
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        );
                    }
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

unsafe fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
) -> glow::NativeProgram {
    unsafe {
        let vs = gl.create_shader(glow::VERTEX_SHADER).expect("vs");
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            panic!("vertex compile failed: {}", gl.get_shader_info_log(vs));
        }
        let fs = gl.create_shader(glow::FRAGMENT_SHADER).expect("fs");
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            panic!("fragment compile failed: {}", gl.get_shader_info_log(fs));
        }
        let prog = gl.create_program().expect("program");
        gl.attach_shader(prog, vs);
        gl.attach_shader(prog, fs);
        gl.link_program(prog);
        if !gl.get_program_link_status(prog) {
            panic!("link failed: {}", gl.get_program_info_log(prog));
        }
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        prog
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let has = |s: &str| args.iter().any(|a| a == s);
    let force_fallback = has("--force-fallback") || has("-f");
    let selftest = has("--selftest");

    // ---- IGSR plumbing proof (no GPU needed) ----
    let cfg = igsr::IgsrConfig::from_display(1280, 720, igsr::QualityMode::Balanced);
    let mut ctx = igsr::IgsrContext::new(cfg.clone()).expect("IgsrContext::new");
    eprintln!("[igsr] core={} wrapper={}", igsr::core_version(), igsr::version());
    eprintln!(
        "[igsr] render={}x{} display={}x{} passes={} {}",
        cfg.render_w,
        cfg.render_h,
        cfg.display_w,
        cfg.display_h,
        ctx.pass_count(),
        ctx.upscale_stub()
    );
    for f in 1..=3u64 {
        let j = igsr_sys::calc_jitter(f);
        eprintln!("[igsr] jitter f{f} = ({:.6}, {:.6})", j[0], j[1]);
    }
    eprintln!("[igsr] render-dispatch(8) = {:?}", ctx.render_dispatch(8));
    eprintln!("[igsr] display-dispatch(8) = {:?}", ctx.display_dispatch(8));
    ctx.advance_frame();

    // ---- Open window ----
    let event_loop = EventLoop::new().expect("winit EventLoop::new");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(force_fallback, selftest);
    event_loop.run_app(&mut app).expect("run_app");
}
