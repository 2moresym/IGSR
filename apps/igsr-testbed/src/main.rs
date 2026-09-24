//! igsr-testbed — standalone test window (stage 5).
//!
//! Procedural animated 3D scene (spinning cube + moon + floor) rendered at
//! a lower internal resolution with a Halton-jittered camera, fed through
//! the live IGSR chain (convert → [activate] → upscale, fragment or compute
//! per backend) with history ping-pong, and blitted to the window. No game,
//! no Vireo/Lake, no external assets.

mod mat4;
mod pipeline;
mod scene;
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
    three_pass: bool,
    selftest: bool,
    selftest_done: bool,
    window: Option<Window>,
    config: Option<glutin::config::Config>,
    surface: Option<glutin::surface::Surface<WindowSurface>>,
    context: Option<PossiblyCurrentContext>,
    gl: Option<glow::Context>,
    gl_version: (u32, u32),
    compute_advertised: bool,
    backend_compute: bool,
    // Live pipeline state (stage 5).
    ctx: Option<igsr::IgsrContext>,
    scene: Option<scene::Scene>,
    pipeline: Option<pipeline::Pipeline>,
    dump_path: Option<String>,
    scale: f32,
    angle: f32,
    last_frame: std::time::Instant,
    same_camera: u32,
    start: std::time::Instant,
    frames: u64,
}

impl App {
    fn new(force_fallback: bool, three_pass: bool, selftest: bool, dump_path: Option<String>) -> Self {
        let now = std::time::Instant::now();
        Self {
            force_fallback,
            three_pass,
            selftest,
            selftest_done: false,
            window: None,
            config: None,
            surface: None,
            context: None,
            gl: None,
            gl_version: (0, 0),
            compute_advertised: false,
            backend_compute: false,
            ctx: None,
            scene: None,
            pipeline: None,
            dump_path,
            scale: 0.5,
            angle: 0.0,
            last_frame: now,
            same_camera: 0,
            start: now,
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

        // Build the live chain: IGSR context sized to the window, pipeline
        // FBOs, and the procedural scene. Failures here are fatal: without
        // the chain there is nothing to display.
        let win_size = window.inner_size();
        let dw = win_size.width.max(320);
        let dh = win_size.height.max(200);
        let rw = ((dw as f32 * self.scale) as u32).max(8);
        let rh = ((dh as f32 * self.scale) as u32).max(8);
        let cfg = igsr::IgsrConfig::new(rw, rh, dw, dh);
        let ictx = igsr::IgsrContext::new(cfg).expect("IgsrContext::new");
        let (maj, min) = self.gl_version;
        let pipe = unsafe {
            pipeline::Pipeline::new(
                &gl,
                maj,
                min,
                backend.supports_compute(),
                self.three_pass,
                rw,
                rh,
                dw,
                dh,
            )
            .expect("pipeline init")
        };
        let scn = unsafe { scene::Scene::new(&gl).expect("scene init") };
        eprintln!(
            "[testbed] chain: render={}x{} display={}x{} compute={} three_pass={}",
            rw,
            rh,
            dw,
            dh,
            pipe.use_compute,
            self.three_pass && pipe.use_compute,
        );
        self.backend_compute = backend.supports_compute();

        self.window = Some(window);
        self.config = Some(config);
        self.surface = Some(surface);
        self.context = Some(context);
        self.gl = Some(gl);
        self.ctx = Some(ictx);
        self.scene = Some(scn);
        self.pipeline = Some(pipe);
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
        if self.ctx.is_none() || self.scene.is_none() || self.pipeline.is_none() {
            return;
        }
        let size = window.inner_size();
        let dw = size.width.max(320);
        let dh = size.height.max(200);
        let rw = ((dw as f32 * self.scale) as u32).max(8);
        let rh = ((dh as f32 * self.scale) as u32).max(8);

        let dt = self.last_frame.elapsed().as_secs_f32().min(0.1);
        self.last_frame = std::time::Instant::now();
        self.angle += dt * 0.5;

        let ctx = self.ctx.as_mut().unwrap();
        let pipe = self.pipeline.as_mut().unwrap();
        let scn = self.scene.as_mut().unwrap();

        // Window resize → resize the chain (history reset, like a cut).
        if pipe.render_size() != (rw, rh) {
            unsafe { pipe.resize(gl, rw, rh, dw, dh) };
            ctx.resize(rw, rh, dw, dh).expect("ctx resize");
            self.same_camera = 0;
        }

        // Jittered camera: Halton offset from the C core, standard
        // projection-matrix shift at render resolution.
        let jitter = ctx.jitter();
        let aspect = rw as f32 / rh as f32;
        let fov_v = 1.0472; // 60 deg
        let (near, far) = (0.1, 50.0);
        let view = mat4::look_at([0.0, 1.2, 4.5], [0.0, 0.3, 0.0], [0.0, 1.0, 0.0]);
        let mut proj = mat4::perspective(fov_v, aspect, near, far);
        mat4::apply_jitter(&mut proj, jitter[0], jitter[1], rw as f32, rh as f32);

        // Scene pass at render res (color + velocity + linear depth).
        let (vp, vp_prev) = unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(pipe.scene_fbo()));
            gl.viewport(0, 0, rw as i32, rh as i32);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);
            gl.clear_color(0.04, 0.05, 0.08, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            scn.render(gl, self.angle, &view, &proj, far)
        };

        // Frame uniforms: single source of truth via the C core.
        let reset = pipe.needs_reset;
        let clip_col = mat4::mul(&vp_prev, &mat4::inverse(&vp));
        let inputs = igsr::FrameInputs {
            jitter,
            clip_to_prev: mat4::transpose(&clip_col),
            pre_exposure: 1.0,
            camera_fov_hor: (fov_v * 0.5).tan() * aspect,
            camera_near: near,
            min_lerp: 0.2,
            same_camera_frames: self.same_camera,
            reset,
        };
        let params = ctx.frame_params(&inputs);
        let out_tex = unsafe { pipe.execute(gl, &params) };
        unsafe { pipe.blit_to_screen(gl, out_tex, size.width as i32, size.height as i32) };

        ctx.advance_frame();
        self.same_camera = if reset { 0 } else { self.same_camera + 1 };

        // One-shot framebuffer dump for visual verification (stage 5+).
        if let Some(path) = self.dump_path.clone() {
            if self.frames == 120 {
                unsafe {
                    let w = size.width as i32;
                    let h = size.height as i32;
                    let mut px = vec![0u8; (w * h * 4) as usize];
                    gl.read_pixels(
                        0, 0, w, h,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut px)),
                    );
                    // PPM (P6), flipped to top-down row order.
                    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                    for y in (0..h).rev() {
                        for x in 0..w {
                            let i = ((y * w + x) * 4) as usize;
                            ppm.extend_from_slice(&px[i..i + 3]);
                        }
                    }
                    std::fs::write(&path, &ppm).expect("dump write");
                    eprintln!("[testbed] dumped frame {} to {path}", self.frames);
                }
            }
        }

        if let Err(e) = surface.swap_buffers(context) {
            eprintln!("[testbed] swap_buffers err: {e:?}");
        }
        self.frames += 1;
        if self.frames % 600 == 0 {
            let fps = self.frames as f32 / self.start.elapsed().as_secs_f32();
            eprintln!("[testbed] ~{fps:.1} fps over {} frames ({}x{} -> {}x{})", self.frames, rw, rh, dw, dh);
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let has = |s: &str| args.iter().any(|a| a == s);
    let force_fallback = has("--force-fallback") || has("-f");
    let three_pass = has("--three-pass");
    let selftest = has("--selftest");
    let dump_path = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|i| args.get(i + 1).cloned());

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
    let mut app = App::new(force_fallback, three_pass, selftest, dump_path);
    event_loop.run_app(&mut app).expect("run_app");
}
