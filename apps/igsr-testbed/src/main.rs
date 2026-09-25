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
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::Window;

use igsr::backend::gl::{detect_compute, GlBackend};
use igsr::backend::GpuBackend;

/// Timer queries are core since GL 3.3; on older/odd drivers require the
/// extension string. Pure helper so the gating rule is obvious.
fn timer_supported(extensions: &str) -> bool {
    extensions.split_whitespace().any(|e| e == "GL_ARB_timer_query")
}

/// Stage-6 view modes (key V cycles; M/H jump directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Upscaled,
    Native,
    Split,
    Motion,
    Luma,
    Clip,
}

impl ViewMode {
    fn next(self) -> ViewMode {
        match self {
            ViewMode::Upscaled => ViewMode::Native,
            ViewMode::Native => ViewMode::Split,
            ViewMode::Split => ViewMode::Motion,
            ViewMode::Motion => ViewMode::Luma,
            ViewMode::Luma => ViewMode::Clip,
            ViewMode::Clip => ViewMode::Upscaled,
        }
    }
    fn name(self) -> &'static str {
        match self {
            ViewMode::Upscaled => "upscaled",
            ViewMode::Native => "native scene",
            ViewMode::Split => "split native|upscaled",
            ViewMode::Motion => "motion/disocc debug",
            ViewMode::Luma => "luma-history debug",
            ViewMode::Clip => "activate clip/edge debug",
        }
    }
}

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
    view: ViewMode,
    debug_events: bool,
    show_gpu: bool,
    show_pre: bool,
    init_sharp: Option<f32>,
    scale: f32,
    spin: f32,
    angle: f32,
    last_frame: std::time::Instant,
    same_camera: u32,
    start: std::time::Instant,
    frames: u64,
}

impl App {
    fn new(
        force_fallback: bool,
        three_pass: bool,
        selftest: bool,
        dump_path: Option<String>,
        debug_events: bool,
    ) -> Self {        let now = std::time::Instant::now();
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
            view: ViewMode::Upscaled,
            debug_events,
            show_gpu: true,
            show_pre: false,
            init_sharp: None,
            scale: 0.5,
            spin: 0.5,
            angle: 0.0,
            last_frame: now,
            same_camera: 0,
            start: now,
            frames: 0,
        }
    }

    fn init_gl(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_title("IGSR testbed");
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
        let mut pipe = unsafe {
            pipeline::Pipeline::new(
                &gl,
                maj,
                min,
                backend.supports_compute(),
                self.three_pass,
                timer_supported(&extensions),
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
        eprintln!("[testbed] keys: V cycle view | M motion | H luma-history | C clip/edge | B pre/post-RCAS | Z/X sharpness | +/- scale | R reset | F compute/frag | T 2/3-pass | G gpu times");
        eprintln!(
            "[testbed] timer queries: {}",
            if pipe.timers_supported() { "supported" } else { "UNSUPPORTED (overlay shows placeholder)" }
        );
        self.backend_compute = backend.supports_compute();
        if let Some(sh) = self.init_sharp {
            pipe.sharpness = sh;
        }

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
        self.angle += dt * self.spin;

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
            // MRT attachments need their own clears: velocity must be exact
            // zero (static) and linear depth must start at far (1.0). The
            // gl.clear above only sets attachment 0 correctly.
            gl.clear_buffer_f32_slice(glow::COLOR, 1, &[0.0, 0.0, 0.0, 0.0]);
            gl.clear_buffer_f32_slice(glow::COLOR, 2, &[1.0, 0.0, 0.0, 0.0]);
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
        // Stage-6 views.
        let ww = size.width as i32;
        let wh = size.height as i32;
        unsafe {
            match self.view {
                ViewMode::Upscaled => {
                    let tex = if self.show_pre { pipe.pre_sharpen_tex() } else { out_tex };
                    pipe.blit_to_screen(gl, tex, ww, wh)
                }
                ViewMode::Native => {
                    pipe.blit_to_screen(gl, pipe.scene_color_tex(), ww, wh)
                }
                ViewMode::Split => {
                    pipe.blit_region(gl, pipe.scene_color_tex(), pipe.blit_prog(), 0, 0, ww / 2, wh);
                    pipe.blit_region(gl, out_tex, pipe.blit_prog(), ww / 2, 0, ww - ww / 2, wh);
                }
                ViewMode::Motion => {
                    if let Some(d) = pipe.data_tex_debug() {
                        pipe.blit_region(gl, d, pipe.motion_prog(), 0, 0, ww, wh);
                    } else {
                        pipe.blit_to_screen(gl, out_tex, ww, wh);
                    }
                }
                ViewMode::Luma => {
                    pipe.blit_region(gl, pipe.luma_tex_debug(), pipe.luma_prog(), 0, 0, ww, wh);
                }
                ViewMode::Clip => {
                    if let Some(d) = pipe.data_tex_debug() {
                        pipe.blit_region(gl, d, pipe.clip_prog(), 0, 0, ww, wh);
                    } else {
                        pipe.blit_to_screen(gl, out_tex, ww, wh);
                    }
                }
            }
        }

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
                    // Aligned before/after pair: blit the pre-RCAS frame and
                    // dump it alongside, so sharpening is comparable without
                    // a second run (angle would differ).
                    let pre_path = path.replace(".ppm", "_pre.ppm");
                    pipe.blit_to_screen(gl, pipe.pre_sharpen_tex(), w, h);
                    let mut px2 = vec![0u8; (w * h * 4) as usize];
                    gl.read_pixels(
                        0, 0, w, h,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(&mut px2)),
                    );
                    let mut ppm2 = format!("P6\n{w} {h}\n255\n").into_bytes();
                    for y in (0..h).rev() {
                        for x in 0..w {
                            let i = ((y * w + x) * 4) as usize;
                            ppm2.extend_from_slice(&px2[i..i + 3]);
                        }
                    }
                    std::fs::write(&pre_path, &ppm2).expect("dump write");
                    eprintln!("[testbed] dumped pre-RCAS frame to {pre_path}");
                }
            }
        }

        if let Err(e) = surface.swap_buffers(context) {
            eprintln!("[testbed] swap_buffers err: {e:?}");
        }
        self.frames += 1;
        if self.frames % 600 == 0 {
            let fps = self.frames as f32 / self.start.elapsed().as_secs_f32();
            let gpu = if self.show_gpu {
                pipe.timers_report()
            } else {
                String::new()
            };
            eprintln!("[testbed] ~{fps:.1} fps over {} frames ({}x{} -> {}x{}) {}", self.frames, rw, rh, dw, dh, gpu);
        }
    }

    fn print_status(&self) {
        let (rw, rh) = self.pipeline.as_ref().map(|p| p.render_size()).unwrap_or((0, 0));
        let (dw, dh) = self.pipeline.as_ref().map(|p| p.display_size()).unwrap_or((0, 0));
        let path = self
            .pipeline
            .as_ref()
            .map(|p| if p.use_compute { "compute" } else { "fragment" })
            .unwrap_or("?");
        let passes = if self.pipeline.as_ref().map(|p| p.three_pass && p.use_compute).unwrap_or(false) {
            3
        } else {
            2
        };
        let gpu = if self.show_gpu {
            self.pipeline.as_ref().map(|p| p.timers_report()).unwrap_or_default()
        } else {
            String::new()
        };
        eprintln!(
            "[testbed] view={} scale={:.2} sharp={:.2}{} ({}x{}->{}x{}) path={} passes={} {}",
            self.view.name(),
            self.scale,
            self.pipeline.as_ref().map(|p| p.sharpness).unwrap_or(0.0),
            if self.show_pre { " pre" } else { "" },
            rw,
            rh,
            dw,
            dh,
            path,
            passes,
            gpu
        );
    }

    fn handle_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::KeyV => {
                self.view = self.view.next();
                self.print_status();
            }
            KeyCode::KeyM => {
                self.view = ViewMode::Motion;
                self.print_status();
            }
            KeyCode::KeyH => {
                self.view = ViewMode::Luma;
                self.print_status();
            }
            KeyCode::KeyC => {
                self.view = ViewMode::Clip;
                self.print_status();
            }
            KeyCode::Equal | KeyCode::NumpadAdd => {
                self.scale = (self.scale + 0.05).min(1.0);
                self.print_status();
            }
            KeyCode::Minus | KeyCode::NumpadSubtract => {
                self.scale = (self.scale - 0.05).max(0.25);
                self.print_status();
            }
            KeyCode::KeyR => {
                if let Some(p) = self.pipeline.as_mut() {
                    p.needs_reset = true;
                }
                self.same_camera = 0;
                eprintln!("[testbed] history reset (camera-cut path)");
            }
            KeyCode::KeyF => {
                if let Some(p) = self.pipeline.as_mut() {
                    if p.use_compute {
                        p.use_compute = false;
                        eprintln!("[testbed] forced fragment path");
                    } else if p.can_compute() {
                        p.use_compute = true;
                        p.needs_reset = true;
                        self.same_camera = 0;
                        eprintln!("[testbed] compute path");
                    } else {
                        eprintln!("[testbed] compute programs unavailable; staying fragment");
                    }
                }
                self.print_status();
            }
            KeyCode::KeyT => {
                if let Some(p) = self.pipeline.as_mut() {
                    p.three_pass = !p.three_pass;
                    p.needs_reset = true;
                    self.same_camera = 0;
                    eprintln!("[testbed] three_pass={}", p.three_pass);
                }
                self.print_status();
            }
            KeyCode::KeyG => {
                self.show_gpu = !self.show_gpu;
                self.print_status();
            }
            KeyCode::KeyB => {
                self.show_pre = !self.show_pre;
                eprintln!(
                    "[testbed] showing {}",
                    if self.show_pre { "pre-RCAS" } else { "post-RCAS" }
                );
            }
            KeyCode::KeyZ => {
                if let Some(p) = self.pipeline.as_mut() {
                    p.sharpness = (p.sharpness - 0.1).max(0.0);
                }
                self.print_status();
            }
            KeyCode::KeyX => {
                if let Some(p) = self.pipeline.as_mut() {
                    p.sharpness = (p.sharpness + 0.1).min(1.0);
                }
                self.print_status();
            }
            _ => {}
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
        if self.debug_events {
            eprintln!("[events] {event:?}");
        }
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
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        self.handle_key(code);
                    }
                }
            }
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
    let debug_events = has("--debug-events");
    let dump_path = args
        .iter()
        .position(|a| a == "--dump")
        .and_then(|i| args.get(i + 1).cloned());
    let init_view = args
        .iter()
        .position(|a| a == "--view")
        .and_then(|i| args.get(i + 1).cloned())
        .and_then(|v| match v.as_str() {
            "native" => Some(ViewMode::Native),
            "split" => Some(ViewMode::Split),
            "motion" => Some(ViewMode::Motion),
            "luma" => Some(ViewMode::Luma),
            "clip" => Some(ViewMode::Clip),
            _ => Some(ViewMode::Upscaled),
        })
        .unwrap_or(ViewMode::Upscaled);

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
    let mut app = App::new(force_fallback, three_pass, selftest, dump_path, debug_events);
    app.view = init_view;
    if let Some(sp) = args
        .iter()
        .position(|a| a == "--spin")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
    {
        app.spin = sp;
    }
    if let Some(sh) = args
        .iter()
        .position(|a| a == "--sharp")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
    {
        app.init_sharp = Some(sh.clamp(0.0, 1.0));
    }
    if let Some(s) = args
        .iter()
        .position(|a| a == "--scale")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
    {
        app.scale = s.clamp(0.25, 1.0);
    }
    event_loop.run_app(&mut app).expect("run_app");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headless_app() -> App {
        App::new(false, false, false, None, false)
    }

    #[test]
    fn view_cycles_all_modes() {
        let mut a = headless_app();
        let mut seen = vec![a.view];
        for _ in 0..5 {
            a.handle_key(KeyCode::KeyV);
            seen.push(a.view);
        }
        assert_eq!(
            seen,
            vec![
                ViewMode::Upscaled,
                ViewMode::Native,
                ViewMode::Split,
                ViewMode::Motion,
                ViewMode::Luma,
                ViewMode::Clip,
            ]
        );
        a.handle_key(KeyCode::KeyV);
        assert_eq!(a.view, ViewMode::Upscaled);
    }

    #[test]
    fn direct_view_keys() {
        let mut a = headless_app();
        a.handle_key(KeyCode::KeyM);
        assert_eq!(a.view, ViewMode::Motion);
        a.handle_key(KeyCode::KeyH);
        assert_eq!(a.view, ViewMode::Luma);
        a.handle_key(KeyCode::KeyC);
        assert_eq!(a.view, ViewMode::Clip);
    }

    #[test]
    fn scale_clamps() {
        let mut a = headless_app();
        for _ in 0..20 {
            a.handle_key(KeyCode::Equal);
        }
        assert!((a.scale - 1.0).abs() < 1e-6);
        for _ in 0..30 {
            a.handle_key(KeyCode::Minus);
        }
        assert!((a.scale - 0.25).abs() < 1e-6);
        a.handle_key(KeyCode::Equal);
        assert!(a.scale > 0.25);
    }

    #[test]
    fn toggles_safe_without_pipeline() {
        // No GL context here: every arm must degrade gracefully.
        let mut a = headless_app();
        a.handle_key(KeyCode::KeyR);
        a.handle_key(KeyCode::KeyF);
        a.handle_key(KeyCode::KeyT);
        a.handle_key(KeyCode::KeyG);
        assert!(!a.show_gpu);
        a.handle_key(KeyCode::KeyG);
        assert!(a.show_gpu);
        a.handle_key(KeyCode::KeyB);
        assert!(a.show_pre);
        a.handle_key(KeyCode::KeyX);
        a.handle_key(KeyCode::KeyX);
        a.handle_key(KeyCode::KeyZ);
        a.handle_key(KeyCode::KeyX); // unbound: no-op
        assert_eq!(a.same_camera, 0);
    }
}
