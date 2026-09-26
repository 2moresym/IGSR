# IGSR pipeline framework — design proposal (stage 11)

**Status: proposal for review. No code has been touched yet.**

Stage 11 is a pure architecture refactor. Success is defined negatively:
every existing output must be byte-for-byte what it is today. This doc
proposes the abstraction; implementation waits for sign-off.

## 0. The problem being solved

`apps/igsr-testbed/src/pipeline.rs` (953 lines) grew a pass at a time and
now encodes the algorithm's structure in the *orchestration* layer:

- `execute()` is a ~150-line function with four hand-written blocks
  (`// ---- Convert ----`, `Activate`, `Upscale`, `Sharpen`), each with its
  own FBO bind, its own image-unit wiring, its own uniform block, its own
  dispatch-vs-draw branch, and its own timer calls.
- The compute/fragment duality is expressed four times, inline, with
  `if self.use_compute { ...dispatch... } else { ...draw... }` per pass.
  Adding a fifth pass means a fifth copy, and the auto-detect contract
  (stage 8A: one timed query per frame, rotating) has to be respected by
  hand at five sites.
- Persistent state is hardcoded by field: `hist_tex: [T; 2]` +
  `hist_read: usize`, `luma_tex: [T; 2]` + `luma_read: usize`, both
  manipulated inline at their two use sites. Nothing encodes *why* they
  ping-pong or that a third pass might want the same.
- Debug views are accessors naming specific textures
  (`data_tex_debug`, `luma_tex_debug`, `pre_sharpen_tex`) wired in
  `main.rs`'s `ViewMode` match. A new pass cannot expose a debug view
  without edits in three files.
- Uniform uploads: `set_convert_uniforms` / `set_upscale_uniforms` /
  `set_sharpen_uniforms` / an inline block in the activate branch — four
  near-duplicates, each calling `get_uniform_location` by string per
  frame (noted in stage 10 as a µs-scale wart; the refactor is the
  right moment to resolve it, but only because it's *free* here — see
  §6 non-goals caveat).

## 1. Generic `Pass`

```rust
/// One unit of GPU work. Implementors describe resources + dispatch; the
/// framework owns ordering, state binding, timing, and the
/// compute/fragment decision.
pub trait Pass {
    /// Stable identifier, used for timer labels, debug-view lookup, and
    /// resource namespacing. Must be unique in a pipeline.
    fn name(&self) -> &'static str;

    /// Resources this pass reads (sampled) and writes (color targets /
    /// storage images), in declaration order. Binding indices are derived
    /// from this list, not hardcoded per pass.
    fn io(&self) -> PassIo;

    /// Which shader variant to use for each path. A pass that only
    /// implements one variant returns None for the other and the
    /// framework reports it at build time (today: activate is
    /// compute-only; everything else is dual-path).
    fn variants(&self) -> Variants { Variants::Both }

    /// Record resources owned by the pipeline (history, scratch) →
    /// actual GL handles, for this pipeline's current sizes.
    fn declare(&mut self, sizes: Sizes, ctx: &mut ResCtx) { let _ = (sizes, ctx); }

    /// Bind + dispatch. `ctx` carries resolved handles for every declared
    /// resource; the framework has already bound the FBO/image units and
    /// wrapped the call in the timer query.
    fn dispatch(&mut self, gl: &glow::Context, ctx: &mut PassCtx) { let _ = (gl, ctx); }

    /// Advance any cross-frame bookkeeping (ping-pong flip, counters).
    /// Runs after the whole pipeline, once, in order.
    fn end_frame(&mut self) {}
}
```

Key point: `dispatch` never decides compute-vs-fragment, never binds an
FBO, never touches a timer query, never flips history. Those are the
framework's job, derived from `io()` + `variants()`. A pass implements
its shader work and nothing else.

### Declared resources

```rust
pub struct PassIo {
    pub reads:  Vec<ResRef>,   // logical names → resolved handles
    pub writes: Vec<ResRef>,
}
```

`ResRef` is a *logical* name (`"color"`, `"data"`, `"history.prev"`).
Resolution to a GL handle happens once per frame in the framework via the
resource table (§3). This is what lets a future FSR2.2 pass say
`reads: ["color", "data"]` without knowing a texture slot exists.

## 2. `Pipeline`

```rust
pub struct Pipeline {
    passes: Vec<Box<dyn Pass>>,
    resources: ResourceTable,   // ping-pong sets, scratch, static
    timers: PassTimers,          // generalized, see §4
}

impl Pipeline {
    pub fn run(&mut self, gl: &glow::Context, params: &FrameParams) -> Output;
}
```

`run` is the only entry point, and it is a loop over `passes` — not four
hand-written blocks. The default pipeline is constructed as:

```rust
Pipeline::default_igsr(gl, compute_available, three_pass)
// => [Convert::new(), Activate::new(), Upscale::new(), Sharpen::new()]
```

`three_pass = false` omits `Activate` (today: it runs only on the 3-pass
path). That flag becomes a *pipeline construction* decision rather than a
per-frame branch — the framework stops needing the `three_warned` logging
and the runtime "3-pass needs compute" fallback in `execute()`.

`run` resolves the output handle from the last pass that declares a
`"final"` output, so the framework doesn't hardcode "sharpen is last".

## 3. Persistent / ping-ponged state

Two categories, both explicit in the resource table, neither keyed to a
pass name:

```rust
pub enum ResourceKind {
    /// Reallocated on resize. No cross-frame role.
    Scratch { format: TexFormat },
    /// Ping-pong pair. `read` is this frame's input, `write` is this
    /// frame's output; flipped after the pass that declares it.
    PingPong { format: TexFormat, taps: usize },
}
```

`PingPong` is generic over taps (2 today; a 3-deep chain for FSR2.2
interlocks later needs none, but exposure weighting might). The read/write
resolution is framework logic: each frame starts with all pairs at their
current index; a pass that *writes* a pair writes `write` and the
framework flips after dispatch.

Which pairs exist is declared by the passes that need them, at `declare()`
time — so `HistoryColor` today is requested by `Upscale`, and the luma
pair by `Activate`. The framework never says "upscale ping-pongs"; the
pass says "I own a ping-pong pair called `luma`". Adding a third
stateful pass requires no framework change and no knowledge of which of
today's passes are stateful.

Reset semantics (`needs_reset`, resize, camera cut) are a property of the
*frame*, not of a pass: a single `reset` flag on the frame params, read
uniformly by whichever passes care. `Pipeline::resize` recreates all
resources and sets the reset flag; passes don't each need resize logic.

## 4. Timers and debug views — generalized hooks

**Timers.** `PassTimers` today has a hand-maintained `PassId` enum and a
hand-written rotation match. Replace with:

```rust
timers.register(pass.name());          // called once at build
timers.begin_frame();                  // assigns each pass a slot for
                                       //   this frame (one per pass,
                                       //   round-robin by index)
```

The rotation policy (one timed pass per frame — the crocus workaround
from stage 8A) stays as a framework policy, not a per-pass concern. A new
pass is timed automatically the moment it's registered; nothing in the
timing code mentions `Convert` or `Upscale` by name. The report string
builds from the registration order.

**Debug views.** Today: `ViewMode` enum in `main.rs` + texture accessors
+ four small blit programs. Proposal — passes *offer* views, the framework
collects them:

```rust
pub struct DebugView {
    pub label: &'static str,          // "motion", "luma", "clip", ...
    pub source: ResRef,               // logical resource
    pub program: NativeProgram,       // provided by the view itself
}

trait Pass {
    fn debug_views(&self) -> &[DebugView] { &[] }
}
```

`Upscale` offers the motion view (it owns the data buffer), `Activate`
offers clip + luma, the default `None`. `main.rs`'s `ViewMode` becomes a
runtime `Vec<DebugView>` built from the pipeline, so a future FSR2.2
pass exposing "exposure weight" shows up with no `main.rs` change. The
existing five views keep their exact programs and their exact visual
semantics — this changes *how they're found*, not what they draw.

## 5. Migration plan (what actually changes in the code)

1. Extract `Convert`, `Activate`, `Upscale`, `Sharpen` into their own
   files under `apps/igsr-testbed/src/passes/`, each implementing `Pass`.
   Shader handles move inside; `pipeline.rs` shrinks to framework.
2. Move the four `set_*_uniforms` fns into their pass, reading from
   `PassCtx` (params + resolved resources). This is also the moment to
   cache uniform locations in the program struct — free here because the
   location is a property of the program, not of the frame.
3. Replace the four inline dispatch branches with one framework branch
   driven by `variants()`.
4. Replace `hist_read`/`luma_read` fields with `ResourceTable` ping-pong
   indices; keep the "clear on resize" behavior exactly.
5. Replace `PassId` enum with registration; keep the report format string
   identical in content.
6. `main.rs`: `ViewMode` views sourced from the pipeline. Keyboard
   bindings, flags, and output text unchanged.
7. C core (`igsr-core`), the `igsr` crate, the shaders, and all FFI are
   untouched — this is a testbed-orchestration refactor. (A later stage
   can decide whether the framework belongs in the `igsr` crate proper
   for the Vireo path; out of scope here.)

## 6. Non-goals for stage 11 (explicit)

- **No FSR2.2 work** — no disocclusion-domain/lock, no exposure-weight
  state, no RCAS scale integration. These are the follow-ons that get
  *cheaper* once this framework exists; adding any of them now would
  prove nothing about the refactor and would muddy the regression diff.
- **No quality presets** — no Performance/Balanced/Quality switching in
  the pipeline. Presets become pipeline *variants* (different pass
  lists) once the framework supports pass lists; that's a follow-on.
- **No docs/README cleanup** beyond adding this file. No reformatting of
  existing docs; no renaming of existing flags, keys, or output text.
- **No algorithm changes.** The temporal-depth epsilon, the RCAS lobe
  math, the reset semantics, the NDC sign convention — all frozen. If a
  refactor step seems to require touching an algorithm, that's a design
  bug in this proposal, not a licence to change it.

## 7. Verification plan (the regression bar)

Nothing is "done" until all of these, captured **before** the refactor:

1. `cargo test --workspace` — 18 existing tests green, unmodified.
2. `--selftest` — PASS with the same printed lines (readback values are
   printed to 4 decimals; compare strings).
3. `--dump` captures for **every** view (`upscaled`, `native`, `split`,
   `motion`, `luma`, `clip`) at a fixed `--spin`/`--scale`, plus the
   `_pre` (pre-RCAS) pair — pixel-compared against the pre-refactor
   captures. Any nonzero diff is a regression to fix, not to excuse.
   (Animate: dumps are taken at a fixed frame index with a fixed `dt`
   clamp, but `dt` is wall-clock; the existing captures are therefore
   frame-content-dependent. To make the comparison exact, the refactor
   verification run uses a fixed-`dt` mode — or, simpler and already in
   the code path: the diff is asserted on the *static* first-frame
   content at `--spin 0`, which is deterministic.)
4. Same-frame timer shape (which passes report, `--` where a pass doesn't
   run) — report text should be identical in content.

The stage closes when the diff is empty and the tests are green. If the
diff is nonempty I stop and fix the refactor — the algorithm stays frozen.

---

**Decision requested before implementation:**
- Is `Pass` as a trait object the right call, or should passes be an enum
  (closed set) for this stage? Trait objects let a future FSR2.2 pass
  live in a new file with zero edits here; the enum is simpler today.
- Does `PassCtx` carrying resolved resources + params suffice, or does
  anything need direct access to the `ResourceTable` for custom
  multi-tap binding?
