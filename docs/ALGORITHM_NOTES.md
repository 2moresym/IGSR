# IGSR — Algorithm notes (stage 2)

Study notes from reading the SGSR2 v2 reference shaders in
`reference/sgsr2/include/` (`glsl_2_pass_cs`, `glsl_2_pass_fs`,
`glsl_3_pass_cs`). Everything below is a behavioral description in my own
words — no reference code is copied, translated, or linked into IGSR. Our
reimplementation lives in `crates/igsr-shaders/` and only fills the *roles*
identified here, with our own math, naming, and comments.

Files read (8 total): 2-pass convert+upscale in compute and fragment form,
a shared fullscreen vertex shader, and 3-pass convert+activate+upscale in
compute form. All reference shaders are `#version 320 es` (GLES 3.2 style,
Vulkan-flavored bindings); our GL backend targets desktop GL 3.3+/4.2 on
Mesa crocus, so our shaders are written separately against that.

## 0. Correction to the stage-0 guess

The inferred pipeline in ARCHITECTURE.md ("reconstruct + upsample +
sharpen", with a sharpen pass and an "edge direction" pass) does **not**
match v2. What v2 actually is:

- There is **no sharpen pass** anywhere. Neither the 2-pass nor the 3-pass
  variant sharpens; the upscale pass outputs the clamped/temporally blended
  color directly. If we want sharpening later, it is our own addition, not
  part of the studied algorithm.
- There is **no edge-direction reconstruction pass** in v2. The pass called
  "Activate" (3-pass only) is a luma-delta + depth-disocclusion test, not an
  edge-direction filter.
- The real pass list is: **Convert** (render-res) → [**Activate**]
  (render-res, 3-pass only) → **Upscale** (display-res). "Convert" does
  depth dilation, motion derivation, and color packing; "Upscale" does the
  Lanczos upsample, history clamp, and temporal blend.
- The CS and FS 2-pass variants implement the **same algorithm** with
  different intermediates (see §3): packed YCoCg + dual outputs in CS,
  direct RGB + single output in FS.

So our pass mapping is: `pass1 convert`, `pass2 upscale` (two variants:
compute + fragment fallback), and the 3-pass `activate` folded in as an
optional quality stage — not a sharpen stage. `pass3_sharpen.c` will be
renamed/repurposed accordingly in stage 4.

## 1. Shared concepts

- **Jitter.** The uniform block documents `jitterOffset` in [-0.5, 0.5]
  (pixel units at render res). The app jitters the camera/projection each
  frame; every pass compensates by shifting its render-res lookup by that
  offset. Our `igsr_calc_jitter` (Halton 2,3, minus 0.5) already matches
  this contract.
- **Velocity encoding.** Motion vectors arrive encoded so that "no motion"
  is exactly representable: encoded = velocity * ~0.25 + 32767/65535, i.e.
  an all-zero velocity encodes to a distinctive mid-grey value. Decoding
  inverts that scale/bias. A zero x-channel means "static pixel": the
  shader ignores the velocity texture there and reprojects using depth +
  the `clipToPrevClip` matrix (prevVP * invCurrVP) instead. Consequence
  for us: static geometry needs no velocity writes at all; only dynamic
  objects must output vectors.
- **Depth convention.** Nearest-depth-wins (`min` over gathers; flip to
  `max` under reverse-Z), skybox/depth==1.0 skips disocclusion work.
- **NDC Y handling.** A compile-time switch flips how motion maps to
  history UVs (`PrevUV = ±0.5 * motion + Hruv`), because Vulkan NDC Y is
  inverted vs GL. Our GL backend uses the non-flipped form.
- **Reset / camera-cut.** A `reset` flag forces full weight on the current
  frame (alpha → 1). A same-camera indicator gates tap count and kernel
  bias (see §4).
- **Exposure.** `preExposure` (prev/curr ratio) normalizes HDR color before
  packing and inverts the tonemap after blending, so history stays in a
  stable range across exposure changes.

## 2. Convert pass (render resolution)

Role: turn app-provided color/depth/velocity into GPU-friendly
intermediates. Runs once per render-res pixel.

What it reads:

- `InputColor` (render-res RGBA), `InputDepth`, `InputVelocity`.
- 3-pass only: extra `InputOpaqueColor` (scene color before transparents).

What it computes:

1. **Dilated (nearest) depth.** Four `textureGather` taps over the
   neighborhood reduce to the minimum depth (~3x3 footprint). This
   dilates foreground depth so thin geometry still claims its pixels'
   motion — the standard FSR2-style disocclusion trick.
2. **Depth-clip factor** (2-pass) or raw depth + **alpha mask** (3-pass):
   - 2-pass: a separation metric (constant × horizontal-FOV factor ×
     render-target diagonal, scaled by 1 − depth) weights how much each
     of the four quadrant depths agrees with the dilated depth; strong
     disagreement → depthclip → 1 (disoccluded, trust history less).
   - 3-pass convert skips depthclip (deferred to Activate) and instead
     stores nearest depth plus a transparency mask = scaled magnitude of
     (InputColor − InputOpaqueColor).
3. **Motion.** Decoded from the velocity texture when present, else
   depth-reprojected through `clipToPrevClip` (clip = current clip pos at
   dilated depth; prev = matrix × clip; motion = clip.xy − prev.xy).
4. **Color packing** (compute variants only): simple tonemap
   (color / (maxChannel + preExposure)), RGB→YCoCg remapped to [0,1],
   bit-packed 11/11/10 into one R32UI texel. This is a bandwidth saver
   for tile/mobile GPUs, not a quality feature.

What it writes:

- Compute: `MotionDepth*` RGBA16F (motion.xy, depthclip-or-depth,
  ColorMax-or-alpha) **and** packed `YCoCgColor` R32UI.
- Fragment: only the motion/depth RGBA buffer (alpha channel 0); the
  upscale pass samples the app's RGB color directly.

## 3. Activate pass (render res, 3-pass only)

Role: refine the convert output with temporal information before upscale.

What it reads: convert's motion/depth/alpha buffer + packed YCoCg color +
**previous frame's luma history** (R32UI packing two half floats: last
frame's luma and a signed luma-delta).

What it computes:

1. Reprojects the pixel into the previous frame with its motion vector.
2. Gathers previous-frame depths around that UV (offset gathers +
   bilinear weights) and recomputes the depthclip factor against current
   depth — i.e. disocclusion is tested **temporally** here, not just
   spatially as in 2-pass convert.
3. Luma edge test: current-frame Y (unpacked from YCoCg) vs previous luma
   and the running signed delta. If the delta keeps its sign and the pixel
   isn't disoccluded/reset, the stored delta shrinks toward the smaller
   magnitude (stable-edge tracking); otherwise it restarts. The integer
   part of the alpha channel additionally records whether the pixel was
   flagged as an edge vs flat, and the fractional part carries the
   transparency mask through.
4. Packs the new (luma, delta) pair back into R32UI luma history.

What it writes: refined motion/depthclip/alpha buffer + updated luma
history. Both are render-res and ping-ponged per frame.

## 4. Upscale pass (display resolution)

Role: produce the final upscaled frame and the next history buffer. Runs
once per display-res pixel. This is where ~80% of the visual behavior is.

Per-pixel flow:

1. **Jitter-compensated fetch.** `JitterUV = Hruv + jitter * renderRcp`
   maps the display pixel back to render-res UVs; motion/depth is sampled
   there (bilinear), giving the reprojected history UV
   (`PrevUV = −0.5 * motion + Hruv`, with the NDC-Y variant).
2. **History sample.** Previous display-res history (RGBA + confidence in
   alpha for CS; plain previous output RGB for FS).
3. **Lanczos upsample + statistics box.** Around the render-res anchor
   texel, a cross of 5 taps (FS always; CS when the camera moved) or a
   full 3x3 of 9 taps (CS when the camera is still) is weighted by a
   fast Lanczos-ish kernel whose width (`kernelbias`) adapts to the
   history-confidence and depthclip values. The same taps feed a
   motion-weighted ( comedy: exponential falloff steepens with motion
   length) mean/variance box plus strict min/max bounds.
4. **History clamp.** The variance box (scaled by a factor derived from
   the upscale ratio, relaxed under motion/depthclip) is intersected with
   the min/max box; history is clamped into it. A lerp-contribution rule
   decides how much out-of-box history to keep (a small floor when the
   pixel has no motion, else discard), suppressing ghosting.
5. **Temporal blend.** Blend weight combines a depth/motion-modulated base
   term with the accumulated Lanczos weight; `reset` forces current-frame
   dominance. Result is written to the history buffer (CS; with updated
   confidence) and, after YCoCg→RGB + inverse-exposure unpack (CS) or
   directly (FS), to the display output.

Tap-count summary: CS still-camera = 9 taps, CS moved-camera = 5 taps,
FS = 5 taps always (corner taps exist in source but are compiled out).
3-pass CS upscale is always 9 taps and additionally folds the
transparency/edge fractions from Activate into the confidence and lerp
terms.

## 5. Resources & chaining

2-pass CS per frame: convert(InputColor/Depth/Velocity → motion/depth
RGBA16F + YCoCg R32UI) → upscale(+ PrevHistory display RGBA → SceneOutput
+ HistoryOutput display RGBA). Ping-pong: PrevHistory ↔ HistoryOutput.

2-pass FS per frame: convert(Depth/Velocity → motion/depth RGBA) →
upscale(+ InputColor RGB + PrevOutput RGB → Output RGB). History IS the
previous output — no separate history or packed buffers, smallest memory
footprint (best fit for 4 GB).

3-pass CS per frame: convert(Opaque/Color/Depth/Velocity → YCoCg R32UI +
motion/depth/alpha) → activate(+ PrevLuma → motion/depthclip/alpha +
LumaHistory) → upscale(+ PrevHistory → Scene + History). Ping-pong:
PrevLumaHistory ↔ LumaHistory **and** PrevHistory ↔ HistoryOutput.

## 6. What this means for IGSR

1. No sharpen stage to reimplement — stages 3–4 cover convert (+activate
   later) and upscale only. Our `pass3_sharpen` stub becomes either the
   3-pass activate stage or is dropped; decision in stage 4.
2. The fragment fallback is a **first-class, smaller** algorithm (RGB, 5
   taps, no packed buffers, history == last output), not just a port of
   the compute shader — implement it as its own shaders, which also
   minimizes VRAM pressure on HD 4000.
3. Our uniform block can be unified across variants (render/display sizes
   + rcps, jitter, clipToPrevClip, preExposure, FOV factor, near,
   minLerp, same-camera, reset) instead of copying the three slightly
   different reference layouts.
4. Numerics to be careful with on IVB/crocus: `textureGather`/`textureGatherOffset`
   on integer (R32UI) textures, `packHalf2x16`/`unpackHalf2x16`, and
   `imageStore` to R32UI — all need runtime verification; the FS path
   avoids integer textures entirely, which is another reason to build it
   first.
5. Testbed must supply: jittered camera + `clipToPrevClip` per frame,
   encoded velocity (or zeros for static scenes), depth, and exposure —
   all doable procedurally in stage 5.

## 7. Stage 8C findings — motion handling re-read line by line
Two questions, two answers.

**Motion-vector dilation: checked, confirmed ABSENT.** All three converts
sample velocity at the pixel's own texel; the "dilation" language in the
reference comments ("using nearest depth for dilated motion",
"FindNearestDepth") refers to depth-only nearest gathering that anchors
the depth-reprojection fallback and the depthclip metric — never to
pulling a neighbor's motion vector. Activate passes motion through
untouched. Our chain already matches on all three points, so nothing was
added for dilation. This item is closed, not a gap.

**Temporal depth comparison in activate: checked, was MISSING, now
implemented.** Reference activate gathers previous-frame depths around
`PrevUV` with offset gathers on the depth channel (pairs at (-1,-1)/(-1,0)
and (0,-1)/(0,0), `.zw` swizzle on the second of each pair, min-reduction,
bilinear weights from the fractional position) and tests each against
current depth with the Ksep separation metric
(`Depthsep = Ksep·Kfov·diag·(1−min)`, `clamp(Depthsep/(|Δ|+eps))`),
`depthclip = clamp(1−W)` with no 0.25 factor (unlike 2-pass convert).
Notably it samples the *current* depth buffer at the reprojected UV —
a single-buffer approximation, no previous-depth texture — and ours does
the same against our convert-output depth channel. Two deliberate
departures: (a) the result is unioned (`max`) with convert's spatial
disocclusion instead of replacing it (reference 3-pass convert stores raw
depth, so its activate is temporal-only; ours has both signals and keeps
both); (b) the luma edge flag has a 1e-4 epsilon guard because our flag
is binary while the reference's feeds a scaled alpha term where tiny
deltas self-attenuate. Verified with the new clip debug view (key `C`):
combined clip fires green on true silhouettes, edge flag red on luma sign
flips, both black in 2-pass mode. Fast-motion dumps (`--spin 5`) show no
ghosting trails on the spinning cube or moon.

## 8. Stage 9 — RCAS sharpen post-pass (new capability, not SGSR2)

Sourcing, stated plainly: the project folder vendors
`FidelityFX-FSR/ffx-fsr/ffx_fsr1.h`, and I read the RCAS algorithm there
(`FsrRcasF`/`FsrRcasH`, lines ~600–870: 5-tap cross, luma×2 noise term,
ring min/max, exact no-clip lobe solve with `FSR_RCAS_LIMIT =
0.25−1/16`, normalized resolve) to understand it. The IGSR shaders
(`sharpen.frag`/`.comp`) are our own implementation — own names,
structure, comments, uniform layout, GLSL targets — not a line-by-line
port, the same "read to understand" distinction kept throughout this
project. The published header comments in that file describe the
derivation and were used as the spec.

Deliberate differences from the published shader:

- No denoise term and no luma computation (published default leaves
  `FSR_RCAS_DENOISE` off; with it off, luma feeds nothing — the lobe
  solve is purely per-channel ring min/max). Grain-after-sharpen per
  AMD's recommendation if grain ever lands.
- Sharpness is a linear 0..1 multiplier on the solved lobe, not the
  stops-based `2^-stops` constant setup. 0 gives exact passthrough
  (lobe 0 → output = center tap, verified bit-close in --selftest).
- Border taps clamp (edge replicate); the published callbacks leave
  addressing to the caller.
- Pure black/white neighborhoods divide 0/0 in the limiters exactly as
  published; relies on the same fmax NaN tolerance, verified with
  explicit black/white selftest pixels on crocus (black→black,
  white→white).

Plumbing: strict post-process after upscale on both paths (compute uses
the prelude convention), reads the upscale output, writes a new
display-res target. History keeps unsharpened pixels — sharpening
history would feed amplified detail back into temporal accumulation.
Default sharpness 0.3, keys `Z`/`X`, `B` toggles pre/post display,
`--sharp`/`--spin`/`--dump` (dump also writes the aligned `_pre` frame).
Verified: selftest edge gap 0.600 → 0.700 at sharp=1, and same-frame
dump diffs localize exactly to silhouettes/edges (flat areas untouched).
