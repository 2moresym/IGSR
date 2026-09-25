#version 420 core
// IGSR sharpen pass — fragment variant. Own implementation of the published
// RCAS technique (read to understand from the vendored FSR1 source,
// `FidelityFX-FSR/ffx-fsr/ffx_fsr1.h` `FsrRcasF`; see ALGORITHM_NOTES §8).
// Every name, comment, layout, and line below is ours; the math follows the
// published algorithm: 5-tap cross, per-channel ring min/max, exact
// no-clip lobe solve, normalized resolve.
//
// Role: strict post-process after upscale. Reads the final upscale output,
// writes the sharpened frame. Never touches history, luma, or depth
// buffers. Sharpness 0 = exact passthrough (lobe 0 → output = center tap).
//
// Assumes input in [0,1] like the published design (our upscale output).
// Pure-black/white neighborhoods hit 0/0 in the limiter ratios; like the
// published shader this relies on fmax NaN tolerance (verified on crocus
// in --selftest with explicit black/white pixels). No denoise term — the
// published default leaves FSR_RCAS_DENOISE off (grain goes after sharpen).

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_sharp;

uniform sampler2D u_image; // display-res upscale output, RGB
uniform vec2 u_display_size;
uniform vec2 u_display_rcp;
uniform float u_sharp; // 0 = off/passthrough .. 1 = full

// Lobe clamp: the published stability limit (0.25 - 1/16).
const float LOBE_LIMIT = 0.1875;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 lo = ivec2(0);
    ivec2 hi = ivec2(u_display_size) - ivec2(1);
    // Clamped taps: outside the viewport, replicate the edge texel so the
    // border behaves instead of reading undefined texels.
    vec3 n = texelFetch(u_image, ivec2(px.x, min(px.y + 1, hi.y)), 0).rgb;
    vec3 w = texelFetch(u_image, ivec2(max(px.x - 1, lo.x), px.y), 0).rgb;
    vec3 m = texelFetch(u_image, px, 0).rgb;
    vec3 e = texelFetch(u_image, ivec2(min(px.x + 1, hi.x), px.y), 0).rgb;
    vec3 s = texelFetch(u_image, ivec2(px.x, max(px.y - 1, lo.y)), 0).rgb;

    // No luma, no noise term: without the (default-off) denoise option the
    // published algorithm never uses luma — the lobe solve is purely
    // per-channel ring min/max. Deliberately omitted; re-add both if grain
    // handling ever lands (grain goes after sharpen, per AMD).
    vec3 mn4 = min(min(n, w), min(e, s));
    vec3 mx4 = max(max(n, w), max(e, s));

    // Exact no-clip lobe solve per channel: hitMin from the dark side
    // (signal hitting 0), hitMax from the bright side (hitting 1), using
    // 4x ring extrema for stability. lobe <= 0 always (sharpening lobe).
    // Pure black/white neighborhoods divide 0/0 here; like the published
    // shader this relies on fmax NaN tolerance (black/white pixels are
    // asserted in --selftest on this driver).
    vec3 hit_min = min(mn4, m) / (vec3(4.0) * mx4);
    vec3 hit_max = (vec3(1.0) - max(mx4, m)) / (vec3(4.0) * mn4 - vec3(4.0));
    vec3 lobe_ch = max(-hit_min, hit_max);
    float lobe = max(-LOBE_LIMIT, min(max(max(lobe_ch.x, lobe_ch.y), lobe_ch.z), 0.0));
    lobe *= u_sharp;

    // Normalized resolve: (lobe*(ring) + m) / (4*lobe + 1).
    float rcp = 1.0 / (4.0 * lobe + 1.0);
    vec3 out_col = (lobe * (n + w + e + s) + m) * rcp;
    o_sharp = vec4(out_col, 1.0);
}
