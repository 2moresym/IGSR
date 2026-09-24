#version 420 core
// IGSR upscale pass — fragment variant. Own implementation; role described
// in docs/ALGORITHM_NOTES.md §4 (Lanczos upsample + statistics box,
// history clamp, temporal blend). Runs at display resolution, single RGB
// output; history IS the previous frame's output (no separate history or
// packed buffers — smallest footprint, the HD 4000 default path).
//
// Companion: upscale.comp (same algorithm, compute dispatch + explicit
// history/confidence buffer). Motion convention: NDC units, see convert.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_scene;

uniform sampler2D u_color;   // render-res scene RGB
uniform sampler2D u_data;    // convert output: motion.xy, disocc, depth
uniform sampler2D u_history; // display-res previous output RGB
uniform vec2 u_render_size;
uniform vec2 u_render_rcp;
uniform vec2 u_display_size;
uniform vec2 u_display_rcp;
uniform vec2 u_jitter;       // pixels at render res, [-0.5, 0.5]
uniform float u_reset;       // 1 on camera cut / first frame
uniform float u_min_lerp;    // history floor kept when it leaves the box
uniform float u_full_taps;   // 1 = full 3x3, 0 = 5-tap cross

// Fast Lanczos-ish falloff over squared pixel distance. Peaks at 1 for a
// centered tap, decays to 0 at unit distance^2.
float tap_weight(vec2 d, float kb2) {
    float base = clamp(dot(d, d) * kb2, 0.0, 1.0);
    float y = base - 1.0;
    float y2 = y * y;
    return (0.75 * y + y2) * y2;
}

void main() {
    vec2 hruv = v_uv;
    vec2 juv = clamp(hruv + u_jitter * u_render_rcp, 0.0, 1.0);
    ivec2 anchor = ivec2(juv * u_render_size);

    vec4 md = textureLod(u_data, juv, 0.0);
    vec2 motion = md.xy;
    float disocc = md.z;

    vec2 prev_uv = clamp(hruv - 0.5 * motion, 0.0, 1.0);
    vec3 hist = textureLod(u_history, prev_uv, 0.0).rgb;

    // Kernel width adapts to disocclusion: trusted pixels get a tight
    // kernel (detail), disoccluded pixels a wide one (denoise).
    float ratio = min(u_display_size.x / u_render_size.x, 1.99);
    float kb = mix(1.0, ratio, clamp(disocc + u_reset, 0.0, 1.0)) * 0.5;
    float kb2 = kb * kb;
    float motion_len = length(motion * u_display_size);
    float curve = mix(-2.0, -3.0, clamp(motion_len * 0.02, 0.0, 1.0));

    vec2 src = vec2(anchor) + vec2(0.5) - u_jitter; // jitter-relative anchor
    vec2 rel = src - hruv * u_render_size;          // anchor vs output pos

    // Cross first (always), corners after (still-camera 9-tap mode).
    const vec2 OFF[9] = vec2[9](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(-1.0, 0.0),
        vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(1.0, 1.0), vec2(-1.0, 1.0),
        vec2(1.0, -1.0), vec2(-1.0, -1.0));

    vec4 acc = vec4(0.0); // rgb * w, w
    vec3 box_c = vec3(0.0), box_v = vec3(0.0), box_min, box_max;
    float box_w = 0.0;
    for (int i = 0; i < 9; i++) {
        if (i >= 5 && u_full_taps < 0.5) {
            break;
        }
        vec3 s = texelFetch(u_color, anchor + ivec2(OFF[i]), 0).rgb;
        vec2 d = rel + OFF[i];
        float w = tap_weight(d, kb2);
        acc += vec4(s * w, w);
        float bw = exp(dot(d, d) * curve);
        if (i == 0) {
            box_min = s;
            box_max = s;
        } else {
            box_min = min(box_min, s);
            box_max = max(box_max, s);
        }
        box_c += s * bw;
        box_v += s * s * bw;
        box_w += bw;
    }
    box_c /= box_w;
    box_v = sqrt(abs(box_v / box_w - box_c * box_c));

    vec3 up = clamp(acc.rgb / max(acc.w, 1e-6), box_min - 0.05, box_max + 0.05);
    float uw = acc.w / 3.0;

    // Confidence box: variance scaled by upscale ratio, relaxed under
    // motion/disocclusion, intersected with the strict min/max box.
    float relax = max(disocc, clamp(motion_len * 0.05, 0.0, 1.0));
    float box_scale = mix(ratio * ratio, 1.0, relax);
    vec3 lo = max(box_min, box_c - box_v * box_scale);
    vec3 hi = min(box_max, box_c + box_v * box_scale);

    vec3 clamped = clamp(hist, lo, hi);
    bool outside = any(lessThan(hist, lo)) || any(greaterThan(hist, hi));
    float keep = 1.0;
    if (outside) {
        keep = u_min_lerp;
        if (abs(motion.x) + abs(motion.y) > 1e-6) {
            keep = 0.0; // moving + out of box = ghost source: drop history
        }
    }
    vec3 h = mix(clamped, hist, keep);

    float base = (1.0 - disocc);
    base = min(base, mix(base, uw * 10.0, clamp(motion_len * 10.0, 0.0, 1.0)));
    base = min(base, mix(base, uw, clamp(motion_len * 0.05, 0.0, 1.0)));
    float alpha = clamp(uw / max(base + uw, 1.19e-07) + u_reset, 0.0, 1.0);
    o_scene = vec4(mix(h, up, alpha), 1.0);
}
