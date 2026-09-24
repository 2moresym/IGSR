#version 420 core
// IGSR convert pass — fragment variant. Own implementation; role described
// in docs/ALGORITHM_NOTES.md §2 (dilated depth, disocclusion factor,
// motion derivation). Reads app color/depth/velocity at render resolution,
// writes one RGBA16F buffer: motion.xy, disocclusion, nearest depth.
//
// Requires GLSL 4.20 (textureGather + explicit fragment locations). That is
// the IGSR baseline: GL 4.0+ / GLES 3.1+ class hardware, which includes our
// HD 4000 target (Mesa crocus: GL 4.2). The *fragment fallback* in IGSR
// means "no compute shaders", not "ancient GL".
//
// Our departures from the studied design (deliberate, documented):
// - Depth arrives as a plain R32F texture, not a depth texture, so plain
//   textureGather works on crocus without shadow-sampler setup.
// - Velocity is raw clip-space deltas (RG float), exact (0,0) = static.
// - Single output buffer; no bit-packed R32UI intermediates.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_data;

uniform sampler2D u_depth;    // R32F, linear 0..1 depth
uniform sampler2D u_velocity; // RG float, raw clip-space delta, (0,0)=static
uniform vec2 u_render_size;
uniform vec2 u_render_rcp;
uniform mat4 u_clip_to_prev;  // prevVP * invCurrVP
uniform float u_fov_hor;      // tan(hfov/2)-style horizontal FOV factor

// Separation sensitivity for the disocclusion test. Same order of
// magnitude as the studied design; retune on the testbed if silhouette
// pixels shimmer. See ALGORITHM_NOTES §2.
const float K_SEP = 1.4e-05;
const float EPS = 1.19e-07;

float quad_min(vec4 g) {
    return min(min(g.x, g.y), min(g.z, g.w));
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec2 center = (vec2(px) + vec2(0.5)) * u_render_rcp;

    // Four gathers around the pixel cover a 4x4 footprint; reduce to the
    // nearest depth plus one minimum per quadrant. (Gather component order
    // is irrelevant: only minima are used.)
    vec4 g00 = textureGather(u_depth, center + vec2(-u_render_rcp.x, -u_render_rcp.y));
    vec4 g10 = textureGather(u_depth, center + vec2(+u_render_rcp.x, -u_render_rcp.y));
    vec4 g01 = textureGather(u_depth, center + vec2(-u_render_rcp.x, +u_render_rcp.y));
    vec4 g11 = textureGather(u_depth, center + vec2(+u_render_rcp.x, +u_render_rcp.y));
    float q0 = quad_min(g00);
    float q1 = quad_min(g10);
    float q2 = quad_min(g01);
    float q3 = quad_min(g11);
    float nearest = min(min(q0, q1), min(q2, q3));

    // Disocclusion factor: pixels whose quadrants disagree about depth are
    // likely silhouette edges; history must not drag there.
    float disocc = 0.0;
    if (nearest < 1.0 - 1.0e-05) {
        float gap = K_SEP * u_fov_hor * length(u_render_size) * (1.0 - nearest);
        float w = 0.0;
        w += clamp(gap / (abs(nearest - q0) + EPS), 0.0, 1.0);
        w += clamp(gap / (abs(nearest - q1) + EPS), 0.0, 1.0);
        w += clamp(gap / (abs(nearest - q2) + EPS), 0.0, 1.0);
        w += clamp(gap / (abs(nearest - q3) + EPS), 0.0, 1.0);
        disocc = clamp(1.0 - w * 0.25, 0.0, 1.0);
    }

    // Motion: dynamic pixels carry it in the velocity texture; static
    // pixels (exact zero) reproject through depth + the clip matrix.
    vec2 vel = texelFetch(u_velocity, px, 0).xy;
    vec2 motion;
    if (vel.x != 0.0 || vel.y != 0.0) {
        motion = vel;
    } else {
        vec2 screen = v_uv * 2.0 - 1.0;
        vec4 prev = u_clip_to_prev * vec4(screen, nearest, 1.0);
        motion = screen - prev.xy / prev.w;
    }

    o_data = vec4(motion, disocc, nearest);
}
