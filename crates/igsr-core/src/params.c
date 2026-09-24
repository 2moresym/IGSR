#include "igsr.h"
#include "igsr_priv.h"

#include <stddef.h>

void igsr_fill_params(const IgsrContext *ctx, const IgsrFrameInputs *in,
                      IgsrParams *out) {
    if (!ctx || !in || !out) {
        return;
    }
    out->render_size[0] = (float)ctx->cfg.render_w;
    out->render_size[1] = (float)ctx->cfg.render_h;
    out->display_size[0] = (float)ctx->cfg.display_w;
    out->display_size[1] = (float)ctx->cfg.display_h;
    out->render_size_rcp[0] = 1.0f / (float)ctx->cfg.render_w;
    out->render_size_rcp[1] = 1.0f / (float)ctx->cfg.render_h;
    out->display_size_rcp[0] = 1.0f / (float)ctx->cfg.display_w;
    out->display_size_rcp[1] = 1.0f / (float)ctx->cfg.display_h;
    out->jitter[0] = in->jitter[0];
    out->jitter[1] = in->jitter[1];
    for (int i = 0; i < 16; i++) {
        out->clip_to_prev_clip[i] = in->clip_to_prev[i];
    }
    out->pre_exposure = in->pre_exposure;
    out->camera_fov_hor = in->camera_fov_hor;
    out->camera_near = in->camera_near;
    out->min_lerp_contrib = in->min_lerp_contrib;
    out->same_camera_frames = in->same_camera_frames;
    out->reset = in->reset;
}

void igsr_reproject_motion(const float clip_xy[2], float depth,
                           const float m[16], float out_motion[2]) {
    if (!clip_xy || !m || !out_motion) {
        return;
    }
    /* Row-major 4x4 times (x, y, depth, 1). */
    float x = m[0] * clip_xy[0] + m[1] * clip_xy[1] + m[2] * depth + m[3];
    float y = m[4] * clip_xy[0] + m[5] * clip_xy[1] + m[6] * depth + m[7];
    float w = m[12] * clip_xy[0] + m[13] * clip_xy[1] + m[14] * depth + m[15];
    if (w == 0.0f) {
        out_motion[0] = 0.0f;
        out_motion[1] = 0.0f;
        return;
    }
    out_motion[0] = clip_xy[0] - x / w;
    out_motion[1] = clip_xy[1] - y / w;
}
