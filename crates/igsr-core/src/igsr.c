#include "igsr.h"

#include <stdlib.h>

struct IgsrContext {
    IgsrConfig cfg;
    uint64_t frame_index; /* starts at 1 so Halton never returns (0,0) */
};

static uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

IgsrContext *igsr_create(const IgsrConfig *cfg) {
    if (!cfg || cfg->render_w == 0 || cfg->render_h == 0 || cfg->display_w == 0 ||
        cfg->display_h == 0) {
        return 0;
    }
    IgsrContext *ctx = (IgsrContext *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        return 0;
    }
    ctx->cfg = *cfg;
    ctx->frame_index = 1;
    return ctx;
}

void igsr_destroy(IgsrContext *ctx) { free(ctx); }

int igsr_resize(IgsrContext *ctx, uint32_t render_w, uint32_t render_h,
                uint32_t display_w, uint32_t display_h) {
    if (!ctx || render_w == 0 || render_h == 0 || display_w == 0 ||
        display_h == 0) {
        return -1;
    }
    ctx->cfg.render_w = render_w;
    ctx->cfg.render_h = render_h;
    ctx->cfg.display_w = display_w;
    ctx->cfg.display_h = display_h;
    return 0;
}

uint32_t igsr_pass_count(const IgsrContext *ctx) {
    if (!ctx) {
        return 0;
    }
    return ctx->cfg.three_pass ? 3u : 2u;
}

void igsr_render_dispatch(const IgsrContext *ctx, uint32_t local,
                          uint32_t out_xyz[3]) {
    if (!ctx || !out_xyz || local == 0) {
        return;
    }
    out_xyz[0] = ceil_div(ctx->cfg.render_w, local);
    out_xyz[1] = ceil_div(ctx->cfg.render_h, local);
    out_xyz[2] = 1;
}

void igsr_display_dispatch(const IgsrContext *ctx, uint32_t local,
                           uint32_t out_xyz[3]) {
    if (!ctx || !out_xyz || local == 0) {
        return;
    }
    out_xyz[0] = ceil_div(ctx->cfg.display_w, local);
    out_xyz[1] = ceil_div(ctx->cfg.display_h, local);
    out_xyz[2] = 1;
}

uint64_t igsr_frame_index(const IgsrContext *ctx) {
    return ctx ? ctx->frame_index : 0;
}

void igsr_advance_frame(IgsrContext *ctx) {
    if (ctx) {
        ctx->frame_index++;
    }
}

const char *igsr_version_string(void) { return "igsr-core 0.1.0 (stage1)"; }
