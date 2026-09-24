#pragma once

/* igsr.h — public C API for the IGSR core.
 *
 * OWN CODE. Backend-agnostic by design: this core never calls a GPU API
 * directly. All GPU work is issued by the Rust `GpuBackend` through the
 * IgsrDispatchTable function-pointer table (or, for the GL backend,
 * by compiling the GLSL in crates/igsr-shaders and dispatching with the
 * sizes computed here).
 *
 * Memory discipline: the target has 4 GB total RAM. All allocations are
 * explicit in IgsrConfig; no hidden mallocs per frame.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IGSR_VERSION_MAJOR 0
#define IGSR_VERSION_MINOR 1
#define IGSR_VERSION_PATCH 0

/* Render (low-res input) and display (upscaled output) extents. */
typedef struct IgsrConfig {
    uint32_t render_w;
    uint32_t render_h;
    uint32_t display_w;
    uint32_t display_h;
    /* 1 = prefer compute path, 0 = force fragment fallback. */
    int32_t prefer_compute;
    /* 1 = 3-pass quality path, 0 = 2-pass speed path (default). */
    int32_t three_pass;
    /* Reset accumulation (camera cut). Set for the first frame. */
    uint32_t reset;
} IgsrConfig;

/* Packed per-frame parameters. Mirrors the uniform block consumed by our
 * own shaders in crates/igsr-shaders (NOT the reference shaders). */
typedef struct IgsrParams {
    float render_size[2];
    float display_size[2];
    float render_size_rcp[2];
    float display_size_rcp[2];
    float jitter[2];
    float clip_to_prev_clip[16];
    float pre_exposure;
    float camera_fov_hor;
    float camera_near;
    float min_lerp_contrib;
    uint32_t same_camera_frames;
    uint32_t reset;
} IgsrParams;

/* Backend callback table. Filled in by Rust; called by C pass code so the
 * C stays GPU-API-agnostic. All stage-1 entries are optional (may be NULL). */
typedef struct IgsrDispatchTable {
    void *user;
    void (*dispatch_compute)(void *user, uint32_t gx, uint32_t gy, uint32_t gz);
    void (*draw_fullscreen)(void *user);
} IgsrDispatchTable;

typedef struct IgsrContext IgsrContext;

IgsrContext *igsr_create(const IgsrConfig *cfg);
void igsr_destroy(IgsrContext *ctx);
/* Returns 0 on success, <0 on invalid sizes. */
int igsr_resize(IgsrContext *ctx, uint32_t render_w, uint32_t render_h,
                uint32_t display_w, uint32_t display_h);

/* Halton(2,3)-based jitter in [-0.5, 0.5]. frame_index starts at 1.
 * out_jitter[2] receives (x, y). Pure function of frame_index. */
void igsr_calc_jitter(uint64_t frame_index, float out_jitter[2]);

/* Number of logical passes for this config (2 or 3). */
uint32_t igsr_pass_count(const IgsrContext *ctx);
/* Work-group count for a render-res pass with given local size (default 8). */
void igsr_render_dispatch(const IgsrContext *ctx, uint32_t local,
                          uint32_t out_xyz[3]);
/* Work-group count for a display-res pass with given local size. */
void igsr_display_dispatch(const IgsrContext *ctx, uint32_t local,
                           uint32_t out_xyz[3]);

uint64_t igsr_frame_index(const IgsrContext *ctx);
void igsr_advance_frame(IgsrContext *ctx);

const char *igsr_version_string(void);

#ifdef __cplusplus
}
#endif
