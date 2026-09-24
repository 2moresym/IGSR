#include "igsr.h"

/* Own implementation of the Halton low-discrepancy sequence.
 * SGSR2's UBO documents jitterOffset in [-0.5, 0.5]; we generate
 * Halton values in [0,1) and shift by -0.5. Bases 2 and 3 match the
 * conventional TAA/FSR2 jitter pattern and the IGSR spec. */

static float halton(uint64_t index, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f /= (float)base;
        r += f * (float)(index % base);
        index /= base;
    }
    return r;
}

void igsr_calc_jitter(uint64_t frame_index, float out_jitter[2]) {
    if (!out_jitter) {
        return;
    }
    /* Frame 0 would yield (0,0); callers start at 1. Guard anyway. */
    uint64_t i = frame_index == 0 ? 1 : frame_index;
    out_jitter[0] = halton(i, 2) - 0.5f;
    out_jitter[1] = halton(i, 3) - 0.5f;
}
