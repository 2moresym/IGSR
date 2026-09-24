#include "../igsr.h"

/* Pass 1 — convert (render-res). See shaders/convert.{frag,comp} for the
 * algorithm (own implementation): dilated depth, disocclusion factor,
 * motion derivation. Dispatch sizing via igsr_render_dispatch(); uniform
 * packing via igsr_fill_params(). No per-pass C dispatch code needed: the
 * backend issues the draw/dispatch directly (stage 3 selftest proves it). */
