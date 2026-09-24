#include "../igsr.h"

/* Pass 2 — upscale (display-res, both variants). See
 * shaders/upscale.{frag,comp} for the algorithm (own implementation):
 * Lanczos upsample + statistics box, history clamp, temporal blend.
 * Fragment variant writes one RGB target (history == previous output);
 * compute variant writes scene + history/confidence images. Dispatch via
 * igsr_display_dispatch(); selected by backend compute_path(). */
