#include "../igsr.h"

/* Pass 3 — activate (render-res, 3-pass quality path only). See
 * shaders/activate.comp (own implementation): temporal clip refinement +
 * luma-delta tracking into an RG-float history. Renamed from pass3_sharpen
 * in stage 4: the studied v2 design has no sharpen pass (ALGORITHM_NOTES
 * §0). First exercised end-to-end in stage 5 when luma-history ping-pong
 * exists; stage 4 wires its compile probe. */
