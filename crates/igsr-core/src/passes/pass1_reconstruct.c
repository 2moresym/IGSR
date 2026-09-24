#include "../igsr.h"

/* Stage 1 stub. Real work lands in stage 3 (convert/reconstruct):
 * depth-dilate + velocity-decode + RGB->YCoCg packing. Kept as a
 * separate TU so pass ordering / dispatch sizes can be unit-tested
 * before the GLSL exists. */
