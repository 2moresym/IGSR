#pragma once
/* Internal layout of IgsrContext. Only core TUs include this; backends see
 * the opaque pointer from igsr.h. */
#include "igsr.h"

struct IgsrContext {
    IgsrConfig cfg;
    uint64_t frame_index; /* starts at 1 so Halton never returns (0,0) */
};
