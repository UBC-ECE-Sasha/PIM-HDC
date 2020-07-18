#ifndef GLOBAL_DPU_H_
#define GLOBAL_DPU_H_

#include "init.h"

#include <defs.h>
#include <perfcounter.h>

extern dpu_hdc_vars hd;

extern uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
extern uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
extern uint32_t aM_32[MAX_N * (MAX_BIT_DIM + 1)];

typedef struct in_buffer {
    int32_t *buffer;
    uint32_t buffer_size;
} in_buffer;

extern perfcounter_t compute_N_gram_cycles;
extern perfcounter_t associative_memory_cycles;
extern perfcounter_t bit_mod_cycles;

#endif
