#ifndef GLOBAL_DPU_H_
#define GLOBAL_DPU_H_

#include <perfcounter.h>

extern perfcounter_t total_cycles;
extern perfcounter_t alloc_buffers_cycles;
extern perfcounter_t compute_N_gram_top_cycles;
extern perfcounter_t compute_N_gram_bottom_cycles;
extern perfcounter_t bit_mod_cycles;
extern perfcounter_t bit_mod_cycles;

#endif
