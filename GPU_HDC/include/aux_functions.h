#ifndef AUX_FUNCTIONS_H_
#define AUX_FUNCTIONS_H_

#include "init.h"

#include "host_only.h"

int
host_associative_memory_32bit(uint32_t q_32[hd.bit_dim + 1], uint32_t *aM_32);
void
host_compute_N_gram(int input[hd.channels], uint32_t query[hd.bit_dim + 1]);

#endif
