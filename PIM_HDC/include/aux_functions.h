#ifndef AUX_FUNCTIONS_H_
#define AUX_FUNCTIONS_H_

#include "init.h"

#ifdef HOST
#    include "host_only.h"
#else
#    include "global_dpu.h"
#endif

int
associative_memory_32bit(uint32_t q_32[hd.bit_dim + 1], uint32_t *aM_32);

void
hamming_dist(uint32_t q[hd.bit_dim + 1], uint32_t *aM, int sims[CLASSES]);

int
max_dist_hamm(int distances[CLASSES]);

void
compute_N_gram(int input[hd.channels], uint32_t query[hd.bit_dim + 1]);

int
number_of_set_bits(uint32_t i);

#ifndef HOST
uint32_t
read_32bits_from_mram(uint32_t ind, uint32_t __mram_ptr *mram_buf);
#endif

#endif
