#ifndef AUX_FUNCTIONS_H_
#define AUX_FUNCTIONS_H_

#include "init.h"

#ifdef HOST
#include "host_only.h"
#else
#include "global_dpu.h"
#endif

void hamming_dist(uint32_t q[bit_dim + 1], uint32_t *aM, int sims[CLASSES]);

int max_dist_hamm(int distances[CLASSES]);

void compute_N_gram(int input[channels], uint32_t *channel_iM, uint32_t *channel_AM, uint32_t query[bit_dim + 1]);

int number_of_set_bits(uint32_t i);

#endif
