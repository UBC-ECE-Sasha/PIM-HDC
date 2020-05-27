#ifndef AUX_FUNCTIONS_H_
#define AUX_FUNCTIONS_H_

#include "init.h"

void hamming_dist(uint32_t q[BIT_DIM + 1], uint32_t aM[][BIT_DIM + 1], int sims[CLASSES]);

int max_dist_hamm(int distances[CLASSES]);

void compute_N_gram(int input[CHANNELS], uint32_t channel_iM[][BIT_DIM + 1], uint32_t channel_AM[][BIT_DIM + 1], uint32_t query[BIT_DIM + 1]);

int number_of_set_bits(uint32_t i);

#endif
