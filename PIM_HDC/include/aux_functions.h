#ifndef AUX_FUNCTIONS_H_
#define AUX_FUNCTIONS_H_

#include "init.h"

void hamming_dist(uint32_t q[BIT_DIM + 1], uint32_t aM[][BIT_DIM + 1], int sims[CLASSES]);
int max_dist_hamm(int distances[CLASSES]);
void computeNgram(int input[CHANNELS], uint32_t iM[][BIT_DIM + 1], uint32_t chAM[][BIT_DIM + 1], uint32_t query[BIT_DIM + 1]);
int numberOfSetBits(uint32_t i);

void quantize(float input_buffer[CHANNELS], int output_buffer[CHANNELS]);

#endif
