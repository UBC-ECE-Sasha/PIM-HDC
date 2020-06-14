#ifndef INIT_H_
#define INIT_H_

#include <stdint.h>

// Dimension of the hypervectors
#define DIMENSION 10000
// Number of CLASSES to be classified
#define CLASSES 5
// Number of acquisition's CHANNELS
#define CHANNELS 4
// Dimension of the hypervectors after compression (dimension/32 rounded to the smallest integer)
#define BIT_DIM (DIMENSION >> 5)

// Number of input samples
#ifdef TEST
#define NUMBER_OF_INPUT_SAMPLES 14883
#else
#define NUMBER_OF_INPUT_SAMPLES 1489
#endif

// Dimension of the N-grams (models for N = 1 and N = 5 are contained in data.h)
#define N 5
// CHANNELS_VOTING for the componentwise majority must be odd
#define CHANNELS_VOTING CHANNELS + 1

typedef struct in_buffer {
    int32_t * buffer;
    uint32_t buffer_size;
} in_buffer;

// Sample size max per DPU in each channel in 32 bit integers (make sure aligned bytes)
#define SAMPLE_SIZE_MAX 512

#endif // INIT_H_
