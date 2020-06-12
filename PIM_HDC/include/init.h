#ifndef INIT_H_
#define INIT_H_

#include <stdint.h>

//dimension of the hypervectors
#define DIMENSION 10000
//number of CLASSES to be classify
#define CLASSES 5
//number of acquisition's CHANNELS
#define CHANNELS 4
//dimension of the hypervectors after compression (dimension/32 rounded to the smallest integer)
#define BIT_DIM 312
//number of input samples
#define NUMBER_OF_INPUT_SAMPLES 14883
//dimension of the N-grams (models for N = 1 and N = 5 are contained in data.h)
#define N 5
//CHANNELS_VOTING for the componentwise majority must be odd
#define CHANNELS_VOTING CHANNELS + 1

#define TEST 1

typedef struct in_buffer {
    int32_t * buffer;
    uint32_t buffer_size;
} in_buffer;

// Sample size max per DPU in each channel in 32 bit integers (make sure aligned bytes)
#define SAMPLE_SIZE_MAX 512


#endif
