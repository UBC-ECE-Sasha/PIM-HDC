#ifndef INIT_H_
#define INIT_H_

#include <stdint.h>

// 2d array to 1d array index
#define A2D1D(d1,i0,i1) (((d1) * (i0)) + (i1))

// Expected versioned binary format (first 4 bytes)
#define VERSION 0

// CHANNELS_VOTING for the componentwise majority must be odd
#define CHANNELS_VOTING (channels + 1)

// Number of CLASSES to be classified
#define CLASSES 5

// Sample size max per DPU in each channel in 32 bit integers (make sure aligned bytes)
#define SAMPLE_SIZE_MAX 512

#endif // INIT_H_
