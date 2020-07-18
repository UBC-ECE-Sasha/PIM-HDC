#ifndef INIT_H_
#define INIT_H_

#include <stdint.h>

// 2d array to 1d array index
#define A2D1D(d1, i0, i1) (((d1) * (i0)) + (i1))

// Expected versioned binary format (first 4 bytes)
#define VERSION 0

// CHANNELS_VOTING for the componentwise majority must be odd
#define CHANNELS_VOTING (channels + 1)

// Number of CLASSES to be classified
#define CLASSES 5

// Sample size max per DPU in each channel in 32 bit integers (make sure aligned bytes)
#define SAMPLE_SIZE_MAX 512

typedef struct dpu_input_data {
    uint32_t dpu_id;
    uint32_t task_begin[NR_TASKLETS];
    uint32_t task_end[NR_TASKLETS];
    uint32_t idx_offset[NR_TASKLETS];
    uint32_t buffer_channel_aligned_size;
    uint32_t buffer_channel_usable_length;
    uint32_t buffer_channel_length;
    uint32_t output_buffer_length;
} dpu_input_data;

typedef struct dpu_hdc_vars {
    int32_t dimension; /**< Dimension of the hypervectors */
    int32_t channels;  /**< Number of acquisition's CHANNELS */
    int32_t bit_dim;   /**< Dimension of the hypervectors after compression */
    int32_t n;         /**< Dimension of the N-grams */
    int32_t im_length; /**< Item memory length */
    uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
    uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
    uint32_t aM_32[MAX_N * (MAX_BIT_DIM + 1)];
} dpu_hdc_vars;

#endif // INIT_H_
