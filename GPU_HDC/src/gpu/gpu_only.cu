#include "init.h"
#include "common.h"

#include <string.h>
#include <stdio.h>
#include <driver_types.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define MASK 1

/**
 * @brief Computes the number of 1's
 *
 * @param i The i-th variable that composes the hypervector
 * @return  Number of 1's in i-th variable of hypervector
 */
__device__ static inline int
number_of_set_bits(uint32_t i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

/**
 * @brief Computes the maximum Hamming Distance.
 *
 * @param[in] distances Distances associated to each class
 * @return              The class related to the maximum distance
 */
__device__ static int
max_dist_hamm(int distances[CLASSES]) {
    int max = distances[0];
    int max_index = 0;

    for (int i = 1; i < CLASSES; i++) {
        if (max > distances[i]) {
            max = distances[i];
            max_index = i;
        }
    }

    return max_index;
}

/**
 * @brief Computes the Hamming Distance for each class.
 *
 * @param[in] q     Query hypervector
 * @param[in] aM    Associative Memory matrix
 * @param[out] sims Distances' vector
 */
__device__ static void
hamming_dist(uint32_t *q, uint32_t *aM, int sims[CLASSES], gpu_hdc_vars *hd) {
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < hd->bit_dim + 1; j++) {
            sims[i] += number_of_set_bits(q[j] ^ aM[A2D1D(hd->bit_dim + 1, i, j)]);
        }
    }
}

/**
 * @brief Tests the accuracy based on input testing queries.
 *
 * @param[in] q_32  Query hypervector
 * @param[in] aM_32 Trained associative memory
 * @return          Classification result
 */
__device__ static int
associative_memory_32bit(uint32_t *q_32, uint32_t *aM_32, gpu_hdc_vars *hd) {
    int sims[CLASSES] = {0};

    // Computes Hamming Distances
    hamming_dist(q_32, aM_32, sims, hd);

    // Classification with Hamming Metric
    return max_dist_hamm(sims);
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[out] query      Query hypervector
 */
__device__ static void
compute_N_gram(int32_t *input, uint32_t *query, gpu_hdc_vars *hd) {

    uint32_t chHV[MAX_CHANNELS + 1];

    for (int i = 0; i < hd->bit_dim + 1; i++) {
        query[i] = 0;
        for (int j = 0; j < hd->channels; j++) {
            int ix = input[j];

            uint32_t im = hd->iM[A2D1D(hd->bit_dim + 1, ix, i)];
            uint32_t cham = hd->chAM[A2D1D(hd->bit_dim + 1, j, i)];

            chHV[j] = im ^ cham;
        }
        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[hd->channels] = chHV[0] ^ chHV[1];

        // componentwise majority: compute the number of 1's
        for (int z = 31; z >= 0; z--) {
            uint32_t cnt = 0;
            for (int j = 0; j < hd->channels + 1; j++) {
                uint32_t a = chHV[j] >> z;
                uint32_t mask = a & 1;
                cnt += mask;
            }

            if (cnt > 2) {
                query[i] = query[i] | (1 << z);
            }
        }
    }
}


/**
 * @brief Run HDC algorithm
 * @param[out] result         Buffer to place results in
 * @param[out] result_offset  Offset to start placing results from
 * @param[in] task_begin      Position to start task from
 * @param[in] task_end        Position to end task at
 * @param[in] hd              HDC vars
 *
 * @return                    Non-zero on failure.
 */
__global__ void
gpu_hdc(gpu_input_data *gpu_data, int32_t *read_buf, int32_t *result, gpu_hdc_vars *hd) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;

    uint32_t q[MAX_BIT_DIM + 1] = {0};
    uint32_t q_N[MAX_BIT_DIM + 1] = {0};
    int32_t quantized_buffer[MAX_CHANNELS] = {0};

    int result_num = 0;

    int thr = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ((gpu_data->task_end[thr] - gpu_data->task_begin[thr]) <= 0) {
        dbg_printf("%u: No work to do\n", thr);
        return;
    }

    for (int ix = gpu_data->task_begin[thr]; ix < gpu_data->task_end[thr]; ix += hd->n) {

        for (int z = 0; z < hd->n; z++) {

            for (int j = 0; j < hd->channels; j++) {
                if (ix + z < gpu_data->buffer_channel_usable_length) {
                    int ind = A2D1D(gpu_data->buffer_channel_usable_length, j, ix + z);
                    quantized_buffer[j] = read_buf[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the N-gram.
            // N.B. if N = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, q, hd);
            } else {
                compute_N_gram(quantized_buffer, q_N, hd);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                int32_t shifted_q;
                overflow = q[0] & MASK;
                for (int i = 1; i < hd->bit_dim; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & MASK;
                    shifted_q = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ shifted_q;
                }

                old_overflow = overflow;
                overflow = (q[hd->bit_dim] >> 16) & MASK;
                shifted_q = (q[hd->bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[hd->bit_dim] = q_N[hd->bit_dim] ^ shifted_q;

                shifted_q = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ shifted_q;
            }
        }

        // Classifies the new N-gram through the Associative Memory matrix.
        result[gpu_data->idx_offset[thr] + result_num] = associative_memory_32bit(q, hd->aM_32, hd);
        // printf("i=%i,r=%i\n", result_num, result[gpu_data->idx_offset[thr] + result_num]);
        result_num++;
    }
}
