#include <mram.h>
#include <seqread.h>
#include <defs.h>
#include <perfcounter.h>
#include <stdio.h>
#include <errno.h>
#include "alloc.h"
#include "common.h"

#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "data.h"

__host __mram_ptr int8_t * input_buffer;
__host uint32_t buffer_channel_length;
__host uint32_t buffer_channel_offset;
__host uint32_t buffer_channel_usable_length;

/**
 * @brief Fill @p read_buf with data from @p input_buffer.
 *
 * @param[out] read_buf    Buffers filled with sample data.
 * @param[in] num_samples  Number of samples to read from MRAM.
 * @return                 @p ENOMEM on failure. Zero on success.
 */
static int alloc_buffers(int32_t read_buf[CHANNELS][SAMPLE_SIZE_MAX]) {
    if (buffer_channel_usable_length > SAMPLE_SIZE_MAX) {
        printf("Cannot use buffer of sample size over %d, use smaller dataset\n", SAMPLE_SIZE_MAX);
        return ENOMEM;
    }

    for (int i = 0; i < CHANNELS; i++) {
        mram_read(&input_buffer[i * buffer_channel_offset], read_buf[i], buffer_channel_offset);
    }

    return 0;
}

/**
 * @breif Run HDC algorithm on host
 *
 * @return Non-zero on failure.
 */
static int dpu_hdc() {
    /* Buffer for each channel */

    int32_t quantized_buffer[CHANNELS];

    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[BIT_DIM + 1] = {0};
    uint32_t q_N[BIT_DIM + 1] = {0};
    int class;

    int ret = 0;

    __dma_aligned int32_t read_buf[CHANNELS][SAMPLE_SIZE_MAX];

    ret = alloc_buffers(read_buf);
    if (ret != 0) {
        return ret;
    }

    for(int ix = 0; ix < buffer_channel_length; ix += N) {

        for(int z = 0; z < N; z++) {

            for(int j = 0; j < CHANNELS; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < buffer_channel_usable_length) {
                    quantized_buffer[j] = read_buf[j][ix + z];
                    dbg_printf("quantized_buffer[%d] = data_set[%d][%d + %d] = %d\n", j, j, ix, z, quantized_buffer[j]);
                }
            }

            // Spatial and Temporal Encoder: computes the N-gram.
            // N.B. if N = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, iM, chAM, q);
            } else {
                compute_N_gram(quantized_buffer, iM, chAM, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for(int i = 1; i < BIT_DIM; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & mask;
                    q[i] = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ q[i];
                }

                old_overflow = overflow;
                overflow = (q[BIT_DIM] >> 16) & mask;
                q[BIT_DIM] = (q[BIT_DIM] >> 1) | (old_overflow << (32 - 1));
                q[BIT_DIM] = q_N[BIT_DIM] ^ q[BIT_DIM];

                q[0] = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ q[0];
            }
        }

        // Classifies the new N-gram through the Associative Memory matrix.
        class = associative_memory_32bit(q, aM_32);

        printf("%d\n", class);

    }

    return ret;
}

int main() {
    uint8_t idx = me();
    (void)idx;

    dbg_printf("DPU starting, tasklet %d\n", idx);

    perfcounter_config(COUNT_CYCLES, true);

    int ret = 0;

    if (buffer_channel_length > 0) {
        ret = dpu_hdc();
    } else {
        printf("No work to do\n");
    }

    dbg_printf("Tasklet %d: completed in %ld cycles\n", idx, perfcounter_get());

    return ret;
}
