#include <mram.h>
#include <seqread.h>
#include <defs.h>
#include <perfcounter.h>
#include <stdio.h>
#include <errno.h>
#include <alloc.h>
#include <built_ins.h>
#include <string.h>

#include "global_dpu.h"
#include "common.h"
#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "cycle_counter.h"

#define MASK 1
#define MRAM_MAX_READ_SIZE 2048

__host __mram_ptr int8_t *input_buffer;
__host __mram_ptr uint8_t *mram_chAM;
__host __mram_ptr uint8_t *mram_iM;
__host __mram_ptr uint8_t *mram_aM_32;

__host uint32_t buffer_channel_length;
__host uint32_t buffer_channel_aligned_size;
__host uint32_t buffer_channel_usable_length;
__host int32_t dimension;
__host int32_t channels;
__host int32_t bit_dim;
__host int32_t number_of_input_samples;
__host int32_t n;
__host int32_t im_length;

__dma_aligned uint32_t *chAM;
__dma_aligned uint32_t *iM;
__dma_aligned uint32_t *aM_32;
__dma_aligned uint32_t *chHV;

perfcounter_t counter = 0;
perfcounter_t compute_N_gram_cycles = 0;
perfcounter_t associative_memory_cycles = 0;
perfcounter_t bit_mod_cycles = 0;

// Original array lengths
// double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
// uint32_t chAM[CHANNELS][BIT_DIM + 1];
// uint32_t iM[IM_LENGTH][BIT_DIM + 1];
// uint32_t aM_32[N][BIT_DIM + 1];

/**
 * @brief Fill @p read_buf with data from @p input_buffer, populate globals
 *
 * @param[out] read_buf    Buffers filled with sample data.
 * @return                 @p ENOMEM on failure. Zero on success.
 */
static int alloc_buffers(int32_t **read_buf) {
    uint32_t transfer_size = ALIGN(buffer_channel_aligned_size, 8);
    *read_buf = mem_alloc(channels * transfer_size);
    for (int i = 0; i < channels; i++) {
        mram_read(&input_buffer[i * buffer_channel_aligned_size],
                      &(*read_buf)[i * buffer_channel_usable_length], transfer_size);
    }

    // chAM
    transfer_size = ALIGN(channels * (bit_dim + 1) * sizeof(uint32_t), 8);
    chAM = mem_alloc(transfer_size);
    uint32_t transfer_chunks = transfer_size / MRAM_MAX_READ_SIZE;
    uint32_t transfer_remainder = transfer_size % MRAM_MAX_READ_SIZE;

    for (int i = 0; i < transfer_chunks; i++) {
        mram_read(&mram_chAM[i * MRAM_MAX_READ_SIZE], (uint8_t *)chAM + i * MRAM_MAX_READ_SIZE, MRAM_MAX_READ_SIZE);
    }

    if (transfer_remainder != 0) {
        mram_read(&mram_chAM[transfer_chunks * MRAM_MAX_READ_SIZE],
                  &((uint8_t *)chAM)[transfer_chunks * MRAM_MAX_READ_SIZE], ALIGN(transfer_remainder, 8));
    }

    // iM
    transfer_size = ALIGN(im_length * (bit_dim + 1) * sizeof(uint32_t), 8);
    iM = mem_alloc(transfer_size);
    transfer_chunks = transfer_size / MRAM_MAX_READ_SIZE;
    transfer_remainder = transfer_size % MRAM_MAX_READ_SIZE;

    for (int i = 0; i < transfer_chunks; i++) {
        mram_read(&mram_iM[i * MRAM_MAX_READ_SIZE], (uint8_t *)iM + i * MRAM_MAX_READ_SIZE, MRAM_MAX_READ_SIZE);
    }

    if (transfer_remainder != 0) {
        mram_read(&mram_iM[transfer_chunks * MRAM_MAX_READ_SIZE],
                  (uint8_t *)iM + transfer_chunks * MRAM_MAX_READ_SIZE, ALIGN(transfer_remainder, 8));
    }

    // aM_32
    transfer_size = ALIGN(n * (bit_dim + 1) * sizeof(uint32_t), 8);
    aM_32 = mem_alloc(transfer_size);
    transfer_chunks = transfer_size / MRAM_MAX_READ_SIZE;
    transfer_remainder = transfer_size % MRAM_MAX_READ_SIZE;

    for (int i = 0; i < transfer_chunks; i++) {
        mram_read(&mram_aM_32[i * MRAM_MAX_READ_SIZE], (uint8_t *)aM_32 + i * MRAM_MAX_READ_SIZE, MRAM_MAX_READ_SIZE);
    }

    if (transfer_remainder != 0) {
        mram_read(&mram_aM_32[transfer_chunks * MRAM_MAX_READ_SIZE],
                  (uint8_t *)aM_32 + transfer_chunks * MRAM_MAX_READ_SIZE, ALIGN(transfer_remainder, 8));
    }

    // uint32_t chHV[channels + 1][bit_dim + 1];

    return 0;
}

/**
 * @breif Run HDC algorithm on host
 *
 * @return Non-zero on failure.
 */
static int dpu_hdc() {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t *q = mem_alloc((bit_dim + 1) * sizeof(uint32_t));
    uint32_t *q_N = mem_alloc((bit_dim + 1) * sizeof(uint32_t));
    int32_t *quantized_buffer = mem_alloc(channels * sizeof(uint32_t));
    chHV = mem_alloc((channels + 1) * (bit_dim + 1) * sizeof(uint32_t));

    memset(q, 0, (bit_dim + 1) * sizeof(uint32_t));
    memset(q_N, 0, (bit_dim + 1) * sizeof(uint32_t));
    memset(quantized_buffer, 0, channels * sizeof(uint32_t));
    memset(chHV, 0, (channels + 1) * (bit_dim + 1) * sizeof(uint32_t));

    int class;

    int ret = 0;

    __dma_aligned int32_t *read_buf;

    ret = alloc_buffers(&read_buf);
    if (ret != 0) {
        return ret;
    }

    for(int ix = 0; ix < buffer_channel_length; ix += n) {

        for(int z = 0; z < n; z++) {

            for(int j = 0; j < channels; j++) {
                if (ix + z < buffer_channel_usable_length) {
                    int ind = A2D1D(buffer_channel_usable_length, j, ix + z);
                    quantized_buffer[j] = read_buf[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the N-gram.
            // N.B. if N = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                CYCLES_COUNT_START(&counter);
                compute_N_gram(quantized_buffer, iM, chAM, q);
                CYCLES_COUNT_FINISH(counter, &compute_N_gram_cycles);
            } else {
                CYCLES_COUNT_START(&counter);
                compute_N_gram(quantized_buffer, iM, chAM, q_N);
                CYCLES_COUNT_FINISH(counter, &compute_N_gram_cycles);

                CYCLES_COUNT_START(&counter);
                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                int32_t shifted_q;
                overflow = q[0] & MASK;
                for(int i = 1; i < bit_dim; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & MASK;
                    shifted_q = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ shifted_q;
                }

                old_overflow = overflow;
                overflow = (q[bit_dim] >> 16) & MASK;
                shifted_q = (q[bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[bit_dim] = q_N[bit_dim] ^ shifted_q;

                shifted_q = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ shifted_q;
                CYCLES_COUNT_FINISH(counter, &bit_mod_cycles);
            }
        }
        CYCLES_COUNT_START(&counter);
        // Classifies the new N-gram through the Associative Memory matrix.
        class = associative_memory_32bit(q, aM_32);
        CYCLES_COUNT_FINISH(counter, &associative_memory_cycles);
        printf("%d\n", class);

    }

    return ret;
}

int main() {
    uint8_t idx = me();
    // (void) idx;

    dbg_printf("DPU starting, tasklet %d\n", idx);

    /* Initialize cycle counters */
    perfcounter_config(COUNTER_CONFIG, true);

    int ret = 0;

    if (buffer_channel_length > 0) {
        ret = dpu_hdc();
    } else {
        printf("No work to do\n");
    }

    perfcounter_t total_cycles = perfcounter_get();
    dbg_printf("Tasklet %d: completed in %ld cycles\n", idx, total_cycles);
    dbg_printf("compute_N_gram_cycles used %ld cycles (%f%%)\n", compute_N_gram_cycles,
               (double)compute_N_gram_cycles / total_cycles);
    dbg_printf("associative_memory_cycles used %ld cycles (%f%%)\n", associative_memory_cycles,
               (double)associative_memory_cycles / total_cycles);
    dbg_printf("bit_mod used %ld cycles (%f%%)\n", bit_mod_cycles,
               (double)bit_mod_cycles / total_cycles);
    (void)total_cycles;

    return ret;
}
