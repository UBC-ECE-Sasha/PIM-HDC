#include <mram.h>
#include <seqread.h>
#include <defs.h>
#include <perfcounter.h>
#include <stdio.h>
#include <errno.h>
#include <alloc.h>
#include <built_ins.h>
#include <handshake.h>
#include <string.h>
#include <barrier.h>

#include "global_dpu.h"
#include "common.h"
#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "cycle_counter.h"

#define MASK 1
#define MRAM_MAX_READ_SIZE 2048

#define TASKLET_SETUP 0

BARRIER_INIT(start_barrier, NR_TASKLETS);
BARRIER_INIT(finish_barrier, NR_TASKLETS);

__host __mram_ptr int8_t *input_buffer;
__host __mram_ptr uint8_t *mram_chAM;
__host __mram_ptr uint8_t *mram_iM;
__host __mram_ptr uint8_t *mram_aM_32;

__dma_aligned uint32_t *chAM;
__dma_aligned uint32_t *iM;
__dma_aligned uint32_t *aM_32;
__dma_aligned int32_t *read_buf;

__host uint32_t buffer_channel_aligned_size;
__host uint32_t buffer_channel_usable_length;
__host int32_t dimension;
__host int32_t channels;
__host int32_t bit_dim;
__host int32_t number_of_input_samples;
__host int32_t n;
__host uint32_t dpu_id;
__host int32_t im_length;
__host dpu_input_data dpu_data;

__host __mram_ptr uint8_t *output_buffer;
__host uint32_t output_buffer_length;
__dma_aligned int32_t *output;

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
 * @brief Fill @p buf with data from @p mram_ptr
 *
 * @param[out] buf       Buffer to transfer @p transfer_size bytes to
 * @param[in]  mram_ptr  Buffer to transfer @p transfer_size bytes from
 */
static void alloc_chunks(uint32_t **buf, __mram_ptr uint8_t * mram_ptr, uint32_t transfer_size) {
    uint32_t aligned_transfer_size = ALIGN(transfer_size, 8);
    *buf = mem_alloc(transfer_size);
    uint32_t transfer_chunks = aligned_transfer_size / MRAM_MAX_READ_SIZE;
    uint32_t transfer_remainder = aligned_transfer_size % MRAM_MAX_READ_SIZE;

    uint8_t * buf_loc = NULL;
    for (int i = 0; i < transfer_chunks; i++) {
        buf_loc = &((uint8_t *)(*buf))[i * MRAM_MAX_READ_SIZE];
        mram_read(&mram_ptr[i * MRAM_MAX_READ_SIZE], buf_loc, MRAM_MAX_READ_SIZE);
    }

    if (transfer_remainder != 0) {
        buf_loc = &((uint8_t *)(*buf))[transfer_chunks * MRAM_MAX_READ_SIZE];
        mram_read(&mram_ptr[transfer_chunks * MRAM_MAX_READ_SIZE], buf_loc, ALIGN(transfer_remainder, 8));
    }
}

/**
 * @brief Fill @p read_buf with data from @p input_buffer, populate globals
 *
 * @return                 @p ENOMEM on failure. Zero on success.
 */
static int alloc_buffers(uint32_t out_size) {
    output = mem_alloc(out_size);

    uint32_t transfer_size = ALIGN(buffer_channel_aligned_size, 8);
    read_buf = mem_alloc(channels * transfer_size);
    for (int i = 0; i < channels; i++) {
        mram_read(&input_buffer[i * buffer_channel_aligned_size],
                      &read_buf[i * buffer_channel_usable_length], transfer_size);
    }

    // chAM
    transfer_size = channels * (bit_dim + 1) * sizeof(uint32_t);
    alloc_chunks(&chAM, mram_chAM, transfer_size);

    // iM
    transfer_size = im_length * (bit_dim + 1) * sizeof(uint32_t);
    alloc_chunks(&iM, mram_iM, transfer_size);

    // aM_32
    transfer_size = n * (bit_dim + 1) * sizeof(uint32_t);
    alloc_chunks(&aM_32, mram_aM_32, transfer_size);

    return 0;
}

/**
 * @breif Run HDC algorithm on host
 *
 * @return Non-zero on failure.
 */
static int dpu_hdc(int32_t *result, uint32_t result_offset, uint32_t task_begin, uint32_t task_end) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;

    // No room on heap
    // uint32_t *q = mem_alloc((bit_dim + 1) * sizeof(uint32_t));
    // uint32_t *q_N = mem_alloc((bit_dim + 1) * sizeof(uint32_t));
    // int32_t *quantized_buffer = mem_alloc(channels * sizeof(uint32_t));

    uint32_t q[MAX_BIT_DIM+1] = {0};
    uint32_t q_N[MAX_BIT_DIM+1] = {0};
    int32_t quantized_buffer[MAX_CHANNELS] = {0};

    int ret = 0;
    int result_num = 0;

    for(int ix = task_begin; ix < task_end; ix += n) {

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
        result[result_offset + result_num] = associative_memory_32bit(q, aM_32);
        CYCLES_COUNT_FINISH(counter, &associative_memory_cycles);

        dbg_printf("%u: result[%d] = %d\n", me(), result_offset + result_num, result[result_offset + result_num]);

        result_num++;
    }

    dbg_printf("%u: %d results\n", me(), result_num);

    return ret;
}

int main() {
    uint8_t idx = me();

    printf("DPU starting, tasklet %u / %u\n", idx, NR_TASKLETS);

    /* Initialize cycle counters */
    perfcounter_config(COUNTER_CONFIG, true);

    int ret = 0;

    uint32_t out_size = ALIGN(output_buffer_length * sizeof(int32_t), 8);
    if (idx == TASKLET_SETUP) {
        if ((ret = alloc_buffers(out_size)) != 0) {
            return ret;
        }
    }

    barrier_wait(&start_barrier);

    if ((dpu_data.task_end[idx] - dpu_data.task_begin[idx]) > 0) {
        ret = dpu_hdc(output, dpu_data.idx_offset[idx], dpu_data.task_begin[idx], dpu_data.task_end[idx]);
    } else {
        printf("%u: No work to do\n", idx);
    }

    barrier_wait(&finish_barrier);

    mram_write(output, output_buffer, out_size);

    perfcounter_t total_cycles = perfcounter_get();
    printf("Tasklet %d: completed in %ld cycles\n", idx, total_cycles);
    printf("compute_N_gram_cycles used %ld cycles (%f%%)\n", compute_N_gram_cycles,
               (double)compute_N_gram_cycles / total_cycles);
    printf("associative_memory_cycles used %ld cycles (%f%%)\n", associative_memory_cycles,
               (double)associative_memory_cycles / total_cycles);
    printf("bit_mod used %ld cycles (%f%%)\n", bit_mod_cycles,
               (double)bit_mod_cycles / total_cycles);

    return ret;
}
