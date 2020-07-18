#include "associative_memory.h"
#include "aux_functions.h"
#include "common.h"
#include "cycle_counter.h"
#include "global_dpu.h"
#include "init.h"

#include <alloc.h>
#include <barrier.h>
#include <built_ins.h>
#include <defs.h>
#include <errno.h>
#include <handshake.h>
#include <mram.h>
#include <perfcounter.h>
#include <seqread.h>
#include <stdio.h>
#include <string.h>

#define MASK 1
#define MRAM_MAX_READ_SIZE 2048

#define TASKLET_SETUP 0

BARRIER_INIT(start_barrier, NR_TASKLETS);
BARRIER_INIT(finish_barrier, NR_TASKLETS);

// WRAM
__host dpu_input_data dpu_data;
__host dpu_hdc_vars hd;

__host int32_t read_buf[MAX_INPUT];
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
 * @brief Run HDC algorithm
 *
 * @return Non-zero on failure.
 */

/**
 * @brief Run HDC algorithm
 * @param[out] result         Buffer to place results in
 * @param[out] result_offset  Offset to start placing results from
 * @param[in] task_begin      Position to start task from
 * @param[in] task_end        Position to end task at
 *
 * @return                    Non-zero on failure.
 */
static int
dpu_hdc(int32_t *result, uint32_t result_offset, uint32_t task_begin, uint32_t task_end) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;

    uint32_t q[MAX_BIT_DIM + 1] = {0};
    uint32_t q_N[MAX_BIT_DIM + 1] = {0};
    int32_t quantized_buffer[MAX_CHANNELS] = {0};

    int ret = 0;
    int result_num = 0;

    for (int ix = task_begin; ix < task_end; ix += hd.n) {

        for (int z = 0; z < hd.n; z++) {

            for (int j = 0; j < hd.channels; j++) {
                if (ix + z < dpu_data.buffer_channel_usable_length) {
                    int ind = A2D1D(dpu_data.buffer_channel_usable_length, j, ix + z);
                    quantized_buffer[j] = read_buf[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the N-gram.
            // N.B. if N = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                CYCLES_COUNT_START(&counter);
                compute_N_gram(quantized_buffer, hd.iM, hd.chAM, q);
                CYCLES_COUNT_FINISH(counter, &compute_N_gram_cycles);
            } else {
                CYCLES_COUNT_START(&counter);
                compute_N_gram(quantized_buffer, hd.iM, hd.chAM, q_N);
                CYCLES_COUNT_FINISH(counter, &compute_N_gram_cycles);

                CYCLES_COUNT_START(&counter);
                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                int32_t shifted_q;
                overflow = q[0] & MASK;
                for (int i = 1; i < hd.bit_dim; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & MASK;
                    shifted_q = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ shifted_q;
                }

                old_overflow = overflow;
                overflow = (q[hd.bit_dim] >> 16) & MASK;
                shifted_q = (q[hd.bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[hd.bit_dim] = q_N[hd.bit_dim] ^ shifted_q;

                shifted_q = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ shifted_q;
                CYCLES_COUNT_FINISH(counter, &bit_mod_cycles);
            }
        }

        CYCLES_COUNT_START(&counter);
        // Classifies the new N-gram through the Associative Memory matrix.
        result[result_offset + result_num] = associative_memory_32bit(q, hd.aM_32);
        CYCLES_COUNT_FINISH(counter, &associative_memory_cycles);

        dbg_printf("%u: result[%d] = %d\n", me(), result_offset + result_num,
                   result[result_offset + result_num]);

        result_num++;
    }

    dbg_printf("%u: %d results\n", me(), result_num);

    return ret;
}

int
main() {
    uint8_t idx = me();

    printf("DPU starting, tasklet %u / %u\n", idx, NR_TASKLETS);

    /* Initialize cycle counters */
    perfcounter_config(COUNTER_CONFIG, true);

    int ret = 0;

    uint32_t out_size = ALIGN(dpu_data.output_buffer_length * sizeof(int32_t), 8);
    if (idx == TASKLET_SETUP) {
        output = mem_alloc(out_size);
    }

    barrier_wait(&start_barrier);

    if ((dpu_data.task_end[idx] - dpu_data.task_begin[idx]) > 0) {
        ret = dpu_hdc(output, dpu_data.idx_offset[idx], dpu_data.task_begin[idx],
                      dpu_data.task_end[idx]);
    } else {
        printf("%u: No work to do\n", idx);
    }

    barrier_wait(&finish_barrier);

    if (idx == TASKLET_SETUP) {
        (void) memcpy(read_buf, output, out_size);
    }

    perfcounter_t total_cycles = perfcounter_get();
    printf("Tasklet %d: completed in %ld cycles\n", idx, total_cycles);
    printf("compute_N_gram_cycles used %ld cycles (%f%%)\n", compute_N_gram_cycles,
           (double) compute_N_gram_cycles / total_cycles);
    printf("associative_memory_cycles used %ld cycles (%f%%)\n", associative_memory_cycles,
           (double) associative_memory_cycles / total_cycles);
    printf("bit_mod used %ld cycles (%f%%)\n", bit_mod_cycles,
           (double) bit_mod_cycles / total_cycles);

    return ret;
}
