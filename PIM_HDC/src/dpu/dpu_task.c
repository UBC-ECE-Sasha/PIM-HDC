#include <mram.h>
#include <seqread.h>
#include <defs.h>
#include <perfcounter.h>
#include <stdio.h>
#include "alloc.h"

#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "data.h"

#define READ_INTS_BUFFER_SIZE 2

__host __mram_ptr uint8_t * input_buffer;

/**
 * Run HDC algorithm on host
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

    __dma_aligned int32_t read_buf[READ_INTS_BUFFER_SIZE];

    for(int ix = 0; ix < NUMBER_OF_INPUT_SAMPLES; ix = ix + N) {

        for(int z = 0; z < N; z++) {

            for(int j = 0; j < CHANNELS; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < NUMBER_OF_INPUT_SAMPLES) {
                    // Read from input_buffer into quantized_buffer

                    // Case where reading 2 int32s would read past buffer
                    if (ix + z == (NUMBER_OF_INPUT_SAMPLES - 1)) {
                        mram_read(&input_buffer[(j * NUMBER_OF_INPUT_SAMPLES) + ix + z - 1], read_buf, READ_INTS_BUFFER_SIZE * sizeof(int32_t));
                        quantized_buffer[j] = read_buf[1];
                    } else {
                        mram_read(&input_buffer[(j * NUMBER_OF_INPUT_SAMPLES) + ix + z], read_buf, READ_INTS_BUFFER_SIZE * sizeof(int32_t));
                        quantized_buffer[j] = read_buf[0];
                    }
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

    return 0;
}

int main() {
    uint8_t idx = me();

    printf("DPU starting, tasklet %d\n", idx);

    perfcounter_config(COUNT_CYCLES, true);

    dpu_hdc();

    printf("Tasklet %d: completed in %ld cycles\n", idx, perfcounter_get());

    return 0;
}
