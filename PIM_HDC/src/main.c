#include <dpu.h>
#include <dpu_memory.h>
#include <dpu_log.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>

#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "data.h"
#include "host_only.h"
#include "common.h"

#define DPU_PROGRAM "src/dpu/hdc.dpu"

/**
 * Prepare the DPU context and upload the program to the DPU.
 */
static int prepare_dpu(in_buffer data_set) {

    struct dpu_set_t dpus;
    struct dpu_set_t dpu;

    uint32_t input_buffer_start = 1024 * 1024;

    // Allocate a DPU
    DPU_ASSERT(dpu_alloc(1, NULL, &dpus));

    DPU_FOREACH(dpus, dpu) {
        break;
    }

    DPU_ASSERT(dpu_load(dpu, DPU_PROGRAM, NULL));

    DPU_ASSERT(dpu_copy_to(dpu, "input_buffer", 0, &input_buffer_start, sizeof(input_buffer_start)));

    DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, input_buffer_start, (uint8_t *)data_set.buffer, data_set.buffer_size, 0));

    int ret = dpu_launch(dpu, DPU_SYNCHRONOUS);
    if (ret != 0) {
        DPU_ASSERT(dpu_free(dpus));
        return -1;
    }

    // Deallocate the DPUs
    DPU_FOREACH(dpus, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    DPU_ASSERT(dpu_free(dpus));

    return 0;
}

/**
 * Run HDC algorithm on host
 */
static int host_hdc(int32_t *data_set) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[BIT_DIM + 1] = {0};
    uint32_t q_N[BIT_DIM + 1] = {0};
    int class = 0;

    int32_t quantized_buffer[CHANNELS];

    for(int ix = 0; ix < NUMBER_OF_INPUT_SAMPLES; ix = ix + N) {

        for(int z = 0; z < N; z++) {

            for(int j = 0; j < CHANNELS; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < NUMBER_OF_INPUT_SAMPLES) {
                    // Original code:
                    // quantized_buffer[j] = round_to_int(TEST_SET[j][ix + z]);

                    quantized_buffer[j] = data_set[(j * NUMBER_OF_INPUT_SAMPLES) + ix + z];
                }
            }

            // Spatial and Temporal Encoder: computes the N-gram.
            // N.B. if N = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, iM, chAM, q);
            } else {
                compute_N_gram(quantized_buffer, iM, chAM, q_N);

                //Here the hypervector q is shifted by 1 position as permutation,
                //before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for(int i = 1; i < BIT_DIM; i++){

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


        //classifies the new N-gram through the Associative Memory matrix.
        if (N == 1) {
            class = associative_memory_32bit(q, aM_32);
        } else {
            class = associative_memory_32bit(q, aM_32);
        }

        printf("%d\n", class);

    }

    return 0;
}

static void usage(char const * exe_name) {
#ifdef DEBUG
	fprintf(stderr, "**DEBUG BUILD**\n");
#endif

    fprintf(stderr, "usage: %s -d [ -o <output> ]\n", exe_name);
    fprintf(stderr, "\td: use DPU\n");
    fprintf(stderr, "\to: redirect output to file\n");
}

int main(int argc, char **argv) {

    unsigned int use_dpu = 0;
    int ret = 0;
    char const options[] = "dho:";

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch(opt) {
            case 'd':
                use_dpu = 1;
                break;

            case 'h':
                usage(argv[0]);
                return EXIT_SUCCESS;

            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    uint32_t buffer_size = (sizeof(int32_t) * NUMBER_OF_INPUT_SAMPLES * CHANNELS);
    in_buffer data_set;

    data_set.buffer_size = ALIGN(buffer_size, 8);

    if ((data_set.buffer = malloc(data_set.buffer_size)) == NULL) {
        return EXIT_FAILURE;
    }

    quantize_set(TEST_SET, data_set.buffer);

    if (use_dpu) {
        ret = prepare_dpu(data_set);
    } else {
        ret = host_hdc(data_set.buffer);
    }

    free(data_set.buffer);

    return ret;
}
