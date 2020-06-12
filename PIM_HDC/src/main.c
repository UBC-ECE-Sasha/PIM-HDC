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

/* Must be <= NUMBER_OF_INPUT_SAMPLES */
#define TEST_SAMPLE_SIZE NUMBER_OF_INPUT_SAMPLES

/**
 * @brief Prepare the DPU context and upload the program to the DPU.
 *
 * @param[in] data_set Quantized data buffer
 * @return             Non-zero on failure.
 */
static int prepare_dpu(in_buffer data_set) {

    struct dpu_set_t dpus;
    struct dpu_set_t dpu;

    uint32_t dpu_id = 0;

    uint32_t input_buffer_start = MEGABYTE(1);

    /* Section of buffer for one channel, without samples not divisible by N */
    uint32_t samples = TEST_SAMPLE_SIZE / NR_DPUS;
    /* Remove samples not divisible by N */
    samples -= samples % N;
    /* Extra data for last DPU */
    uint32_t extra_data = TEST_SAMPLE_SIZE - (samples * NR_DPUS);

    dbg_printf("samples = %d / %d = %d\n", TEST_SAMPLE_SIZE, NR_DPUS, samples);
    if (samples > SAMPLE_SIZE_MAX) {
        fprintf(stderr, "samples per dpu (%u) cannot be greater than SAMPLE_SIZE_MAX = (%d)\n",
                samples, SAMPLE_SIZE_MAX);
    }

    uint32_t buffer_channel_length = samples;

    /* + N to account for + z in algorithm (unless 1 DPU) */
    uint32_t buffer_channel_usable_length = buffer_channel_length;
    if (NR_DPUS > 1) {
        buffer_channel_usable_length += N;
    }

    uint32_t aligned_buffer_size = ALIGN(buffer_channel_usable_length * sizeof(int32_t), 8);
    uint32_t buff_offset = 0;

    // Allocate DPUs
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpus));

    DPU_FOREACH(dpus, dpu) {
        dbg_printf("DPU %d\n", dpu_id);
        dbg_printf("buff_offset = %d\n", buff_offset);
        /* Modified only for last DPU in case uneven */
        if ((dpu_id == (NR_DPUS - 1)) && (NR_DPUS > 1) && (extra_data != 0)) {
            buffer_channel_length = buffer_channel_length + extra_data;
            buffer_channel_usable_length = buffer_channel_length; /* No N on last in algorithm */
            aligned_buffer_size = ALIGN(buffer_channel_length * sizeof(int32_t), 8);
        }

        DPU_ASSERT(dpu_load(dpu, DPU_PROGRAM, NULL));
        DPU_ASSERT(dpu_copy_to(dpu, "input_buffer", 0, &input_buffer_start, sizeof(input_buffer_start)));
        DPU_ASSERT(dpu_copy_to(dpu, "buffer_channel_length", 0, &buffer_channel_length, sizeof(buffer_channel_length)));
        DPU_ASSERT(dpu_copy_to(dpu, "buffer_channel_offset", 0, &aligned_buffer_size, sizeof(aligned_buffer_size)));
        DPU_ASSERT(dpu_copy_to(dpu, "buffer_channel_usable_length", 0, &buffer_channel_usable_length, sizeof(buffer_channel_usable_length)));

        if (buffer_channel_length > 0) {
            /* Copy each channel into DPU array */
            for (uint32_t i = 0; i < CHANNELS; i++) {
                uint8_t * ta = (uint8_t *)(&data_set.buffer[(i * NUMBER_OF_INPUT_SAMPLES) + buff_offset]);
                /* Check output dataset */
                dbg_printf("OUTPUT data_set[%d] (%u samples, %u usable):\n", i, buffer_channel_length, buffer_channel_usable_length);
                for (uint32_t j = 0; j < buffer_channel_length; j++) dbg_printf("td[%d]=%d\n", j, ((int32_t *)ta)[j]);
                DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, input_buffer_start + i*aligned_buffer_size, ta, aligned_buffer_size, 0));
            }
        }
        buff_offset += buffer_channel_length;
        dpu_id++;
    }

    int ret = dpu_launch(dpus, DPU_SYNCHRONOUS);

    DPU_FOREACH(dpus, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    if (ret != 0) {
        fprintf(stderr, "%s\nReturn code (%d)\n", "Failure occurred during DPU run.", ret);
    }

    // Deallocate the DPUs
    DPU_ASSERT(dpu_free(dpus));

    return ret;
}

/**
 * @brief Prepare the DPU context and upload the program to the DPU.
 *
 * @param[in] data_set Quantized data buffer
 * @return             Non-zero on failure.
 */
static int host_hdc(int32_t * data_set) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[BIT_DIM + 1] = {0};
    uint32_t q_N[BIT_DIM + 1] = {0};
    int class = 0;

    int32_t quantized_buffer[CHANNELS];

    for(int ix = 0; ix < TEST_SAMPLE_SIZE; ix += N) {

        for(int z = 0; z < N; z++) {

            for(int j = 0; j < CHANNELS; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < TEST_SAMPLE_SIZE) {
                    quantized_buffer[j] = data_set[(j * NUMBER_OF_INPUT_SAMPLES) + ix + z];
                    dbg_printf("quantized_buffer[%d] = data_set[%d][%d + %d] = %d\n", j, j, ix, z, quantized_buffer[j]);
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
        class = associative_memory_32bit(q, aM_32);

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
