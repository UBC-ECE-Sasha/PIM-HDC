#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>

#include <dpu.h>
#include <dpu_memory.h>
#include <dpu_log.h>

#include "associative_memory.h"
#include "aux_functions.h"
#include "init.h"
#include "host_only.h"
#include "common.h"

// Original array lengths
// double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
// uint32_t chAM[CHANNELS][BIT_DIM + 1];
// uint32_t iM[IM_LENGTH][BIT_DIM + 1];
// uint32_t aM_32[N][BIT_DIM + 1];

#define DPU_PROGRAM "src/dpu/hdc.dpu"

static dpu_input_data setup_dpu_data(uint32_t buffer_channel_length) {
    /* Computations must be n divisible unless extra at end */
    int32_t num_computations = buffer_channel_length / n;
    int32_t remaining_computations = buffer_channel_length % n;

    dpu_input_data input;

    for (uint8_t idx = 0; idx < NR_TASKLETS; idx++) {
        uint32_t task_begin = 0;
        uint32_t task_end = 0;

        if (num_computations >= (idx + 1)) {
            if (num_computations < NR_TASKLETS) {
                task_begin = idx * n;
                task_end = task_begin + n;
            } else {
                uint32_t split_computations = (num_computations / NR_TASKLETS) * n;
                task_begin = idx * split_computations;
                task_end = task_begin + split_computations;

                if ((idx + 1) == NR_TASKLETS) {
                    uint32_t task_extra = (num_computations % NR_TASKLETS);
                    task_end += remaining_computations + (task_extra * n);
                }
            }
        }

        input.task_begin[idx] = task_begin;
        input.task_end[idx] = task_end;

        uint32_t task_samples = input.task_end[0] - input.task_begin[0];
        input.idx_offset[idx] = (task_samples / n) * idx;

        dbg_printf("%u: idx_offset = %u\n", idx, input.idx_offset[idx]);
        dbg_printf("%u: task_end = %u, task_begin = %u\n", idx, task_end, task_begin);
    }

    return input;
}

/**
 * @brief Prepare the DPU context and upload the program to the DPU.
 *
 * @param[in] data_set Quantized data buffer
 * @return             Non-zero on failure.
 */
static int prepare_dpu(int32_t * data_set, int32_t *results) {

    struct dpu_set_t dpus;
    struct dpu_set_t dpu;

    uint32_t dpu_id = 0;

    uint32_t input_buffer_start = MEGABYTE(1);
    uint32_t mram_buffers_loc = input_buffer_start;

    /* Section of buffer for one channel, without samples not divisible by n */
    uint32_t samples = number_of_input_samples / NR_DPUS;
    /* Remove samples not divisible by n */
    uint32_t chunk_size = samples - (samples % n);
    /* Extra data for last DPU */
    uint32_t extra_data = number_of_input_samples - (chunk_size * NR_DPUS);

    dbg_printf("chunk_size = %d / %d = %d\n", number_of_input_samples, NR_DPUS, chunk_size);
    if (chunk_size > SAMPLE_SIZE_MAX) {
        fprintf(stderr, "chunk_size per dpu (%u) cannot be greater than SAMPLE_SIZE_MAX = (%d)\n",
                chunk_size, SAMPLE_SIZE_MAX);
    }

    uint32_t buffer_channel_length = chunk_size;

    /* + n to account for + z in algorithm (unless 1 DPU) */
    uint32_t buffer_channel_usable_length = buffer_channel_length;
    if (NR_DPUS > 1) {
        buffer_channel_usable_length += n;
    }

    uint32_t aligned_buffer_size = ALIGN(buffer_channel_usable_length * sizeof(int32_t), 8);
    uint32_t buff_offset = 0;

    /* output */

    /* First entry for each tasklet contains length */
    uint32_t output_buffer_loc[NR_DPUS];
    uint32_t output_buffer_length[NR_DPUS];
    output_buffer_length[0] = (buffer_channel_length / n);

    // Allocate DPUs
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpus));

    DPU_FOREACH(dpus, dpu) {
        dbg_printf("DPU %d\n", dpu_id);
        DPU_ASSERT(dpu_load(dpu, DPU_PROGRAM, NULL));

        dpu_input_data input = setup_dpu_data(buffer_channel_length);

        // Variables in WRAM
        DPU_ASSERT(dpu_copy_to(dpu, "buffer_channel_aligned_size", 0, &aligned_buffer_size, sizeof(aligned_buffer_size)));
        DPU_ASSERT(dpu_copy_to(dpu, "buffer_channel_usable_length", 0, &buffer_channel_usable_length, sizeof(buffer_channel_usable_length)));
        DPU_ASSERT(dpu_copy_to(dpu, "dimension", 0, &dimension, sizeof(dimension)));
        DPU_ASSERT(dpu_copy_to(dpu, "channels", 0, &channels, sizeof(channels)));
        DPU_ASSERT(dpu_copy_to(dpu, "bit_dim", 0, &bit_dim, sizeof(bit_dim)));
        DPU_ASSERT(dpu_copy_to(dpu, "n", 0, &n, sizeof(n)));
        DPU_ASSERT(dpu_copy_to(dpu, "im_length", 0, &im_length, sizeof(im_length)));
        DPU_ASSERT(dpu_copy_to(dpu, "dpu_id", 0, &dpu_id, sizeof(dpu_id)));
        DPU_ASSERT(dpu_copy_to(dpu, "dpu_data", 0, &input, sizeof(input)));

        // Variables in MRAM

        // chAM;
        uint32_t transfer_size = ALIGN(channels * (bit_dim + 1) * sizeof(uint32_t), 8);
        DPU_ASSERT(dpu_copy_to(dpu, "mram_chAM", 0, &mram_buffers_loc, sizeof(mram_buffers_loc)));
        DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, mram_buffers_loc, (uint8_t *)chAM, transfer_size));
        mram_buffers_loc += transfer_size;

        // iM;
        transfer_size = ALIGN(im_length * (bit_dim + 1) * sizeof(uint32_t), 8);
        DPU_ASSERT(dpu_copy_to(dpu, "mram_iM", 0, &mram_buffers_loc, sizeof(mram_buffers_loc)));
        DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, mram_buffers_loc, (uint8_t *)iM, transfer_size));
        mram_buffers_loc += transfer_size;

        // aM_32;
        transfer_size = ALIGN(n * (bit_dim + 1) * sizeof(uint32_t), 8);
        DPU_ASSERT(dpu_copy_to(dpu, "mram_aM_32", 0, &mram_buffers_loc, sizeof(mram_buffers_loc)));
        DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, mram_buffers_loc, (uint8_t *)aM_32, transfer_size));
        mram_buffers_loc += transfer_size;

        // dataset:
        DPU_ASSERT(dpu_copy_to(dpu, "input_buffer", 0, &mram_buffers_loc, sizeof(mram_buffers_loc)));
        if (buffer_channel_length > 0) {
            /* Copy each channel into DPU array */
            for (int i = 0; i < channels; i++) {
                uint8_t * ta = (uint8_t *)(&data_set[(i * number_of_input_samples) + buff_offset]);
                /* Check input dataset */
                dbg_printf("INPUT data_set[%d] (%u bytes) (%u chunk_size, %u usable):\n", i, aligned_buffer_size, buffer_channel_length, buffer_channel_usable_length);
                DPU_ASSERT(dpu_copy_to_mram(dpu.dpu, mram_buffers_loc, ta, aligned_buffer_size));
                mram_buffers_loc += aligned_buffer_size;
            }
        }

        // Output
        output_buffer_loc[dpu_id] = mram_buffers_loc;
        dbg_printf("OUTPUT output_buffer_length (%u):\n", output_buffer_length[dpu_id]);
        DPU_ASSERT(dpu_copy_to(dpu, "output_buffer_length", 0, &output_buffer_length[dpu_id], sizeof(output_buffer_length[dpu_id])));
        DPU_ASSERT(dpu_copy_to(dpu, "output_buffer", 0, &mram_buffers_loc, sizeof(mram_buffers_loc)));

        mram_buffers_loc = input_buffer_start;
        buff_offset += buffer_channel_length;
        dpu_id++;

        /* Modified only for last DPU in case uneven */
        if ((dpu_id == (NR_DPUS - 1)) && (NR_DPUS > 1) && (extra_data != 0)) {
            /* Input */
            buffer_channel_length = buffer_channel_length + extra_data;
            buffer_channel_usable_length = buffer_channel_length; /* No n on last in algorithm */
            aligned_buffer_size = ALIGN(buffer_channel_length * sizeof(int32_t), 8);

            /* Output */
            uint32_t extra_result = (buffer_channel_length % n) != 0;
            output_buffer_length[dpu_id] = (buffer_channel_length / n) + extra_result;
        } else if (dpu_id < NR_DPUS) {
            output_buffer_length[dpu_id] = output_buffer_length[0];
        }
    }

    int ret = dpu_launch(dpus, DPU_SYNCHRONOUS);

    dpu_id = 0;
    uint32_t * output_buffer[NR_DPUS];
    DPU_FOREACH(dpus, dpu) {
        uint32_t out_len = ALIGN(sizeof(int32_t) * output_buffer_length[dpu_id], 8);
        output_buffer[dpu_id] = malloc(out_len);
        if (output_buffer[dpu_id] == NULL) {
            nomem();
        }

        DPU_ASSERT(dpu_copy_from_mram(dpu.dpu, (uint8_t *)output_buffer[dpu_id], output_buffer_loc[dpu_id], out_len));

        printf("------DPU %d Logs------\n", dpu_id);
        DPU_ASSERT(dpu_log_read(dpu, stdout));

        dpu_id++;
    }

    uint32_t result_num = 0;
    for (dpu_id = 0; dpu_id < NR_DPUS; dpu_id++) {
        for (uint32_t j = 0; j < output_buffer_length[dpu_id]; j++) {
            results[result_num++] = output_buffer[dpu_id][j];
        }
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
static int host_hdc(int32_t * data_set, int32_t *results) {
    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[bit_dim + 1];
    uint32_t q_N[bit_dim + 1];
    int32_t quantized_buffer[channels];

    int result_num = 0;

    memset(q, 0, (bit_dim + 1) * sizeof(uint32_t));
    memset(q_N, 0, (bit_dim + 1) * sizeof(uint32_t));
    memset(quantized_buffer, 0, channels * sizeof(uint32_t));

    for(int ix = 0; ix < number_of_input_samples; ix += n) {

        for(int z = 0; z < n; z++) {

            for(int j = 0; j < channels; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < number_of_input_samples) {
                    int ind = A2D1D(number_of_input_samples, j, ix + z);
                    quantized_buffer[j] = data_set[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the n-gram.
            // N.B. if n = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, iM, chAM, q);
            } else {
                compute_N_gram(quantized_buffer, iM, chAM, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for(int i = 1; i < bit_dim; i++){
                    old_overflow = overflow;
                    overflow = q[i] & mask;
                    q[i] = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ q[i];
                }

                old_overflow = overflow;
                overflow = (q[bit_dim] >> 16) & mask;
                q[bit_dim] = (q[bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[bit_dim] = q_N[bit_dim] ^ q[bit_dim];

                q[0] = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ q[0];
            }
        }
        // classifies the new N-gram through the Associative Memory matrix.
        results[result_num++] = associative_memory_32bit(q, aM_32);
    }

    return 0;
}

static double time_difference(struct timeval * start, struct timeval * end) {
    double start_time = start->tv_sec + start->tv_usec / 1000000.0;
    double end_time = end->tv_sec + end->tv_usec / 1000000.0;
    return (end_time - start_time);
}

typedef struct hdc_data {
    int32_t *data_set;
    int32_t *results;
    uint32_t result_len;
    double execution_time;
} hdc_data;

typedef int (*hdc)(int32_t * data_set, int32_t *results);

static double run_hdc(hdc fn, hdc_data *data) {
    struct timeval start;
    struct timeval end;

    int ret = 0;

    uint8_t extra_result = (number_of_input_samples % n) != 0;
    data->result_len = (number_of_input_samples / n) + extra_result;
    uint32_t result_size = data->result_len * sizeof(int32_t);

    if ((data->results = malloc(result_size)) == NULL) {
        nomem();
    }

    gettimeofday(&start, NULL);
    ret = fn(data->data_set, data->results);
    gettimeofday(&end, NULL);

    data->execution_time = time_difference(&start, &end);

    return ret;
}

static int compare_results(hdc_data *dpu_data, hdc_data *host_data) {
    int ret = 0;

    printf("--- Compare --\n");
    for (uint32_t i = 0; i < host_data->result_len; i++) {
        if (host_data->results[i] != dpu_data->results[i]) {
            fprintf(stderr, "(host_results[%u] = %d) != (dpu_results[%u] = %d)\n",
                    i, host_data->results[i], i, dpu_data->results[i]);
            ret = 1;
        }
    }
    double time_diff = dpu_data->execution_time - host_data->execution_time;
    double percent_diff = dpu_data->execution_time / host_data->execution_time;
    printf("Host was %fs (%f x) faster than dpu\n", time_diff, percent_diff);

    return ret;
}

static void print_results(hdc_data *data) {
    for (uint32_t i = 0; i < data->result_len; i++) {
        printf("%d\n", data->results[i]);
    }
}

static void usage(FILE *stream, char const * exe_name) {
#ifdef DEBUG
	fprintf(stream, "**DEBUG BUILD**\n");
#endif

    fprintf(stream, "usage: %s -d -i <INPUT_FILE>\n", exe_name);
    fprintf(stream, "\td: use DPU\n");
    fprintf(stream, "\ti: input file\n");
    fprintf(stream, "\ts: show results\n");
    fprintf(stream, "\tt: test results\n");
    fprintf(stream, "\th: help message\n");
}

int main(int argc, char **argv) {
    bool use_dpu = false;
    bool show_results = false;
    bool test_results = false;
    int ret = 0;
    int dpu_ret = 0;
    int host_ret = 0;
    char const options[] = "dsthi:";
    char *input = NULL;

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch(opt) {
            case 'd':
                use_dpu = true;
                break;

            case 'i':
                input = optarg;
                break;

            case 's':
                show_results = true;
                break;

            case 't':
                test_results = true;
                break;

            case 'h':
                usage(stdout, argv[0]);
                return EXIT_SUCCESS;

            default:
                usage(stderr, argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (input == NULL) {
        fprintf(stderr, "Please add an input file\n");
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    double *test_set;
    ret = read_data(input, &test_set);
    if (ret != 0) {
        return ret;
    }

    uint32_t buffer_size = (sizeof(int32_t) * number_of_input_samples * channels);
    int32_t *data_set = malloc(buffer_size);
    if (data_set == NULL) {
        nomem();
    }

    quantize_set(test_set, data_set);

    hdc_data dpu_results = { .data_set = data_set };
    hdc_data host_results = { .data_set = data_set };

    if (use_dpu || test_results) {
        dpu_ret = run_hdc(prepare_dpu, &dpu_results);
    }

    if (!use_dpu || test_results) {
        host_ret = run_hdc(host_hdc, &host_results);
    }

    if (use_dpu || test_results) {
        printf("--- DPU --\n");
        if (show_results) {
            print_results(&dpu_results);
        }
        printf("DPU took %fs\n", dpu_results.execution_time);
    }

    if (!use_dpu || test_results) {
        printf("--- Host --\n");
        if (show_results) {
            print_results(&host_results);
        }
        printf("Host took %fs\n", host_results.execution_time);
    }

    if (test_results) {
        ret = compare_results(&dpu_results, &host_results);
    }

    free(data_set);
    free(test_set);
    free(host_results.results);
    free(dpu_results.results);
    free(chAM);
    free(aM_32);
    free(iM);

    return (ret + dpu_ret + host_ret);
}
