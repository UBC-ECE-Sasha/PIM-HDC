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

typedef struct dpu_runtime {
    double execution_time_copy_in;
    double execution_time_launch;
    double execution_time_copy_out;
} dpu_runtime;

static double time_difference(struct timeval * start, struct timeval * end) {
    double start_time = start->tv_sec + start->tv_usec / 1000000.0;
    double end_time = end->tv_sec + end->tv_usec / 1000000.0;
    return (end_time - start_time);
}

static int setup_dpu_data(dpu_input_data *input, uint32_t buffer_channel_length, in_buffer *data_in, int32_t * data_set, uint32_t *buff_offset, uint32_t dpu_id) {
    /* Computations must be n divisible unless extra at end */
    int32_t num_computations = buffer_channel_length / hd.n;
    int32_t remaining_computations = buffer_channel_length % hd.n;

    for (uint8_t idx = 0; idx < NR_TASKLETS; idx++) {
        uint32_t task_begin = 0;
        uint32_t task_end = 0;

        if (num_computations < NR_TASKLETS) {
            if (num_computations >= (idx + 1)) {
                task_begin = idx * hd.n;
                task_end = task_begin + hd.n;
            } else if (remaining_computations > 0) {
                task_begin = idx * hd.n;
                task_end = task_begin + remaining_computations;
            }
        } else {
            uint32_t split_computations = (num_computations / NR_TASKLETS) * hd.n;
            task_begin = idx * split_computations;
            task_end = task_begin + split_computations;

            if ((idx + 1) == NR_TASKLETS) {
                uint32_t task_extra = (num_computations % NR_TASKLETS);
                task_end += remaining_computations + (task_extra * hd.n);
            }
        }

        input->task_begin[idx] = task_begin;
        input->task_end[idx] = task_end;

        uint32_t task_samples = input->task_end[0] - input->task_begin[0];
        input->idx_offset[idx] = (task_samples / hd.n) * idx;

        dbg_printf("%u: idx_offset = %u\n", idx, input->idx_offset[idx]);
        dbg_printf("%u: task_end = %u, task_begin = %u\n", idx, task_end, task_begin);
    }

    /* Input */
    if (dpu_id == NR_DPUS - 1) {
        /* No n on last in algorithm */
        input->buffer_channel_usable_length = buffer_channel_length;
    } else {
        input->buffer_channel_usable_length = buffer_channel_length + hd.n;
    }
    input->buffer_channel_aligned_size = ALIGN(buffer_channel_length * sizeof(int32_t), 8);

    if ((input->buffer_channel_usable_length * hd.channels) > SAMPLE_SIZE_MAX) {
        fprintf(stderr, "buffer_channel_usable_length * hd.channels (%u) cannot be greater than MAX_INPUT = (%d)\n",
                input->buffer_channel_usable_length * hd.channels, SAMPLE_SIZE_MAX);
        return -1;
    }

    /* Output */
    uint32_t extra_result = (buffer_channel_length % hd.n) != 0;
    input->output_buffer_length = (buffer_channel_length / hd.n) + extra_result;

    input->buffer_channel_length = buffer_channel_length;

    size_t sz_xfer = input->buffer_channel_usable_length * sizeof(int32_t);
    data_in->buffer_size = sz_xfer * hd.channels;
    if (input->buffer_channel_length > 0) {
        /* Copy each channel into DPU array */
        for (int i = 0; i < hd.channels; i++) {
            int32_t * ta = &data_set[(i * number_of_input_samples) + *buff_offset];
            dbg_printf("INPUT data_set[%d] (%u bytes) (%u chunk_size, %u usable):\n", i, input->buffer_channel_aligned_size, input->buffer_channel_length, input->buffer_channel_usable_length);
            (void) memcpy(&data_in->buffer[i * input->buffer_channel_usable_length], ta, sz_xfer);
        }
    }

    *buff_offset += input->buffer_channel_length;

    size_t total_xfer = 0;
    if (total_xfer > (MAX_INPUT * sizeof(int32_t))) {
        fprintf(stderr, "Error %lu is too large for read_buf[%d]\n", total_xfer / sizeof(int32_t), MAX_INPUT);
        return -1;
    }

    return 0;
}

static void calculate_buffer_lengths(uint32_t buffer_channel_lengths[NR_DPUS]) {
    /* Section of buffer for one channel, without samples not divisible by n */
    uint32_t samples = number_of_input_samples / NR_DPUS;
    /* Remove samples not divisible by n */
    uint32_t chunk_size = samples - (samples % hd.n);
    /* Extra data for last DPU */
    uint32_t extra_data = number_of_input_samples - (chunk_size * NR_DPUS);
    uint32_t extra_data_divisible = extra_data / hd.n;

    for (int i = 0; i < NR_DPUS; i++) {
        buffer_channel_lengths[i] = chunk_size;
    }

    int i = 0;
    while (extra_data_divisible != 0) {
        buffer_channel_lengths[i] += hd.n;
        extra_data_divisible--;
        i++;
        if (i == NR_DPUS) {
            i = 0;
        }
    }
    buffer_channel_lengths[NR_DPUS - 1] += extra_data % hd.n;
}

/**
 * @brief Prepare the DPU context and upload the program to the DPU.
 *
 * @param[in] data_set Quantized data buffer
 * @return             Non-zero on failure.
 */
static int prepare_dpu(int32_t * data_set, int32_t *results, void *runtime) {
    int ret = 0;

    struct timeval start, end;
    struct dpu_set_t dpus, dpu, dpu_rank;

    dpu_runtime *rt = runtime;

    uint32_t buff_offset = 0;

    dpu_input_data inputs[NR_DPUS];
    in_buffer read_bufs[NR_DPUS];

    uint32_t dpu_id = 0;
    uint32_t dpu_id_rank = 0;

    uint32_t buffer_channel_lengths[NR_DPUS];
    calculate_buffer_lengths(buffer_channel_lengths);

    // Allocate DPUs
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpus));
    DPU_ASSERT(dpu_load(dpus, DPU_PROGRAM, NULL));

    gettimeofday(&start, NULL);

    // Copy in:

    DPU_RANK_FOREACH(dpus, dpu_rank) {
        dpu_id = dpu_id_rank;
        DPU_FOREACH(dpu_rank, dpu) {
            ret = setup_dpu_data(&inputs[dpu_id], buffer_channel_lengths[dpu_id], &read_bufs[dpu_id], data_set, &buff_offset, dpu_id);
            if (ret != 0) {
                return ret;
            }

            dbg_printf("%u: buffer_channel_length = %u\n", dpu_id, inputs[dpu_id].buffer_channel_length);

            DPU_ASSERT(dpu_prepare_xfer(dpu, &hd));

            dpu_id++;
        }
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "hd", 0, sizeof(hd), DPU_XFER_DEFAULT));

        dpu_id = dpu_id_rank;

        DPU_FOREACH(dpu_rank, dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &inputs[dpu_id++]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "dpu_data", 0, sizeof(inputs[dpu_id]), DPU_XFER_DEFAULT));

        dpu_id = dpu_id_rank;

        // Slightly faster than individual copy in without prepare
        size_t largest = 0;
        DPU_FOREACH(dpu_rank, dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, read_bufs[dpu_id].buffer));
            if (read_bufs[dpu_id].buffer_size > largest) {
                largest = read_bufs[dpu_id].buffer_size;
            }
            dpu_id++;
        }
        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_TO_DPU, "read_buf", 0, largest, DPU_XFER_DEFAULT));

        dpu_id_rank = dpu_id;
    }

    gettimeofday(&end, NULL);

    rt->execution_time_copy_in = time_difference(&start, &end);

    gettimeofday(&start, NULL);
    ret = dpu_launch(dpus, DPU_SYNCHRONOUS);
    gettimeofday(&end, NULL);

    rt->execution_time_launch = time_difference(&start, &end);

    gettimeofday(&start, NULL);
    uint32_t output_buffer[NR_DPUS][MAX_INPUT];
    dpu_id_rank = dpu_id = 0;

    // Copy out:
    DPU_RANK_FOREACH(dpus, dpu_rank) {
        dpu_id = dpu_id_rank;

        uint32_t largest_size_xfer = 0;
        DPU_FOREACH(dpu_rank, dpu) {
            printf("------DPU %d Logs------\n", dpu_id);
            DPU_ASSERT(dpu_log_read(dpu, stdout));

            uint32_t size_xfer = inputs[dpu_id].output_buffer_length * sizeof(int32_t);
            if (size_xfer > largest_size_xfer) {
                largest_size_xfer = size_xfer;
            }
            DPU_ASSERT(dpu_prepare_xfer(dpu, output_buffer[dpu_id]));

            dpu_id++;
        }

        DPU_ASSERT(dpu_push_xfer(dpu_rank, DPU_XFER_FROM_DPU, "read_buf", 0, largest_size_xfer, DPU_XFER_DEFAULT));

        dpu_id_rank = dpu_id;
    }

    gettimeofday(&end, NULL);

    rt->execution_time_copy_out = time_difference(&start, &end);

    uint32_t result_num = 0;
    for (dpu_id = 0; dpu_id < NR_DPUS; dpu_id++) {
        for (uint32_t j = 0; j < inputs[dpu_id].output_buffer_length; j++) {
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
static int host_hdc(int32_t * data_set, int32_t *results, void *runtime) {

    (void) runtime;

    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[hd.bit_dim + 1];
    uint32_t q_N[hd.bit_dim + 1];
    int32_t quantized_buffer[hd.channels];

    int result_num = 0;

    memset(q, 0, (hd.bit_dim + 1) * sizeof(uint32_t));
    memset(q_N, 0, (hd.bit_dim + 1) * sizeof(uint32_t));
    memset(quantized_buffer, 0, hd.channels * sizeof(uint32_t));

    for(int ix = 0; ix < number_of_input_samples; ix += hd.n) {

        for(int z = 0; z < hd.n; z++) {

            for(int j = 0; j < hd.channels; j++) {
                // NOTE: Buffer overflow in original code?
                if (ix + z < number_of_input_samples) {
                    int ind = A2D1D(number_of_input_samples, j, ix + z);
                    quantized_buffer[j] = data_set[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the n-gram.
            // N.B. if n = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, hd.iM, hd.chAM, q);
            } else {
                compute_N_gram(quantized_buffer, hd.iM, hd.chAM, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for(int i = 1; i < hd.bit_dim; i++){
                    old_overflow = overflow;
                    overflow = q[i] & mask;
                    q[i] = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ q[i];
                }

                old_overflow = overflow;
                overflow = (q[hd.bit_dim] >> 16) & mask;
                q[hd.bit_dim] = (q[hd.bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[hd.bit_dim] = q_N[hd.bit_dim] ^ q[hd.bit_dim];

                q[0] = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ q[0];
            }
        }
        // classifies the new N-gram through the Associative Memory matrix.
        results[result_num++] = associative_memory_32bit(q, hd.aM_32);
    }

    return 0;
}

typedef struct hdc_data {
    int32_t *data_set;
    int32_t *results;
    uint32_t result_len;
    double execution_time;
} hdc_data;

typedef int (*hdc)(int32_t * data_set, int32_t *results, void *runtime);

static double run_hdc(hdc fn, hdc_data *data, void *runtime) {
    struct timeval start;
    struct timeval end;

    int ret = 0;

    uint8_t extra_result = (number_of_input_samples % hd.n) != 0;
    data->result_len = (number_of_input_samples / hd.n) + extra_result;
    uint32_t result_size = data->result_len * sizeof(int32_t);

    if ((data->results = malloc(result_size)) == NULL) {
        nomem();
    }

    gettimeofday(&start, NULL);
    ret = fn(data->data_set, data->results, runtime);
    gettimeofday(&end, NULL);

    data->execution_time = time_difference(&start, &end);

    return ret;
}

static int compare_results(hdc_data *dpu_data, hdc_data *host_data) {
    int ret = 0;

    printf("--- Compare --\n");
    printf("(%u) results\n", host_data->result_len);
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

    uint32_t buffer_size = (sizeof(int32_t) * number_of_input_samples * hd.channels);
    int32_t *data_set = malloc(buffer_size);
    if (data_set == NULL) {
        nomem();
    }

    quantize_set(test_set, data_set);

    hdc_data dpu_results = { .data_set = data_set };
    hdc_data host_results = { .data_set = data_set };

    dpu_runtime runtime;

    if (use_dpu || test_results) {
        dpu_ret = run_hdc(prepare_dpu, &dpu_results, &runtime);
    }

    if (!use_dpu || test_results) {
        host_ret = run_hdc(host_hdc, &host_results, NULL);
    }

    if (use_dpu || test_results) {
        printf("--- DPU --\n");
        if (show_results) {
            print_results(&dpu_results);
        }
        printf("DPU took %fs\n", dpu_results.execution_time);
        printf("DPU copy_in took %fs\n", runtime.execution_time_copy_in);
        printf("DPU launch took %fs\n", runtime.execution_time_launch);
        printf("DPU copy_out took %fs\n", runtime.execution_time_copy_out);
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

    return (ret + dpu_ret + host_ret);
}
