#include "aux_functions.h"
#include "common.h"
#include "host.h"
#include "host_only.h"
#include "init.h"

#include <string.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct in_buffer
 *
 * @brief   Input buffer for a DPU
 */
typedef struct in_buffer {
    int32_t buffer[HDC_MAX_INPUT];
    size_t buffer_size;
} in_buffer;

/**
 * @struct hdc_data
 * @brief HDC data for HDC task
 */
typedef struct hdc_data {
    int32_t *data_set;     /**< Input HDC dataset */
    int32_t *results;      /**< Output from run */
    uint32_t result_len;   /**< Length of the results */
    double execution_time; /**< Total execution time of run */
} hdc_data;

/**
 * @brief Function for @p run_hdc to run HDC task
 */
typedef int (*hdc)(int32_t *data_set, int32_t *results, void *runtime);

/**
 * @brief Calculate individual buffer lengths for each DPU or tasklet with as even distribution as
 * possible
 * @param[in]  length                  Lengths of @p buffer_channel_lengths
 * @param[in]  samples                 Samples to distribute
 * @param[out] buffer_channel_lengths  Lengths for each channel
 */
static void
calculate_buffer_lengths(uint32_t length, uint32_t buffer_channel_lengths[length],
                         uint32_t input_samples) {
    /* Section of buffer for one channel, without samples not divisible by n */
    uint32_t samples = input_samples / length;
    /* Remove samples not divisible by n */
    uint32_t chunk_size = samples - (samples % hd.n);
    /* Extra data for last DPU */
    uint32_t extra_data = input_samples - (chunk_size * length);
    uint32_t extra_data_divisible = extra_data / hd.n;

    for (uint32_t i = 0; i < length; i++) {
        buffer_channel_lengths[i] = chunk_size;
    }

    uint32_t i = 0;
    while (extra_data_divisible != 0) {
        buffer_channel_lengths[i] += hd.n;
        extra_data_divisible--;
        i++;
        if (i == length) {
            i = 0;
        }
    }
    buffer_channel_lengths[length - 1] += extra_data % hd.n;
}

/**
 * @brief Run a HDC workload and time the execution
 *
 * @param[in] fn        Function to run HDC algorithm
 * @param[out] data     Results from HDC run
 * @param[out] runtime  Run times from sections of @p fn
 *
 * @return Non-zero On failure
 */
static double
run_hdc(hdc fn, hdc_data *data, void *runtime) {
    struct timespec start, end;

    int ret = 0;

    uint8_t extra_result = (number_of_input_samples % hd.n) != 0;
    data->result_len = (number_of_input_samples / hd.n) + extra_result;
    uint32_t result_size = data->result_len * sizeof(int32_t);

    if ((data->results = malloc(result_size)) == NULL) {
        nomem();
    }

    TIME_NOW(&start);
    ret = fn(data->data_set, data->results, runtime);
    TIME_NOW(&end);

    data->execution_time = TIME_DIFFERENCE(start, end);

    return ret;
}

/**
 * @brief Compare the results from the host and DPU confirming they are the same
 *        or printing differences
 *
 * @param[in] dpu_data   Results to be tested from DPU
 * @param[in] host_data  Results to be tested from host
 * @param[in] check_only Only check results are equal, dont print differences
 *
 * @return               Non-zero if results are not the same
 */
static int
compare_results(hdc_data *dpu_data, hdc_data *host_data, bool check_only) {
    int ret = 0;

    if (!check_only) {
        printf("--- Compare --\n");
        printf("(%u) results\n", host_data->result_len);
    }

    for (uint32_t i = 0; i < host_data->result_len; i++) {
        if (host_data->results[i] != dpu_data->results[i]) {
            if (check_only) {
                return -1;
            }
            fprintf(stderr, "(host_results[%u] = %d) != (dpu_results[%u] = %d)\n", i,
                    host_data->results[i], i, dpu_data->results[i]);
            ret = -1;
        }
    }

    if (check_only) {
        return 0;
    }

    // char *faster;
    // double time_diff, percent_diff;
    // if (dpu_data->execution_time > host_data->execution_time) {
    //     faster = "Host";
    //     time_diff = dpu_data->execution_time - host_data->execution_time;
    //     percent_diff = dpu_data->execution_time / host_data->execution_time;
    // } else {
    //     faster = "DPU";
    //     time_diff = host_data->execution_time - dpu_data->execution_time;
    //     percent_diff = host_data->execution_time / dpu_data->execution_time;
    // }

    // printf("%s was %fs (%f x) faster\n", faster, time_diff, percent_diff);

    return ret;
}

/**
 * @brief Print results from HDC run
 * @param[in] data  Results to print
 */
static void
print_results(hdc_data *data) {
    for (uint32_t i = 0; i < data->result_len; i++) {
        printf("%d\n", data->results[i]);
    }
}

/**
 * @brief Set up the data for each GPU block
 * @param[out]    input                  Datastructure to be populated for GPU
 * @param[in]     buffer_channel_length  Length of an individual channel
 * @param[in]     data_in                Data buffer for DPU
 * @param[in]     data_set               Input data for populating @p data_in
 * @param[in,out] buff_offset            Current offset in @p dataset
 * @param[in]     gpu_id                 ID of DPU
 *
 * @return                               Non-zero on failure
 */
static int
setup_gpu_data(gpu_input_data *input, uint32_t buffer_channel_length, in_buffer *data_in,
               int32_t *data_set, uint32_t *buff_offset, uint32_t gpu_id) {

    uint32_t buffer_channel_lengths[NR_THREADS];
    calculate_buffer_lengths(NR_THREADS, buffer_channel_lengths, buffer_channel_length);

    uint32_t loc = 0;
    uint32_t idx_offset = 0;
    for (uint8_t idx = 0; idx < NR_THREADS; idx++) {
        input->task_begin[idx] = loc;
        loc += buffer_channel_lengths[idx];
        input->task_end[idx] = loc;

        uint32_t task_samples = input->task_end[idx] - input->task_begin[idx];

        input->idx_offset[idx] = idx_offset;
        idx_offset += task_samples / hd.n;

        dbg_printf("%u: samples = %u\n", idx, task_samples);
        dbg_printf("%u: idx_offset = %u\n", idx, input->idx_offset[idx]);
        dbg_printf("%u: task_end = %u, task_begin = %u\n", idx, input->task_end[idx],
                   input->task_begin[idx]);
    }

    /* Input */
    if (gpu_id == NR_BLOCKS - 1) {
        /* No n on last in algorithm */
        input->buffer_channel_usable_length = buffer_channel_length;
    } else {
        input->buffer_channel_usable_length = buffer_channel_length + hd.n;
    }
    input->buffer_channel_aligned_size = ALIGN(buffer_channel_length * sizeof(int32_t), 8);

    if ((input->buffer_channel_usable_length * hd.channels) > HDC_MAX_INPUT) {
        fprintf(stderr,
                "buffer_channel_usable_length * hd.channels (%u) cannot be greater than HDC_MAX_INPUT "
                "= (%d)\n",
                input->buffer_channel_usable_length * hd.channels, HDC_MAX_INPUT);
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
            int32_t *ta = &data_set[(i * number_of_input_samples) + *buff_offset];
            dbg_printf("INPUT data_set[%d] (%u bytes) (%u chunk_size, %u usable):\n", i,
                       input->buffer_channel_aligned_size, input->buffer_channel_length,
                       input->buffer_channel_usable_length);
            (void) memcpy(&data_in->buffer[i * input->buffer_channel_usable_length], ta, sz_xfer);
        }
    }

    *buff_offset += input->buffer_channel_length;

    size_t total_xfer = 0;
    if (total_xfer > (HDC_MAX_INPUT * sizeof(int32_t))) {
        fprintf(stderr, "Error %lu is too large for read_buf[%d]\n", total_xfer / sizeof(int32_t),
                HDC_MAX_INPUT);
        return -1;
    }

    return 0;
}

/**
 * @brief Run the HDC algorithm for the GPU
 *
 * @param[in]  data_set  Input dataset
 * @param[out] results   Results from run
 * @param[out] runtime   Runtimes of individual sections (unused)
 *
 * @return               Non-zero on failure.
 */
static int
gpu_setup_hdc(int32_t *data_set, int32_t *results, void *runtime) {

    int ret = 0;

    uint32_t buff_offset = 0;

    gpu_input_data inputs[NR_BLOCKS];
    in_buffer read_bufs[NR_BLOCKS];

    uint32_t gpu_id = 0;
    uint32_t gpu_id_rank = 0;

    uint32_t buffer_channel_lengths[NR_BLOCKS];

    uint32_t result[NR_BLOCKS][HDC_MAX_INPUT];

    calculate_buffer_lengths(NR_BLOCKS, buffer_channel_lengths, number_of_input_samples);

    // Copy in:
    uint32_t result_num = 0;
    for (int i = 0; i < NR_BLOCKS; i++) {
        setup_gpu_data(&inputs[i], buffer_channel_lengths[i],
                       &read_bufs[i], data_set, &buff_offset, i);

        for (int t = 0; t < NR_THREADS; t++) {
            if ((inputs[i].task_end[t] - inputs[i].task_begin[t]) > 0) {
                ret = gpu_hdc(inputs[i], read_bufs[i].buffer, result[i], inputs[i].idx_offset[t],
                              inputs[i].task_begin[t], inputs[i].task_end[t]);


            } else {
                printf("%u:%u: No work to do\n", i, t);
            }
        }

        for (uint32_t j = 0; j < inputs[i].output_buffer_length; j++) {
            results[result_num++] = result[i][j];
        }

    }



    return 0;
}

/**
 * @brief Run the HDC algorithm for the host
 *
 * @param[in]  data_set  Input dataset
 * @param[out] results   Results from run
 * @param[out] runtime   Runtimes of individual sections (unused)
 *
 * @return               Non-zero on failure.
 */
static int
host_hdc(int32_t *data_set, int32_t *results, void *runtime) {

    (void) runtime;

    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[hd.bit_dim + 1];
    uint32_t q_N[hd.bit_dim + 1];
    int32_t quantized_buffer[hd.channels];

    int result_num = 0;

    for (int ix = 0; ix < number_of_input_samples; ix += hd.n) {

        for (int z = 0; z < hd.n; z++) {

            for (int j = 0; j < hd.channels; j++) {
                if (ix + z < number_of_input_samples) {
                    int ind = A2D1D(number_of_input_samples, j, ix + z);
                    quantized_buffer[j] = data_set[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the n-gram.
            // N.B. if n = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                host_compute_N_gram(quantized_buffer, q);
            } else {
                host_compute_N_gram(quantized_buffer, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for (int i = 1; i < hd.bit_dim; i++) {
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
        results[result_num++] = host_associative_memory_32bit(q, hd.aM_32);
    }

    return 0;
}

/**
 * @brief Display usage information to @p stream
 * @param[in] stream    File pointer to write usage to
 * @param[in] exe_name  Name of executable
 */
static void
usage(FILE *stream, char const *exe_name) {
#ifdef DEBUG
    fprintf(stream, "**DEBUG BUILD**\n");
#endif

    fprintf(stream, "usage: %s [ -d ] -i <INPUT_FILE>\n", exe_name);
    fprintf(stream, "\ti: input file\n");
    fprintf(stream, "\tr: show runtime only\n");
    fprintf(stream, "\ts: show results\n");
    fprintf(stream, "\tt: test results\n");
    fprintf(stream, "\th: help message\n");
}

int
main(int argc, char **argv) {
    bool use_gpu = false;
    bool show_results = false;
    bool test_results = false;
    bool runtime_only = false;
    int ret = 0;
    int host_ret = 0;
    int gpu_ret = 0;
    char const options[] = "sgthri:";
    char *input = NULL;

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'i':
                input = optarg;
                break;

            case 'g':
                use_gpu = true;
                break;

            case 's':
                show_results = true;
                break;

            case 't':
                test_results = true;
                break;

            case 'r':
                runtime_only = true;
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

    hdc_data gpu_results = {.data_set = data_set, .results = NULL};
    hdc_data host_results = {.data_set = data_set, .results = NULL};

    if (test_results) {
        host_ret = run_hdc(host_hdc, &host_results, NULL);
        if (host_ret != 0) {
            goto err;
        }
    }

    if (use_gpu) {
        gpu_ret = run_hdc(gpu_setup_hdc, &gpu_results, NULL);
        if (gpu_ret != 0) {
            goto err;
        }
    }

    if (test_results) {
        if (!runtime_only) {
            printf("--- Host --\n");
            if (show_results) {
                print_results(&host_results);
            }
            printf("Host took %fs\n", host_results.execution_time);
        } else {
            printf("%f\n", host_results.execution_time);
        }
    }

    if (test_results) {
        ret = compare_results(&gpu_results, &host_results, runtime_only);
    }

err:
    free(data_set);
    free(test_set);
    free(host_results.results);
    free(gpu_results.results);

    return (ret + gpu_ret + host_ret);
}
