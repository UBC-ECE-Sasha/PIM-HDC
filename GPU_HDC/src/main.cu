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
#include <driver_types.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct gpu_runtime
 * @brief GPU execution times
 */
typedef struct gpu_runtime {
    double execution_time_alloc;
    double execution_time_copy_in;
    double execution_time_launch;
    double execution_time_copy_out;
} gpu_runtime;

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
typedef int (*hdc)(hdc_data *data, void *runtime);

#define gpuErrchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
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

    if ((data->results = (int32_t *)malloc(result_size)) == NULL) {
        nomem();
    }

    TIME_NOW(&start);
    ret = fn(data, runtime);
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

    char *faster;
    double time_diff, percent_diff;
    if (dpu_data->execution_time > host_data->execution_time) {
        faster = "Host";
        time_diff = dpu_data->execution_time - host_data->execution_time;
        percent_diff = dpu_data->execution_time / host_data->execution_time;
    } else {
        faster = "GPU";
        time_diff = host_data->execution_time - dpu_data->execution_time;
        percent_diff = host_data->execution_time / dpu_data->execution_time;
    }

    printf("%s was %fs (%f x) faster\n", faster, time_diff, percent_diff);

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
 * @brief Calculate individual buffer lengths for each DPU or tasklet with as even distribution as
 * possible
 * @param[in]  length                  Lengths of @p buffer_channel_lengths
 * @param[in]  samples                 Samples to distribute
 * @param[out] buffer_channel_lengths  Lengths for each channel
 */
static void
calculate_buffer_lengths(uint32_t length, uint32_t *buffer_channel_lengths,
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
setup_gpu_data(gpu_input_data *input, uint32_t buffer_channel_length,
               int32_t *data_set, uint32_t *buff_offset, uint32_t gpu_id) {

    uint32_t num_splits = NR_BLOCKS*NR_THREADS;

    uint32_t buffer_channel_lengths[NR_BLOCKS*NR_THREADS];
    calculate_buffer_lengths(NR_BLOCKS*NR_THREADS, buffer_channel_lengths, buffer_channel_length);

    uint32_t loc = 0;
    uint32_t idx_offset = 0;
    for (uint32_t idx = 0; idx < num_splits; idx++) {
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

    /* Output */
    uint32_t extra_result = (buffer_channel_length % hd.n) != 0;
    input->output_buffer_length = (buffer_channel_length / hd.n) + extra_result;

    input->buffer_channel_length = buffer_channel_length;

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
gpu_setup_hdc(hdc_data *data, void *runtime) {

    uint32_t buff_offset = 0;

    gpu_runtime *rt = (gpu_runtime *)runtime;

    struct timespec start, end;

    gpu_input_data *g_inputs;
    gpu_hdc_vars *g_hd;
    int32_t *g_results;

    TIME_NOW(&start);
    uint32_t result_size = data->result_len * sizeof(int32_t);
    gpuErrchk(cudaMalloc((void **)&g_results, result_size));
    gpuErrchk(cudaMallocManaged((void **)&g_inputs, sizeof(gpu_input_data), cudaMemAttachGlobal));
    gpuErrchk(cudaMallocManaged((void **)&g_hd, sizeof(gpu_hdc_vars), cudaMemAttachGlobal))

    cudaDeviceSynchronize();
    TIME_NOW(&end);

    rt->execution_time_alloc = TIME_DIFFERENCE(start, end);

    TIME_NOW(&start);
    memcpy(g_hd, &hd, sizeof(gpu_hdc_vars));
    memcpy(g_hd->iM, iM, MAX_IM_LENGTH * (MAX_BIT_DIM + 1) * sizeof(uint32_t));
    memcpy(g_hd->chAM, chAM, MAX_CHANNELS * (MAX_BIT_DIM + 1) * sizeof(uint32_t));

    // Copy in:
    setup_gpu_data(g_inputs, number_of_input_samples,
                   data->data_set, &buff_offset, NR_BLOCKS-1);
    TIME_NOW(&end);

    rt->execution_time_copy_in = TIME_DIFFERENCE(start, end);

    TIME_NOW(&start);
    gpu_hdc<<<NR_BLOCKS,NR_THREADS>>>(g_inputs, data->data_set, g_results, g_hd);

    cudaDeviceSynchronize();
    TIME_NOW(&end);

    rt->execution_time_launch = TIME_DIFFERENCE(start, end);

    TIME_NOW(&start);
    gpuErrchk(cudaMemcpy((void *)data->results, (void *)g_results, result_size, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    TIME_NOW(&end);


    rt->execution_time_copy_out = TIME_DIFFERENCE(start, end);

    gpuErrchk(cudaFree(g_inputs));
    gpuErrchk(cudaFree(g_hd));

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
host_hdc(hdc_data *data, void *runtime) {

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
                    quantized_buffer[j] = data->data_set[ind];
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
        data->results[result_num] = host_associative_memory_32bit(q, hd.aM_32);
        // printf("i=%i,r=%i\n", result_num, results[result_num]);
        result_num++;
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
    int32_t *data_set = (int32_t *)malloc(buffer_size);
    if (data_set == NULL) {
        nomem();
    }

    int32_t *g_data_set;
    gpuErrchk(cudaMallocManaged((void **)&g_data_set, buffer_size, cudaMemAttachGlobal));

    quantize_set(test_set, data_set);

    memcpy(g_data_set, data_set, buffer_size);

    hdc_data gpu_results = {.data_set = g_data_set, .results = NULL};
    hdc_data host_results = {.data_set = data_set, .results = NULL};

    gpu_runtime rt;

    if (test_results) {
        host_ret = run_hdc(host_hdc, &host_results, NULL);
        if (host_ret != 0) {
            goto err;
        }
    }

    if (use_gpu) {
        gpu_ret = run_hdc(gpu_setup_hdc, &gpu_results, &rt);
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

    if (use_gpu || test_results) {
        if (!runtime_only) {
            printf("--- GPU --\n");
            if (show_results) {
                print_results(&gpu_results);
            }
            printf("GPU took %fs\n", gpu_results.execution_time);
            printf("GPU alloc took %fs\n", rt.execution_time_alloc);
            printf("GPU copy_in took %fs\n", rt.execution_time_copy_in);
            printf("GPU launch took %fs\n", rt.execution_time_launch);
            printf("GPU copy_out took %fs\n", rt.execution_time_copy_out);
        } else {
            printf("%f,%f,%f,%f,%f\n", gpu_results.execution_time,
                   rt.execution_time_alloc, rt.execution_time_copy_in,
                   rt.execution_time_launch, rt.execution_time_copy_out);
        }
    }

    if (test_results) {
        ret = compare_results(&gpu_results, &host_results, runtime_only);
    }

err:
    free(data_set);
    free(test_set);
    free(host_results.results);
    cudaFree(gpu_results.results);
    cudaFree(g_data_set);

    return (ret + gpu_ret + host_ret);
}
