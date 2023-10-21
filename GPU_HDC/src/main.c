#include "aux_functions.h"
#include "common.h"
#include "host.h"
#include "host_only.h"
#include "init.h"

#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

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
                compute_N_gram(quantized_buffer, q);
            } else {
                compute_N_gram(quantized_buffer, q_N);

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
        results[result_num++] = associative_memory_32bit(q, hd.aM_32);
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
    bool show_results = false;
    bool test_results = false;
    bool runtime_only = false;
    int ret = 0;
    int host_ret = 0;
    char const options[] = "sthri:";
    char *input = NULL;

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'i':
                input = optarg;
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

    hdc_data dpu_results = {.data_set = data_set, .results = NULL};
    hdc_data host_results = {.data_set = data_set, .results = NULL};

    if (test_results) {
        host_ret = run_hdc(host_hdc, &host_results, NULL);
        if (host_ret != 0) {
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

    // if (test_results) {
    //     ret = compare_results(&dpu_results, &host_results, runtime_only);
    // }

err:
    free(data_set);
    free(test_set);
    free(host_results.results);
    free(dpu_results.results);

    return (ret + host_ret);
}
