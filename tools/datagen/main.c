#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <errno.h>

#include "data/data.h"

#define CONSTANTS 7

// Data order:

// Versioned binary format (first 4 bytes)
#define VERSION 0

// Next (CONSTANTS-1 * sizeof(int)) bytes
#define DIMENSION_INDEX 1
#define CHANNELS_INDEX 2
#define BIT_DIM_INDEX 3
#define NUMBER_OF_INPUT_SAMPLES_INDEX 4
#define N_INDEX 5
#define IM_LENGTH_INDEX 6

// Followed by
// double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
// uint32_t chAM[CHANNELS][BIT_DIM + 1];
// uint32_t iM[IM_LENGTH][BIT_DIM + 1];
// uint32_t aM_32[N][BIT_DIM + 1];

/**
 * @brief Test constant is expected value
 *
 * @param[in] expected Value to check matches @p actual
 * @param[in] actual   Value to check matches @p expected
 * @param[in] id       Identifier for stderr output
 * @return             Non-zero on failure
 */
static int
check_constant(int expected, int actual, char const * id) {
    int ret = actual != expected;
    if (ret) {
        fprintf(stderr, "ERROR: %s=%d != %d\n", id, expected, actual);
    }
    return ret;
}

/**
 * @brief Validate correctness of array
 *
 * @param[in] d0        Length of dimension 0 of @p expected
 * @param[in] d1        Length of dimension 1 of @p expected
 * @param[in] expected  Expected values to compare @p data to
 * @param[in] data      Actual values to compare @p expected to
 * @return              Number of wrong entries
 */
static int
check_array_uint32(int d0, int d1, uint32_t expected[d0][d1], uint32_t const * data) {
    int ret = 0;
    for (int i = 0; i < d0; i++) {
        for (int j = 0; j < d1; j++) {
            int ind = (i * d1) + j;
            uint32_t entry = data[ind];
            if (entry != expected[i][j]) {
                fprintf(stderr, "ERROR: expected[%d][%d]=%u != data[%d]=%u\n", i, j, expected[i][j], ind, entry);
                ret++;
            }
        }
    }

    return ret;
}

/**
 * @brief Exit if NOMEM
 */
static void
nomem() {
    fprintf(stderr, "ERROR: No memory\n");
    exit(ENOMEM);
}

/**
 * @brief Validate data in @p output file
 *
 * @param[in] output File name to check against data
 * @return Non-zero on failure
 */
static int
validate_data(char const * output) {
    int ret = 0;
    errno = 0;
    FILE *file = fopen(output, "rb");
    if (file == NULL) {
        return errno != 0 ? errno : -1;
    }

    int32_t constants[CONSTANTS];
    size_t sz = sizeof(constants);
    if (fread(constants, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = sizeof(TEST_SET);
    double *test_set = malloc(sz);
    if (test_set == NULL) {
        nomem();
    }
    if (fread(test_set, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = sizeof(chAM);
    uint32_t *test_chAM = malloc(sz);
    if (test_chAM == NULL) {
        nomem();
    }
    if (fread(test_chAM, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = sizeof(iM);
    uint32_t *test_iM = malloc(sz);
    if (test_iM == NULL) {
        nomem();
    }

    if (fread(test_iM, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = sizeof(aM_32);
    uint32_t *test_aM_32 = malloc(sz);
    if (test_aM_32 == NULL) {
        nomem();
    }

    if (fread(test_aM_32, 1, sz, file) != sz) {
        return ferror(file);
    }

    ret += check_constant(DIMENSION, constants[DIMENSION_INDEX], "DIMENSION");
    ret += check_constant(CHANNELS, constants[CHANNELS_INDEX], "CHANNELS");
    ret += check_constant(BIT_DIM, constants[BIT_DIM_INDEX], "BIT_DIM");
    ret += check_constant(NUMBER_OF_INPUT_SAMPLES, constants[NUMBER_OF_INPUT_SAMPLES_INDEX], "NUMBER_OF_INPUT_SAMPLES");
    ret += check_constant(N, constants[N_INDEX], "N");
    ret += check_constant(IM_LENGTH, constants[IM_LENGTH_INDEX], "IM_LENGTH");

    // Check TEST set
    for (int i = 0; i < CHANNELS; i++) {
        for (int j = 0; j < NUMBER_OF_INPUT_SAMPLES; j++) {
            int ind = (i * constants[NUMBER_OF_INPUT_SAMPLES_INDEX] + j);
            double entry = test_set[ind];
            if (entry != TEST_SET[i][j]) {
                fprintf(stderr, "ERROR: TEST_SET[%d][%d]=%f != test_set[%d]=%f\n", i, j, TEST_SET[i][j], ind, entry);
                ret++;
            }
        }
    }

    ret += check_array_uint32(CHANNELS, BIT_DIM+1, chAM, test_chAM);
    ret += check_array_uint32(IM_LENGTH, BIT_DIM+1, iM, test_iM);
    ret += check_array_uint32(CHANNELS, BIT_DIM+1, chAM, test_chAM);
    ret += check_array_uint32(N, BIT_DIM+1, aM_32, test_aM_32);

    fclose(file);
    return ret;
}

/**
 * @brief Generate a binary data file
 * @param[in] output Filename to create & fill with data
 * @return Non-zero on failure
 */
static int
generate_data_file(char const * output) {
    int ret = 0;
    errno = 0;
    FILE *file = fopen(output, "wb");
    if (file == NULL) {
        return errno != 0 ? errno : -1;
    }

    int32_t constants[CONSTANTS] = {VERSION, DIMENSION, CHANNELS, BIT_DIM, NUMBER_OF_INPUT_SAMPLES, N, IM_LENGTH};
    fwrite(constants, sizeof(constants), 1, file);
    if ((ret = ferror(file)) != 0) {
        fprintf(stderr, "Failed to write constants to %s", output);
        goto err;
    }

    fwrite(TEST_SET, 1, sizeof(TEST_SET), file);
    if ((ret = ferror(file)) != 0) {
        fprintf(stderr, "Failed to write TEST_SET to %s", output);
        goto err;
    }

    fwrite(chAM, 1, sizeof(chAM), file);
    if ((ret = ferror(file)) != 0) {
        fprintf(stderr, "Failed to write chAM to %s", output);
        goto err;
    }

    fwrite(iM, 1, sizeof(iM), file);
    if ((ret = ferror(file)) != 0) {
        fprintf(stderr, "Failed to write iM to %s", output);
        goto err;
    }

    fwrite(aM_32, 1, sizeof(aM_32), file);
    if ((ret = ferror(file)) != 0) {
        fprintf(stderr, "Failed to write aM_32 to %s", output);
        goto err;
    }

err:
    fclose(file);
    return ret;
}

/**
 * @brief Output usage information
 * @param stream
 * @param executable
 */
static void
usage(FILE *stream, char const *executable) {
    fprintf(stream, "usage: %s [ -o <output> ] [ -t ]\n", executable);
    fprintf(stream, "\to: redirect output to file\n");
    fprintf(stream, "\tt: validate data after creating\n");
}

int
main(int argc, char *argv[]) {
    char const options[] = "hto:";
    char *output = NULL;
    bool test_data = false;

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'o':
                output = optarg;
                break;

            case 't':
                test_data = true;
                break;

            case 'h':
                usage(stdout, argv[0]);
                return EXIT_SUCCESS;

            default:
                usage(stderr, argv[0]);
                return EXIT_FAILURE;
        }
    }

    if ((argc - optind) != 0) {
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    int ret = generate_data_file(output);
    if (ret != 0) {
        return ret;
    }

    if (test_data && ((ret = validate_data(output)) != 0)) {
        fprintf(stderr, "Data validation failed\n");
    }

    return ret;
}
