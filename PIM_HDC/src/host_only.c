#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include "host_only.h"

int32_t dimension;
int32_t channels;
int32_t bit_dim;
int32_t number_of_input_samples;
int32_t n;
int32_t im_length;

uint32_t *chAM;
uint32_t *iM;
uint32_t *aM_32;

/**
 * @brief Exit if NOMEM
 */
void nomem() {
    fprintf(stderr, "ERROR: No memory\n");
    exit(ENOMEM);
}

/**
 * @brief Read data from @p input_file into globals and test_set
 *
 * @param[in] input_file    File to read from
 * @param[in,out] test_set  Test data to allocate and fill
 */
int read_data(char const * input_file, double **test_set) {
    int ret = 0;
    errno = 0;
    FILE *file = fopen(input_file, "rb");
    if (file == NULL) {
        return errno != 0 ? errno : -1;
    }

    int32_t version;
    size_t sz = sizeof(version);
    if (fread(&version, 1, sz, file) != sz) {
        return ferror(file);
    }
    if (version != VERSION) {
        fprintf(stderr, "Binary file version (%d) does not match expected (%d)\n", version, VERSION);
        return -1;
    }
    if ((fread(&dimension, 1, sz, file) != sz) ||
        (fread(&channels, 1, sz, file) != sz) ||
        (fread(&bit_dim, 1, sz, file) != sz) ||
        (fread(&number_of_input_samples, 1, sz, file) != sz) ||
        (fread(&n, 1, sz, file) != sz) ||
        (fread(&im_length, 1, sz, file) != sz)) {
        return ferror(file);
    }

    sz = channels * number_of_input_samples * sizeof(double);
    *test_set = malloc(sz);
    if (*test_set == NULL) {
        nomem();
    }
    if (fread(*test_set, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = channels * (bit_dim + 1) * sizeof(uint32_t);
    chAM = malloc(sz);
    if (chAM == NULL) {
        nomem();
    }
    if (fread(chAM, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = im_length * (bit_dim + 1) * sizeof(uint32_t);
    iM = malloc(sz);
    if (iM == NULL) {
        nomem();
    }

    if (fread(iM, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = n * (bit_dim + 1) * sizeof(uint32_t);
    aM_32 = malloc(sz);
    if (aM_32 == NULL) {
        nomem();
    }

    if (fread(aM_32, 1, sz, file) != sz) {
        return ferror(file);
    }

    fclose(file);
    return ret;
}

/**
 * @brief Round a double to an integer.
 *
 * @param[in] num Double to round
 * @return        Rounded integer value
 */
int round_to_int(double num) {
    return (num - floor(num) > 0.5) ? ceil(num) : floor(num);
}

/**
 * @brief Quantization: each sample is rounded to the nearest integer.
 *
 * @param[out] buffer Rounded integers
 */
void quantize_set(double const * input_set, int32_t * buffer) {
    for(int i = 0; i < channels; i++) {
        for(int j = 0; j < number_of_input_samples; j++) {
            buffer[(i * number_of_input_samples) + j] = round_to_int(
                input_set[(i * number_of_input_samples) + j]);
        }
    }
}
