#include <math.h>
#include <stdio.h>

#include "host_only.h"

/**
 * @brief Round a float to an integer.
 *
 * @param[in] num Float to round
 * @return        Rounded integer value
 */
int round_to_int(float num) {
    return (num - floorf(num) > 0.5f) ? ceilf(num) : floorf(num);
}

/**
 * @brief Quantization: each sample is rounded to the nearest integer.
 *
 * @param[out] buffer Rounded integers
 */
void quantize_set(float input_set[CHANNELS][NUMBER_OF_INPUT_SAMPLES], int32_t * buffer) {
    for(int i = 0; i < CHANNELS; i++) {
        for(int j = 0; j < NUMBER_OF_INPUT_SAMPLES; j++) {
            buffer[(i * NUMBER_OF_INPUT_SAMPLES) + j] = round_to_int(input_set[i][j]);
        }
    }
}
