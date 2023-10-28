#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"

#include <uchar.h>

// Number of samples in each channel's dataset
extern int32_t number_of_input_samples;

extern gpu_hdc_vars hd;

extern uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
extern uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];

int
gpu_hdc(gpu_input_data gpu_data, int32_t *read_buf, int32_t *result);
int
read_data(char const *input_file, double **test_set);
int
round_to_int(double num);
void
quantize_set(double const *input_set, int32_t *buffer);
void
nomem();
#endif
