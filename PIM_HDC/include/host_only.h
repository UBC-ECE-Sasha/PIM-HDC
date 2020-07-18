#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"

#include <uchar.h>

// Number of samples in each channel's dataset
extern int32_t number_of_input_samples;

extern dpu_hdc_vars hd;

int
read_data(char const *input_file, double **test_set);
int
round_to_int(double num);
void
quantize_set(double const *input_set, int32_t *buffer);
void
nomem();
#endif
