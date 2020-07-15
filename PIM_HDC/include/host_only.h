#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"

// Number of samples in each channel's dataset
extern int32_t number_of_input_samples;

extern uint32_t *chAM;
extern uint32_t *iM;
extern uint32_t *aM_32;

// double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
// uint32_t chAM[CHANNELS][BIT_DIM + 1];
// uint32_t iM[IM_LENGTH][BIT_DIM + 1];
// uint32_t aM_32[N][BIT_DIM + 1];

extern dpu_hdc_vars hd;

typedef struct in_buffer {
    int32_t * buffer;
    uint32_t buffer_size;
} in_buffer;

int read_data(char const * input_file, double **test_set);
int round_to_int(double num);
void quantize_set(double const * input_set, int32_t * buffer);
void nomem();
#endif
