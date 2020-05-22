#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"

int round_to_int(float num);
void quantize_set(float input_set[CHANNELS][NUMBER_OF_INPUT_SAMPLES], int buffer[CHANNELS][NUMBER_OF_INPUT_SAMPLES]);

#endif
