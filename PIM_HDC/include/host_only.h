#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"

#define ALIGN(_p, _width) (((unsigned int)_p + (_width-1)) & (0-_width))

int round_to_int(float num);
void quantize_set(float input_set[CHANNELS][NUMBER_OF_INPUT_SAMPLES], int32_t * buffer);

#endif
