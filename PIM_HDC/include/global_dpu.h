#ifndef GLOBAL_DPU_H_
#define GLOBAL_DPU_H_

#include "init.h"

#include <defs.h>
#include <perfcounter.h>

extern dpu_hdc_vars hd;

#ifndef IM_IN_WRAM
extern uint32_t __mram_ptr mram_iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
#endif

#ifndef CHAM_IN_WRAM
extern uint32_t __mram_ptr mram_chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
#endif

#endif
