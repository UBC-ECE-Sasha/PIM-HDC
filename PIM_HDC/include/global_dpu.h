#ifndef GLOBAL_DPU_H_
#define GLOBAL_DPU_H_

#include "init.h"
#include "read_helper.h"

#include <defs.h>
#include <perfcounter.h>

extern dpu_hdc_vars hd;
extern in_buffer_context reader;
// extern in_buffer_context readers[NR_TASKLETS];

#ifndef IM_IN_WRAM
extern uint32_t __mram_ptr mram_iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
#endif

#ifndef CHAM_IN_WRAM
extern uint32_t __mram_ptr mram_chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
#endif

#ifndef AM_IN_WRAM
extern uint32_t __mram_ptr mram_aM_32[MAX_N * (MAX_BIT_DIM + 1)];
#endif

#endif
