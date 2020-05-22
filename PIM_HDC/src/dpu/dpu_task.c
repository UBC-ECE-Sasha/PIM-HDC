#include <mram.h>
#include <defs.h>
#include <perfcounter.h>
#include <stdio.h>
#include "alloc.h"

int main() {
    uint8_t idx = me();

    printf("DPU starting, tasklet %d\n", idx);

    perfcounter_config(COUNT_CYCLES, true);

    printf("Tasklet %d: completed in %ld cycles\n", idx, perfcounter_get());

    return 0;
}
