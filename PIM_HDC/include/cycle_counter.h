#ifndef CYCLE_COUNTER_H_
#define CYCLE_COUNTER_H_

#include <perfcounter.h>
#include <stdlib.h>

#ifdef DEBUG
#    define CYCLES_COUNT_START(c) cycles_count_start(c)
#else
#    define CYCLES_COUNT_START(c)
#endif

#ifdef DEBUG
#    define CYCLES_COUNT_FINISH(tc, sc) cycles_count_finish(tc, sc)
#else
#    define CYCLES_COUNT_FINISH(tc, sc)
#endif

perfcounter_t
cycles_count_start(perfcounter_t *total_counter);

perfcounter_t
cycles_count_finish(perfcounter_t counter, perfcounter_t *section_counter);

#endif // CYCLE_COUNTER_H_
