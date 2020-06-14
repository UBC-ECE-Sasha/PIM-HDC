#include "cycle_counter.h"

/**
 * @brief Begin counting cycles for a section
 *
 * @param[out] counter   Counter used for section
 */
perfcounter_t cycles_count_start(perfcounter_t *counter) {
    return (*counter = perfcounter_config(COUNTER_CONFIG, false));
}

/**
 * @brief Stop counting cycles for a section
 *
 * @param[in] total_counter           Counter used for cycles_count_start
 * @param[out] section_counter        Total section's cycles.
 */
perfcounter_t cycles_count_finish(perfcounter_t counter, perfcounter_t *section_counter) {
    perfcounter_t diff = perfcounter_get() - counter;
    *section_counter += diff;
    return diff;
}
