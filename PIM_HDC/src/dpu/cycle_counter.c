#include "cycle_counter.h"

/**
 * @brief Begin counting cycles for a section
 *
 * @param[in,out] total_counter   Total current cycles.
 *                                if `total_counter != NULL`, it will be used to
 *                                keep track of the total cycles.
 *
 * @pre                           Function is not nested
 * @post                          perfcounter is reset
 */
void cycles_count_start(perfcounter_t *total_counter) {
    if (total_counter != NULL) {
        *total_counter += perfcounter_get();
    }
    (void) perfcounter_config(COUNTER_CONFIG, true);
}

/**
 * @brief Stop counting cycles for a section
 *
 * @param[in,out] total_counter   Total current cycles.
 *                                if `total_counter != NULL`, it will be used to
 *                                keep track of the total cycles.
 * @param section_counter         Total sections cycles.
 *                                if `section_counter != NULL`, it will be used to
 *                                keep track of the total cycles for a section.
 * @return                        Count from section
 *
 * @pre                           Function is not nested
 * @post                          perfcounter is reset
 */
perfcounter_t cycles_count_finish(perfcounter_t *total_counter, perfcounter_t *section_counter) {
    perfcounter_t cnt = perfcounter_get();
    if (total_counter != NULL) {
        *total_counter += cnt;
    }
    if (section_counter != NULL) {
        *section_counter += cnt;
    }
    (void) perfcounter_config(COUNTER_CONFIG, true);
    return cnt;
}
