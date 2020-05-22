#include "associative_memory.h"
#include "aux_functions.h"

/**
 * @brief Tests the accuracy based on input testing queries.
 *
 * @param[in] q_32  Query hypervector
 * @param[in] aM_32 Trained associative memory
 * @return          Classification result
 */
int associative_memory_32bit(uint32_t q_32[BIT_DIM + 1], uint32_t aM_32[][BIT_DIM + 1]) {
    int sims[CLASSES] = {0};
    int class;

    // Computes Hamming Distances
    hamming_dist(q_32, aM_32, sims);

    // Classification with Hamming Metric
    class = max_dist_hamm(sims);

    return class;
}
