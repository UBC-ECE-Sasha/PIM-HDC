#include "host_only.h"

#include <string.h>

/**
 * @brief Computes the number of 1's
 *
 * @param i The i-th variable that composes the hypervector
 * @return  Number of 1's in i-th variable of hypervector
 */
static inline int
number_of_set_bits(uint32_t i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

/**
 * @brief Computes the maximum Hamming Distance.
 *
 * @param[in] distances Distances associated to each class
 * @return              The class related to the maximum distance
 */
static int
max_dist_hamm(int distances[CLASSES]) {
    int max = distances[0];
    int max_index = 0;

    for (int i = 1; i < CLASSES; i++) {
        if (max > distances[i]) {
            max = distances[i];
            max_index = i;
        }
    }

    return max_index;
}

/**
 * @brief Computes the Hamming Distance for each class.
 *
 * @param[in] q     Query hypervector
 * @param[in] aM    Associative Memory matrix
 * @param[out] sims Distances' vector
 */
static void
hamming_dist(uint32_t *q, uint32_t *aM, int sims[CLASSES]) {
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < hd.bit_dim + 1; j++) {
            sims[i] += number_of_set_bits(q[j] ^ aM[A2D1D(hd.bit_dim + 1, i, j)]);
        }
    }
}

/**
 * @brief Tests the accuracy based on input testing queries.
 *
 * @param[in] q_32  Query hypervector
 * @param[in] aM_32 Trained associative memory
 * @return          Classification result
 */
int
host_associative_memory_32bit(uint32_t *q_32, uint32_t *aM_32) {
    int sims[CLASSES] = {0};

    // Computes Hamming Distances
    hamming_dist(q_32, aM_32, sims);

    // Classification with Hamming Metric
    return max_dist_hamm(sims);
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[out] query      Query hypervector
 */
void
host_compute_N_gram(int32_t *input, uint32_t *query) {

    uint32_t chHV[MAX_CHANNELS + 1];

    for (int i = 0; i < hd.bit_dim + 1; i++) {
        query[i] = 0;
        for (int j = 0; j < hd.channels; j++) {
            int ix = input[j];

            uint32_t im = iM[A2D1D(hd.bit_dim + 1, ix, i)];
            uint32_t cham = chAM[A2D1D(hd.bit_dim + 1, j, i)];

            chHV[j] = im ^ cham;
        }
        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[hd.channels] = chHV[0] ^ chHV[1];

        // componentwise majority: compute the number of 1's
        for (int z = 31; z >= 0; z--) {
            uint32_t cnt = 0;
            for (int j = 0; j < hd.channels + 1; j++) {
                uint32_t a = chHV[j] >> z;
                uint32_t mask = a & 1;
                cnt += mask;
            }

            if (cnt > 2) {
                query[i] = query[i] | (1 << z);
            }
        }
    }
}

