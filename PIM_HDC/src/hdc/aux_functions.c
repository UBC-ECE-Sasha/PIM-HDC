#include "aux_functions.h"

#include "host_only.h"

#include <string.h>

/**
 * @brief Computes the maximum Hamming Distance.
 *
 * @param[in] distances Distances associated to each class
 * @return              The class related to the maximum distance
 */
int
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
void
hamming_dist(uint32_t q[hd.bit_dim + 1], uint32_t *aM, int sims[CLASSES]) {
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < hd.bit_dim + 1; j++) {
            sims[i] += number_of_set_bits(q[j] ^ aM[A2D1D(hd.bit_dim + 1, i, j)]);
        }
    }
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[out] query      Query hypervector
 */
void
compute_N_gram(int32_t input[hd.channels], uint32_t query[hd.bit_dim + 1]) {

    uint32_t chHV[hd.channels + 1];

    for (int i = 0; i < hd.bit_dim + 1; i++) {
        query[i] = 0;
        for (int j = 0; j < hd.channels; j++) {
            int ix = input[j];
            uint32_t im;
#ifdef IM_IN_WRAM
            im = hd.iM[A2D1D(hd.bit_dim + 1, ix, i)];
#else
            im = iM[A2D1D(hd.bit_dim + 1, ix, i)];
#endif
            chHV[j] = im ^ hd.chAM[A2D1D(hd.bit_dim + 1, j, i)];
        }
        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[hd.channels] = chHV[0] ^ chHV[1];

        // componentwise majority: insert the value of the ith bit of each chHV row in the variable
        // "majority" and then compute the number of 1's with the function
        // numberOfSetBits(uint32_t).
        for (int z = 31; z >= 0; z--) {
            uint32_t majority = 0;
            for (int j = 0; j < hd.channels + 1; j++) {
                majority = majority | (((chHV[j] >> z) & 1) << j);
            }

            if (number_of_set_bits(majority) > 2) {
                query[i] = query[i] | (1 << z);
            }
        }
    }
}

/**
 * @brief Computes the number of 1's
 *
 * @param i The i-th variable that composes the hypervector
 * @return  Number of 1's in i-th variable of hypervector
 */
int
number_of_set_bits(uint32_t i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
