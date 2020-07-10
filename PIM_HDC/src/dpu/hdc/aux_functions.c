#include <string.h>
#include <built_ins.h>
#include <alloc.h>
#include <mutex.h>
#include <stdio.h>

#include "aux_functions.h"
#include "cycle_counter.h"
#include "common.h"

MUTEX_INIT(chHV_mutex);

#define BUILTIN_CAO

/**
 * @brief Computes the maximum Hamming Distance.
 *
 * @param[in] distances Distances associated to each class
 * @return              The class related to the maximum distance
 */
int max_dist_hamm(int distances[CLASSES]) {
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
void hamming_dist(uint32_t q[bit_dim + 1], uint32_t *aM, int sims[CLASSES]){
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < bit_dim + 1; j++) {
            sims[i] += number_of_set_bits(q[j] ^ aM[A2D1D(bit_dim + 1, i, j)]);
        }
    }
}

// Original array lengths
// double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
// uint32_t chAM[CHANNELS][BIT_DIM + 1];
// uint32_t iM[IM_LENGTH][BIT_DIM + 1];
// uint32_t aM_32[N][BIT_DIM + 1];

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[in] channel_iM  Item Memory for the IDs of @p CHANNELS
 * @param[in] channel_AM  Continuous Item Memory for the values of a channel
 * @param[out] query      Query hypervector
 */
void compute_N_gram(int32_t input[channels], uint32_t *channel_iM, uint32_t *channel_AM, uint32_t query[bit_dim + 1]) {

    // Pseudo-2d array:

    uint32_t chHV[MAX_CHANNELS + 1] = {0};

    for (int i = 0; i < bit_dim + 1; i++) {
        query[i] = 0;
        for (int j = 0; j < channels; j++) {
            int32_t ix = input[j];

            // TODO: Why is this needed?
            mutex_lock(chHV_mutex);
            chHV[j] = channel_iM[A2D1D(bit_dim + 1, ix, i)] ^ channel_AM[A2D1D(bit_dim + 1, j, i)];
            mutex_unlock(chHV_mutex);
        }

        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[channels] = chHV[0] ^ chHV[1];
        // componentwise majority: insert the value of the ith bit of each chHV row in the variable "majority"
        // and then compute the number of 1's with the function numberOfSetBits(uint32_t).
        for (int z = 31; z >= 0; z--) {
            uint32_t majority = 0;
            for (int j = 0 ; j < channels + 1; j++) {
                majority = majority | (((chHV[j] >> z) & 1) << j);
            }

            if (number_of_set_bits(majority) > 2) {
                query[i] = query[i] | ( 1 << z );
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
inline int number_of_set_bits(uint32_t i) {
    int set_bits;
#ifdef BUILTIN_CAO
    // Retrieve number of set bits (count all ones)
    __builtin_cao_rr(set_bits, i);
#else
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     set_bits = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
    return set_bits;
}
