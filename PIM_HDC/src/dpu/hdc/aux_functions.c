#include "aux_functions.h"
#include "global_dpu.h"
#include "cycle_counter.h"
#include "built_ins.h"

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
void hamming_dist(uint32_t q[BIT_DIM + 1], uint32_t aM[][BIT_DIM + 1], int sims[CLASSES]){
    
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < BIT_DIM + 1; j++) {
            uint32_t tmp = q[j] ^ aM[i][j];

#ifdef BUILTIN_CAO
            int set_bits;
            // Retrieve number of set bits (count all ones)
            __builtin_cao_rr(set_bits, tmp);
            sims[i] += set_bits;
#else
            sims[i] += number_of_set_bits(tmp);
#endif
        }
    }
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[in] channel_iM  Item Memory for the IDs of @p CHANNELS
 * @param[in] channel_AM  Continuous Item Memory for the values of a channel
 * @param[out] query      Query hypervector
 */
void compute_N_gram(int32_t input[CHANNELS], uint32_t channel_iM[][BIT_DIM + 1], uint32_t channel_AM[][BIT_DIM + 1], uint32_t query[BIT_DIM + 1]) {
    uint32_t chHV[CHANNELS + 1][BIT_DIM + 1] = {0};

    for (int i = 0; i < BIT_DIM + 1; i++) {
        query[i] = 0;

        for (int j = 0; j < CHANNELS; j++) {
            int ix = input[j];
            chHV[j][i] = channel_iM[ix][i] ^ channel_AM[j][i];
        }

        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[CHANNELS][i] = chHV[0][i] ^ chHV[1][i];

        // componentwise majority: insert the value of the ith bit of each chHV row in the variable "majority"
        // and then compute the number of 1's with the function numberOfSetBits(uint32_t).
        for (int z = 31; z >= 0; z--) {
            uint32_t majority = 0;
            for (int j = 0 ; j < CHANNELS + 1; j++) {
                majority = majority | (((chHV[j][i] >> z) & 1) << j);
            }
            int set_bits;
#ifdef BUILTIN_CAO
            // Retrieve number of set bits (count all ones)
            __builtin_cao_rr(set_bits, majority);
#else
            set_bits = number_of_set_bits(majority);
#endif
            if (set_bits > 2) {
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
int number_of_set_bits(uint32_t i) {
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
