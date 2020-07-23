#include "aux_functions.h"

#include "common.h"
#include "cycle_counter.h"
#include "global_dpu.h"

#include <alloc.h>
#include <built_ins.h>
#include <mram.h>
#include <mutex.h>
#include <stdio.h>
#include <string.h>

#define BUILTIN_CAO
#define MINIMUM_MRAM_32B_READ 2

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
 * @brief Read 32 bits from MRAM
 *
 * @param[in] ind         Read index
 * @param[in] mram_buf    MRAM array to read from at @p ind
 * @return                32bit read from @p mram_buf
 */
static uint32_t
read_32bits_from_mram(uint32_t ind, uint32_t __mram_ptr * mram_buf) {
    __dma_aligned uint32_t buf[MINIMUM_MRAM_32B_READ];
    mram_read(&mram_buf[ind], buf, MINIMUM_MRAM_32B_READ * sizeof(uint32_t));

    /* Data will be offset if the data address is not 8 byte aligned */
    return buf[(ind % 2) != 0];
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[out] query      Query hypervector
 */
void
compute_N_gram(int32_t input[hd.channels], uint32_t query[hd.bit_dim + 1]) {

    uint32_t chHV[MAX_CHANNELS + 1] = {0};

    for (int i = 0; i < hd.bit_dim + 1; i++) {
        query[i] = 0;

        for (int j = 0; j < hd.channels; j++) {
            int32_t ix = input[j];
            uint32_t im;
            uint32_t cham;

            uint32_t im_ind = A2D1D(hd.bit_dim + 1, ix, i);
#ifdef IM_IN_WRAM
            im = hd.iM[im_ind];
#else
            im = read_32bits_from_mram(im_ind, mram_iM);
#endif

            uint32_t cham_ind = A2D1D(hd.bit_dim + 1, j, i);
#ifdef CHAM_IN_WRAM
            cham = hd.chAM[cham_ind];
#else
            cham = read_32bits_from_mram(cham_ind, mram_chAM);
#endif
            chHV[j] = im ^ cham;
        }

        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[hd.channels] = chHV[0] ^ chHV[1];
        
        // componentwise majority: insert the value of the ith bit of each chHV row in the variable
        // "majority" and then compute the number of 1's
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

/**
 * @brief Computes the number of 1's
 *
 * @param i The i-th variable that composes the hypervector
 * @return  Number of 1's in i-th variable of hypervector
 */
inline int
number_of_set_bits(uint32_t i) {
#ifdef BUILTIN_CAO
    // Retrieve number of set bits (count all ones)
    return __builtin_popcount(i);
#else
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}
