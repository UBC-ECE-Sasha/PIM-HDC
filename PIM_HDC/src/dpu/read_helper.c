#include "read_helper.h"


static inline void reader_reset(in_buffer_context * ctx) {
    ctx->ptr = seqread_seek(ctx->mram_addr, &ctx->sr);
    ctx->curr++;
}

/**
 * Read the next input byte from the sequential reader.
 *
 * @param _i: holds input buffer information
 * @return Byte that was read
 */
inline uint32_t read_uint32(in_buffer_context * ctx) {
    uint32_t ret = * ctx->ptr;
    if (ctx->curr < ctx->length) {
        ctx->ptr = seqread_get(ctx->ptr, sizeof(uint32_t), &ctx->sr);
        ctx->curr++;
    } else {
        ctx->curr = 0;
        // Roll back to start
        reader_reset(ctx);
    }
    return ret;
}


