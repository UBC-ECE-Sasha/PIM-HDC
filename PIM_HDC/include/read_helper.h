#ifndef READ_HELPER_H_
#define READ_HELPER_H_

#include <seqread.h>

typedef struct in_buffer_context {
    uint32_t *ptr;
    seqreader_buffer_t cache;
    seqreader_t sr;
    uint32_t curr;
    uint32_t length;
    uint32_t __mram_ptr * mram_addr;
} in_buffer_context;

uint32_t read_uint32(in_buffer_context * ctx);

#endif // READ_HELPER_H_
