CC           = dpu-upmem-dpurte-clang
CFLAGS       = --std=c11 -O2 -flto -g -Wall

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif
# define TEST in the source if we are debugging
ifeq ($(TEST), 1)
	CFLAGS+=-DTEST
endif

CFLAGS += -DNR_DPUS=$(NR_DPUS) -DNR_TASKLETS=$(NR_TASKLETS) -DSTACK_SIZE_DEFAULT=3072 -DMAX_BIT_DIM=$(MAX_BIT_DIM) -DMAX_CHANNELS=$(MAX_CHANNELS)
# To Set stack: -DSTACK_SIZE_DEFAULT=<size>

SOURCES = $(wildcard *.c) $(wildcard hdc/*.c)
HDC_DPU = hdc.dpu

INC=-I../../include  -I../../PIM-common/common/include

.PHONY: default all clean

default: all

all: $(HDC_DPU)

clean:
	$(RM) $(HDC_DPU)

$(HDC_DPU): $(SOURCES)
	$(CC) $(CFLAGS) $(INC)  $^ -o $@
