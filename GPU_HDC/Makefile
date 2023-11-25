CC = nvcc
CFLAGS = -lm -O3 -g -diag-suppress 2464,68
TARGET = gpu_hdc

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif
# define TEST in the source if we are debugging
ifeq ($(TEST), 1)
	CFLAGS+=-DTEST
endif

# Default
NR_BLOCKS ?= 768
NR_THREADS ?= 128

MAX_BIT_DIM ?= 313
MAX_CHANNELS ?= 4
MAX_N ?= 5
MAX_IM_LENGTH ?= 22
HDC_MAX_INPUT ?= 4096

CFLAGS+=-DNR_BLOCKS=$(NR_BLOCKS) -DNR_THREADS=$(NR_THREADS) -DHOST=1
CFLAGS+=-DMAX_BIT_DIM=$(MAX_BIT_DIM) -DMAX_CHANNELS=$(MAX_CHANNELS)
CFLAGS+=-DMAX_N=$(MAX_N) -DMAX_IM_LENGTH=$(MAX_IM_LENGTH)
CFLAGS+=-DCOUNTER_CONFIG=$(COUNTER_CONFIG) -DHDC_MAX_INPUT=$(HDC_MAX_INPUT)

# For clock_gettime
CFLAGS += -D_POSIX_C_SOURCE=199309L

.PHONY: default all clean

default: $(TARGET)
all: default

SOURCES = $(wildcard src/*.cu) $(wildcard src/hdc/*.cu) $(wildcard src/gpu/*.cu)
OBJECTS = $(patsubst %.cu, %.o, $(SOURCES))
HEADERS = $(wildcard include/*.h)
INC=-Iinclude -I../PIM_HDC/PIM-common/common/include -I../PIM_HDC/PIM-common/host/include -I/opt/cuda/include -I/cs/local/lib/pkg/cudatoolkit/include

%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) -o $@

clean:
	find . -type f -name '*.o' -delete
	-rm -f $(TARGET)
