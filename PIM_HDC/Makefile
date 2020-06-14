CC = gcc
CFLAGS = --std=c11 -O3 -g -Wall -Wextra
DPU_OPTS = `dpu-pkg-config --cflags --libs dpu`
TARGET = pim_hdc

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif
# define TEST in the source if we are debugging
ifeq ($(TEST), 1)
	CFLAGS+=-DTEST
endif

# Default
NR_DPUS ?= 32

.PHONY: default all clean

default: $(TARGET) dpu
all: default

SOURCES = $(wildcard src/*.c) $(wildcard src/hdc/*.c)
OBJECTS = $(patsubst %.c, %.o, $(SOURCES))
HEADERS = $(wildcard include/*.h)
INC=-Iinclude -IPIM-common/common/include

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -DNR_DPUS=$(NR_DPUS) $(INC) -c $< -o $@ $(DPU_OPTS)

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) -o $@ $(DPU_OPTS)

dpu:
	DEBUG=$(DEBUG) NR_DPUS=$(NR_DPUS) TEST=$(TEST) $(MAKE) -C src/dpu

clean:
	find . -type f -name '*.o' -delete
	-rm -f $(TARGET)
	$(MAKE) -C src/dpu clean
