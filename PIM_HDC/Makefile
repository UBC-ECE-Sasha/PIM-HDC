CC = gcc
CFLAGS = --std=c11 -O3 -g -Wall -Wextra
DPU_OPTS = `dpu-pkg-config --cflags --libs dpu`
TARGET = pim_hdc

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif

.PHONY: default all clean

default: $(TARGET) dpu
all: default

SOURCES = $(wildcard src/*.c) $(wildcard src/hdc/*.c)
OBJECTS = $(patsubst %.c, %.o, $(SOURCES))
HEADERS = $(wildcard include/*.h)
INC=-Iinclude -IPIM-common/common/include

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@ $(DPU_OPTS)

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(INC) -o $@ $(DPU_OPTS)

dpu:
	DEBUG=$(DEBUG) NR_TASKLETS=$(NR_TASKLETS) $(MAKE) -C src/dpu

clean:
	find . -type f -name '*.o' -delete
	-rm -f $(TARGET)
