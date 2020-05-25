CC           = dpu-upmem-dpurte-clang
CFLAGS       = --std=c11 -O2 -flto -g -Wall

# define DEBUG in the source if we are debugging
ifeq ($(DEBUG), 1)
	CFLAGS+=-DDEBUG
endif

SOURCES = dpu_task.c $(wildcard ../hdc/*.c)
HDC_DPU = hdc.dpu

INC=-I../../include

.PHONY: default all clean

default: all

all: $(HDC_DPU)

clean:
	$(RM) $(HDC_DPU)

$(HDC_DPU): $(SOURCES)
	$(CC) $(CFLAGS) $(INC)  $^ -o $@
