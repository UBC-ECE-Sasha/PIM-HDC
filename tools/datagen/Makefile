INC = data
CC = gcc
CFLAGS = --std=c11 -O3 -g -Wall -Wextra -I$(INC)

ODIR=obj

TARGET = datagen

default: $(TARGET)
all: default

SOURCES = $(wildcard *.c)
OBJECTS = $(patsubst %.c, %.o, $(SOURCES))

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o $(TARGET)