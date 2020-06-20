# Data Generator

## Summary

`datagen` generates binary files for usage in `PIM_HDC`.

```
usage: ./datagen [ -o <output> ] [ -t ]
        o: redirect output to file
        t: validate data after creating
```

## Usage

Place the "input header" in `data/data.h`.

Example datasets can be copied from `data/*.example` to `data/data.h`.

It should contain the following constants:

* `DIMENSION`
* `CHANNELS`
* `BIT_DIM`
* `NUMBER_OF_INPUT_SAMPLES`
* `N`
* `IM_LENGTH`

It should contain the following arrays:

```c
double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
uint32_t chAM[CHANNELS][BIT_DIM + 1];
uint32_t iM[IM_LENGTH][BIT_DIM + 1];
uint32_t aM_32[N][BIT_DIM + 1];
```

## Output Layout

An output file with a specific layout will be created:

* Versioned binary format (first 4 bytes)

Next `(CONSTANTS-1 * sizeof(int))` bytes, in this order

* `DIMENSION`
* `CHANNELS`
* `BIT_DIM`
* `NUMBER_OF_INPUT_SAMPLES`
* `N`
* `IM_LENGTH`

Followed by, in this order:

```c
double TEST_SET[CHANNELS][NUMBER_OF_INPUT_SAMPLES];
uint32_t chAM[CHANNELS][BIT_DIM + 1];
uint32_t iM[IM_LENGTH][BIT_DIM + 1];
uint32_t aM_32[N][BIT_DIM + 1];
```
