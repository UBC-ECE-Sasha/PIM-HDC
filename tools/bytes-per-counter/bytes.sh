#!/bin/sh

# Set COUNTER_CONFIG env var

# Bytes = CHANNELS * NUMBER_OF_INPUT_SAMPLES * sizeof(int32_t)
# ie: 4 * 119064 * 4 = 1905024

bytes="${1:-1905024}"
file="${2:-"data/large8-data.bin"}"

count=$(./pim_hdc -d -i "${file}" | \
grep 'Tasklet [0-9]*: completed in' | \
grep -o -P '(?<=in )[0-9]*(?= cycles)' | \
awk '{ sum+=$1} END {print sum}')

cpb=$(echo "${count} / ${bytes}" | bc -l)

echo "count: ${count}"
echo "bytes: ${bytes}"

echo "${count} / ${bytes} = ${cpb}"

# CYCLES / BYTE =
#     654599782096 / 1905024 = 343617.60381811462742726600

# INSTRUCTIONS / BYTE =
#     58003678064 / 1905024 = 30447.74137438688436471141
