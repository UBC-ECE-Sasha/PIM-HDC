#!/bin/bash

# Set COUNTER_CONFIG env var:
#  export COUNTER_CONFIG=COUNT_INSTRUCTIONS
#     OR
#  export COUNTER_CONFIG=COUNT_CYCLES

# ../tools/bytes-per-counter/bytes_tls.sh 3848888 data/large8-data.bin 512

####

# ADD ___ DPU idx to dpu for correct matching

####

bytes="${1:-86528}"
file="${2:-"data/small-data.bin"}"
dpus="${3:-22}"

out=$(./pim_hdc -d -i "${file}")

declare -a largest_times
declare -a smallest_times

for ((i=0; i<dpus; i++)); do
    temp="$(echo "${out}" | \
        grep "___ DPU $i" | \
        grep 'tasklet [0-9]*: completed in' | \
        grep -o -P '(?<=in )[0-9]*(?= cycles)')"
    largest_times+=("$(echo "$temp" | sort -n | tail -1)")
    smallest_times+=("$(echo "$temp" | sort -n | head -1)")
done

count=0
for t in "${largest_times[@]}"; do
    echo $t
  count=$((count+t))
done

largest=$(IFS=$'\n'; echo "${largest_times[*]}" | sort -nr | head -n1)
smallest=$(IFS=$'\n'; echo "${smallest_times[*]}" | sort -nr | head -n1)

cpb=$(echo "${count} / ${bytes}" | bc -l)

echo "----- DPU ----- "

echo "count: ${count}"
echo "bytes: ${bytes}"

echo "${count} / ${bytes} = ${cpb}"

echo "slowest : fastest ratio"
echo "ratio = ${largest} : ${smallest}"

echo "----- Host ----- "
perf stat -B -e cycles:u,instructions:u ./pim_hdc -i "${file}"

# CYCLES / BYTE =
#     654599782096 / 1905024 = 343617.60381811462742726600

# INSTRUCTIONS / BYTE =
#     58003678064 / 1905024 = 30447.74137438688436471141
