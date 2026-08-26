#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXECUTABLE="$SCRIPT_DIR/../bin/run_coincidence"

make -C "$SCRIPT_DIR" ../bin/run_coincidence || exit 1

cd "$SCRIPT_DIR" || exit 1

for input_file in input_lists/input_run*.txt; do

    run=$(basename "$input_file" | sed -n 's/input_run\([0-9]\{6\}\)\.txt/\1/p')

    echo "========================================"
    echo "Running coincidence analysis for run $run"
    echo "Input: $input_file"
    echo "========================================"

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --config waveform_intervals.ini \
        --channels-coincident-left 2070 2071 2080 2081 \
        --channels-coincident-right 2010 2021 \
        --channels-to-save 1020 1021 1040 1041 1060 1061 1080 1081 2030 2031 2040 2041 2050 2051 2060 2061 \
        --window-ticks 10 \
        --min-amplitude-adc 0

done
