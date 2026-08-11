#!/bin/bash

for input_file in input_lists/input_run*.txt; do

    run=$(basename "$input_file" | sed -n 's/input_run\([0-9]\{6\}\)\.txt/\1/p')

    echo "========================================"
    echo "Running coincidence analysis for run $run"
    echo "Input: $input_file"
    echo "========================================"

    ./run_coincidence "$input_file" \
        --run "$run" \
        --config waveform_intervals.ini \
        --channels-coincident-left 2030 2031 2040 2041 \
        --channels-coincident-right 2050 2051 2060 2061 \
        --channels-to-save 2070 2071 2080 2081 \
        --window-ticks 10 \
        --min-amplitude-adc 0 \

done