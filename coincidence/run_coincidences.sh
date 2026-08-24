#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXECUTABLE="$REPO_DIR/bin/run_coincidence"
INPUT_DIR="$REPO_DIR/input_lists"
THRESHOLD_DIR="$REPO_DIR/analysis/RateAnalysis_data"

make -C "$SCRIPT_DIR" run_coincidence

cd "$SCRIPT_DIR"

shopt -s nullglob
threshold_files=("$THRESHOLD_DIR"/equalized_run_*_thresholds.txt)
if ((${#threshold_files[@]} == 0)); then
    echo "No equalized threshold files found in $THRESHOLD_DIR" >&2
    exit 1
fi

for threshold_file in "${threshold_files[@]}"; do
    threshold_name=$(basename "$threshold_file")
    if [[ ! $threshold_name =~ ^equalized_run_([0-9]{6})_thresholds\.txt$ ]]; then
        continue
    fi
    run=${BASH_REMATCH[1]}
    input_file="$INPUT_DIR/input_run${run}.txt"
    if [[ ! -f $input_file ]]; then
        echo "Missing input list for run $run: $input_file" >&2
        exit 1
    fi

    echo "========================================"
    echo "Running coincidence analysis for run $run"
    echo "Input: $input_file"
    echo "Save thresholds: $threshold_file"
    echo "========================================"

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2030 2031 2040 2041 \
        --channels-coincident-right 2050 2051 2060 2061 \
        --channels-to-save 2080 2081 \
        --window-ticks 10 \
        --min-amplitude-adc 0 \
        --save-threshold-file "$threshold_file"

done
