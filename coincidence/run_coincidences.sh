#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXECUTABLE="$REPO_DIR/bin/run_coincidence"
INPUT_DIR="$REPO_DIR/input_lists"
THRESHOLD_DIR="$REPO_DIR/analysis/RateAnalysis_data"

if (($# > 1)); then
    echo "Usage: $0 [ANALYSIS_TIMESTAMP]" >&2
    exit 2
fi

ANALYSIS_TIMESTAMP=${1:-$(date +%Y%m%d_%H%M%S)}
ANALYSIS_DIR="$SCRIPT_DIR/saved_coincidences/$ANALYSIS_TIMESTAMP"

make -C "$SCRIPT_DIR" run_coincidence

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
    echo "Normalized-rate ADC thresholds: $threshold_file"
    echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
    echo "Analysis directory: $ANALYSIS_DIR"
    echo "========================================"

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --timestamp "$ANALYSIS_TIMESTAMP" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2030 2031 2040 2041 \
        --channels-coincident-right 2070 2071 2080 2081 \
        --channels-to-save 2050 2051 2060 2061  \
        --window-ticks 10 \
        --min-amplitude-adc 0 \
        --norm-rate-adc-threshold-file "$threshold_file"

done

echo
echo "Coincidence analysis complete"
echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
echo "Saved outputs: $ANALYSIS_DIR"
