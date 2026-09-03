#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXECUTABLE="$REPO_DIR/bin/run_coincidence"
INPUT_DIR="$SCRIPT_DIR/input_lists"
THRESHOLD_DIR="$REPO_DIR/analysis/RateAnalysis_data"
OUTPUT_ROOT="$SCRIPT_DIR/saved_coincidences"

if (($# > 1)); then
    echo "Usage: $0 [ANALYSIS_TIMESTAMP]" >&2
    exit 2
fi

# Generate this identifier once so every run in the batch is written into the
# same directory and appended to the same summary file. Supplying it as the
# optional argument makes a batch reproducible or resumable.
ANALYSIS_TIMESTAMP=${1:-$(date +%Y%m%d_%H%M%S)}
ANALYSIS_DIR="$OUTPUT_ROOT/$ANALYSIS_TIMESTAMP"

make -C "$SCRIPT_DIR" run_coincidence

shopt -s nullglob
threshold_files=("$THRESHOLD_DIR"/equalized_run_*_thresholds.txt)
if ((${#threshold_files[@]} == 0)); then
    echo "No equalized threshold files found in $THRESHOLD_DIR" >&2
    exit 1
fi

processed_runs=0
for threshold_file in "${threshold_files[@]}"; do
    threshold_name=${threshold_file##*/}
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
        --output-dir "$OUTPUT_ROOT" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2030 2031 2040 2041 \
        --channels-coincident-right 2050 2051 2060 2061 \
        --channels-to-save 2070 2071 2080 2081 \
        --window-ticks 10 \
        --min-amplitude-adc 0 \
        --norm-rate-adc-threshold-file "$threshold_file"

    ((processed_runs += 1))
done

if ((processed_runs == 0)); then
    echo "No threshold filenames matched equalized_run_RUN_thresholds.txt" >&2
    exit 1
fi

echo
echo "Coincidence analysis complete"
echo "Processed runs: $processed_runs"
echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
echo "Saved outputs: $ANALYSIS_DIR"
