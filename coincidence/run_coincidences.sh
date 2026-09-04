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
# optional argument lets related pipeline stages use the same identifier.
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
reference_runs=0
skipped_runs=()
for threshold_file in "${threshold_files[@]}"; do
    threshold_name=${threshold_file##*/}
    if [[ ! $threshold_name =~ ^equalized_run_([0-9]{6})_thresholds\.txt$ ]]; then
        continue
    fi

    run=${BASH_REMATCH[1]}
    input_file="$INPUT_DIR/input_run${run}.txt"
    if [[ ! -f $input_file ]]; then
        echo "Skipping run $run: missing input list $input_file" >&2
        skipped_runs+=("$run")
        continue
    fi

    echo "========================================"
    echo "Running coincidence analysis for run $run"
    echo "Input: $input_file"
    echo "Normalized-rate ADC thresholds: $threshold_file"
    echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
    echo "Analysis directory: $ANALYSIS_DIR"
    echo "========================================"

    if [[ $run != "043384" &&
          $run != "043385" &&
          $run != "043386" &&
          $run != "043389" &&
          $run != "043390" ]]; then
        echo "Skipping run $run: not in the allowed run list" >&2
        skipped_runs+=("$run")
        continue
    fi

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --timestamp "$ANALYSIS_TIMESTAMP" \
        --output-dir "$OUTPUT_ROOT" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2070 2071 2080 2081 \
        --channels-coincident-right 2050 2051 2060 2061 \
        --channels-to-save 1010 1011 1020 1021 1030 1031 1060 1061 1070 1071 \
        --window-ticks 10 \
        --min-amplitude-adc 0 \
        --norm-rate-adc-threshold-file "$threshold_file"

    ((processed_runs += 1))
done

# A reference sample defines the target rates and therefore has no equalized
# threshold table. Process each reference NPZ's run once without a rate cut.
reference_files=("$THRESHOLD_DIR"/reference_run_*.npz)
for reference_file in "${reference_files[@]}"; do
    reference_name=${reference_file##*/}
    if [[ ! $reference_name =~ ^reference_run_([0-9]{6})\.npz$ ]]; then
        continue
    fi

    run=${BASH_REMATCH[1]}
    if [[ -f "$THRESHOLD_DIR/equalized_run_${run}_thresholds.txt" ]]; then
        continue
    fi

    input_file="$INPUT_DIR/input_run${run}.txt"
    if [[ ! -f $input_file ]]; then
        echo "Skipping reference run $run: missing input list $input_file" >&2
        skipped_runs+=("$run")
        continue
    fi

    echo "========================================"
    echo "Running unthresholded reference analysis for run $run"
    echo "Input: $input_file"
    echo "Reference data: $reference_file"
    echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
    echo "Analysis directory: $ANALYSIS_DIR"
    echo "========================================"

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --timestamp "$ANALYSIS_TIMESTAMP" \
        --output-dir "$OUTPUT_ROOT" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2070 2071 2080 2081 \
        --channels-coincident-right 2050 2051 2060 2061 \
        --channels-to-save 1010 1011 1020 1021 1030 1031 1060 1061 1070 1071 \
        --window-ticks 10 \
        --min-amplitude-adc 0

    ((processed_runs += 1))
    ((reference_runs += 1))
done

if ((processed_runs == 0)); then
    echo "No runnable equalized or reference samples were found" >&2
    exit 1
fi

echo
echo "Coincidence analysis complete"
echo "Processed runs: $processed_runs"
echo "Unthresholded reference runs: $reference_runs"
echo "Skipped runs: ${skipped_runs[*]:-none}"
echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
echo "Saved outputs: $ANALYSIS_DIR"
