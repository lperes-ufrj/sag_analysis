#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXECUTABLE="$REPO_DIR/bin/run_coincidence"
INPUT_DIR="$SCRIPT_DIR/input_lists/xe_input_lists"
OUTPUT_ROOT="$SCRIPT_DIR/saved_coincidences"

if (($# > 1)); then
    echo "Usage: $0 [ANALYSIS_TIMESTAMP]" >&2
    exit 2
fi

# Use one identifier and summary file for every xenon sample in the batch.
ANALYSIS_TIMESTAMP=${1:-$(date +%Y%m%d_%H%M%S)}
ANALYSIS_DIR="$OUTPUT_ROOT/$ANALYSIS_TIMESTAMP"

make -C "$SCRIPT_DIR" run_coincidence

shopt -s nullglob
input_files=("$INPUT_DIR"/input_run*.txt)
if ((${#input_files[@]} == 0)); then
    echo "No xenon input lists found in $INPUT_DIR" >&2
    exit 1
fi

processed_runs=0
for input_file in "${input_files[@]}"; do
    input_name=${input_file##*/}
    if [[ ! $input_name =~ ^input_run([0-9]{6})\.txt$ ]]; then
        continue
    fi

    run=${BASH_REMATCH[1]}
    concentration="unknown"
    # ROOT filenames encode decimal concentrations with 'p': 0p01ppm = 0.01 ppm.
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
        if [[ $line =~ _([0-9]+(p[0-9]+)?)ppm_ ]]; then
            concentration="${BASH_REMATCH[1]//p/.} ppm"
            break
        fi
    done < "$input_file"

    echo "========================================"
    echo "Running coincidence analysis for run $run"
    echo "Xenon concentration: $concentration"
    echo "Input: $input_file"
    echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
    echo "Analysis directory: $ANALYSIS_DIR"
    echo "========================================"

    "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --timestamp "$ANALYSIS_TIMESTAMP" \
        --output-dir "$OUTPUT_ROOT" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --channels-coincident-left 2070 2071 2080 2081 \
        --channels-coincident-right 2010 2011 2020 2021 \
        --channels-to-save 2050 2051 2060 2061 \
        --window-ticks 10 \
        --min-amplitude-adc 0

    ((processed_runs += 1))
done

if ((processed_runs == 0)); then
    echo "No runnable xenon samples were found" >&2
    exit 1
fi

echo
echo "Coincidence analysis complete"
echo "Processed runs: $processed_runs"
echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
echo "Saved outputs: $ANALYSIS_DIR"
