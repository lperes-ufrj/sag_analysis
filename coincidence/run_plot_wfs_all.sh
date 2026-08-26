#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INPUT_DIR="$SCRIPT_DIR/input_lists"
OUTPUT_DIR="$SCRIPT_DIR/selected_waveforms"
EXECUTABLE="$SCRIPT_DIR/../bin/plot_wfs_coincidence"
CSV_SUFFIX="coinc_2070-2071-2080-2081_vs_2010-2021_save_1020-1021-1040-1041-1060-1061-1080-1081-2030-2031-2040-2041-2050-2051-2060-2061_window_10_ticks_min_amplitude_0_adc.csv"
MAX_AUXILIARY_AMPLITUDE="500"

# Compile only when the executable is missing or its sources changed.
make -C "$SCRIPT_DIR" ../bin/plot_wfs_coincidence || exit 1

# Ensure the executable loads the same ROOT libraries used by root-config.
if command -v root-config >/dev/null 2>&1; then
    ROOT_LIBRARY_DIR=$(root-config --libdir)
    export LD_LIBRARY_PATH="$ROOT_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
input_files=("$INPUT_DIR"/input_run*.txt)

if ((${#input_files[@]} == 0)); then
    echo "Error: no input lists found in $INPUT_DIR" >&2
    exit 1
fi

successful_runs=()
failed_runs=()
total=${#input_files[@]}

for index in "${!input_files[@]}"; do
    input_file=${input_files[$index]}
    filename=${input_file##*/}

    if [[ $filename =~ ^input_run([0-9]{6})\.txt$ ]]; then
        run=${BASH_REMATCH[1]}
    else
        echo "Skipping unrecognized input-list filename: $filename" >&2
        continue
    fi

    selection_csv="$SCRIPT_DIR/waveforms_run_${run}_${CSV_SUFFIX}"

    echo
    echo "============================================================"
    printf 'Processing run %s (%d/%d)\n' "$run" "$((index + 1))" "$total"
    echo "Input list:    $input_file"
    echo "Selection CSV: $selection_csv"
    echo "Output:        $OUTPUT_DIR"
    echo "============================================================"

    if [[ ! -f $selection_csv ]]; then
        echo "Error: matching selection CSV was not found" >&2
        failed_runs+=("$run")
        continue
    fi

    if "$EXECUTABLE" \
        --output-dir "$OUTPUT_DIR" \
        --csv-suffix "$CSV_SUFFIX" \
        --max-auxiliary-amplitude "$MAX_AUXILIARY_AMPLITUDE" \
        "$input_file"; then
        successful_runs+=("$run")
    else
        echo "Error: waveform plotting failed for run $run" >&2
        failed_runs+=("$run")
    fi
done

echo
echo "========================= Summary =========================="
echo "Successful runs (${#successful_runs[@]}): ${successful_runs[*]:-none}"
echo "Failed runs     (${#failed_runs[@]}): ${failed_runs[*]:-none}"

if ((${#failed_runs[@]} > 0)); then
    exit 1
fi
