#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
INPUT_DIR="$REPO_DIR/input_lists"
EXECUTABLE="$REPO_DIR/bin/plot_wfs_coincidence"
MAX_AUXILIARY_AMPLITUDE="1000"

if (($# != 1)); then
    echo "Usage: $0 ANALYSIS_TIMESTAMP" >&2
    echo "Example: $0 20260828_132850" >&2
    exit 2
fi

ANALYSIS_TIMESTAMP=$1
ANALYSIS_DIR="$SCRIPT_DIR/saved_coincidences/$ANALYSIS_TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/selected_waveforms/$ANALYSIS_TIMESTAMP"

if [[ ! -d $ANALYSIS_DIR ]]; then
    echo "Error: analysis directory not found: $ANALYSIS_DIR" >&2
    exit 1
fi

# Compile only when the executable is missing or its sources changed.
make -C "$SCRIPT_DIR" plot_wfs_coincidence || exit 1

# Ensure the executable loads the same ROOT libraries used by root-config.
if command -v root-config >/dev/null 2>&1; then
    ROOT_LIBRARY_DIR=$(root-config --libdir)
    export LD_LIBRARY_PATH="$ROOT_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
selection_csvs=(
    "$ANALYSIS_DIR"/coincidence_scan_run_*_"$ANALYSIS_TIMESTAMP".csv
)

if ((${#selection_csvs[@]} == 0)); then
    echo "Error: no timestamped selection CSVs found in $ANALYSIS_DIR" >&2
    exit 1
fi

successful_runs=()
failed_runs=()
total=${#selection_csvs[@]}

for index in "${!selection_csvs[@]}"; do
    selection_csv=${selection_csvs[$index]}
    filename=${selection_csv##*/}

    if [[ $filename =~ ^coincidence_scan_run_([0-9]{6})_(.+)\.csv$ ]] \
        && [[ ${BASH_REMATCH[2]} == "$ANALYSIS_TIMESTAMP" ]]; then
        run=${BASH_REMATCH[1]}
    else
        echo "Skipping unrecognized selection CSV: $filename" >&2
        continue
    fi

    input_file="$INPUT_DIR/input_run${run}.txt"
    if [[ ! -f $input_file ]]; then
        echo "Error: input list not found for run $run: $input_file" >&2
        failed_runs+=("$run")
        continue
    fi

    echo
    echo "============================================================"
    printf 'Processing run %s (%d/%d)\n' "$run" "$((index + 1))" "$total"
    echo "Input list:    $input_file"
    echo "Selection CSV: $selection_csv"
    echo "Output:        $OUTPUT_DIR"
    echo "============================================================"

    if "$EXECUTABLE" \
        --config "$SCRIPT_DIR/waveform_intervals.ini" \
        --output-dir "$OUTPUT_DIR" \
        --csv "$selection_csv" \
        --max-peaks-signal-region 2 \
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
