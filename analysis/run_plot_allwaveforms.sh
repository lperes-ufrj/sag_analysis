#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
INPUT_DIR="$REPO_DIR/coincidence/input_lists"
PLOT_SCRIPT="$SCRIPT_DIR/plot_allwaveforms.py"

if [[ ! -f $PLOT_SCRIPT ]]; then
    echo "Error: plotting script not found: $PLOT_SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
input_files=("$INPUT_DIR"/input_run*.txt)

if ((${#input_files[@]} == 0)); then
    echo "Error: no input files found in $INPUT_DIR" >&2
    exit 1
fi

successful_runs=()
failed_runs=()
total=${#input_files[@]}

for index in "${!input_files[@]}"; do
    input_file=${input_files[$index]}
    filename=${input_file##*/}
    run=${filename#input_run}
    run=${run%.txt}

    printf '\nProcessing run %s (%d/%d)\n' \
        "$run" "$((index + 1))" "$total"

    if python3 "$PLOT_SCRIPT" "$input_file"; then
        successful_runs+=("$run")
    else
        echo "Error: plotting failed for run $run" >&2
        failed_runs+=("$run")
    fi
done

printf '\nSuccessful runs (%d): %s\n' \
    "${#successful_runs[@]}" "${successful_runs[*]:-none}"
printf 'Failed runs     (%d): %s\n' \
    "${#failed_runs[@]}" "${failed_runs[*]:-none}"

if ((${#failed_runs[@]} > 0)); then
    exit 1
fi
