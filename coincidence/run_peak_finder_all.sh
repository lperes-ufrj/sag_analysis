#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
EXECUTABLE="$REPO_DIR/bin/run_peak_finder"
INPUT_DIR="$SCRIPT_DIR/input_lists"
CONFIG="$SCRIPT_DIR/waveform_intervals.ini"
OUTPUT_ROOT="$SCRIPT_DIR/peak_finder_results"
SATURATION_ADC=16383
CHANNELS=(2080 2081)
CUSTOM_CHANNELS=0
ALL_CHANNELS=0

usage() {
    cat <<'EOF'
Usage: run_peak_finder_all.sh [ANALYSIS_TIMESTAMP] [options]

Run the peak finder without coincidence for every input_runNNNNNN.txt list.
Defaults: channels 2080 and 2081, saturation at raw ADC >= 16383.
All runs share one timestamp (generated automatically if omitted).

Options:
  --channel CH        Replace default channels; repeat for multiple channels
  --all-channels      Process all channels instead of the default pair
  --saturation-adc N  Raw ADC saturation threshold (1..32767)
  --input-dir DIR     Input-list directory (default: coincidence/input_lists)
  --config FILE       Waveform interval INI (default: waveform_intervals.ini)
  --output-dir DIR    Output base (default: coincidence/peak_finder_results)
  -h, --help          Show this help
EOF
}

fail_usage() {
    echo "Error: $*" >&2
    usage >&2
    exit 2
}

ANALYSIS_TIMESTAMP=""
if (($# > 0)) && [[ $1 != -* ]]; then
    ANALYSIS_TIMESTAMP=$1
    shift
    [[ $ANALYSIS_TIMESTAMP =~ ^[A-Za-z0-9_-]+$ ]] \
        || fail_usage "ANALYSIS_TIMESTAMP may contain only letters, digits, '-' and '_'"
fi

while (($# > 0)); do
    option=$1
    case "$option" in
        -h|--help)
            usage
            exit 0
            ;;
        --all-channels)
            ((CUSTOM_CHANNELS == 0)) || fail_usage "Cannot combine --all-channels and --channel"
            ALL_CHANNELS=1
            CHANNELS=()
            shift
            ;;
        --channel|--saturation-adc|--input-dir|--config|--output-dir)
            (($# >= 2)) && [[ -n $2 && $2 != --* ]] \
                || fail_usage "Missing value for $option"
            value=$2
            case "$option" in
                --channel)
                    ((ALL_CHANNELS == 0)) || fail_usage "Cannot combine --all-channels and --channel"
                    [[ $value =~ ^[0-9]{1,10}$ ]] && ((10#$value <= 4294967295)) \
                        || fail_usage "--channel requires an unsigned 32-bit channel number"
                    if ((CUSTOM_CHANNELS == 0)); then
                        CHANNELS=()
                        CUSTOM_CHANNELS=1
                    fi
                    CHANNELS+=("$value")
                    ;;
                --saturation-adc)
                    [[ $value =~ ^[0-9]{1,5}$ ]] && ((10#$value >= 1 && 10#$value <= 32767)) \
                        || fail_usage "--saturation-adc must be between 1 and 32767"
                    SATURATION_ADC=$value
                    ;;
                --input-dir) INPUT_DIR=$value ;;
                --config) CONFIG=$value ;;
                --output-dir) OUTPUT_ROOT=$value ;;
            esac
            shift 2
            ;;
        *) fail_usage "Unknown argument: $option" ;;
    esac
done

if [[ -z $ANALYSIS_TIMESTAMP ]]; then
    ANALYSIS_TIMESTAMP=$(date +%Y%m%d_%H%M%S) || exit 1
fi
[[ -d $INPUT_DIR ]] || { echo "Error: input directory not found: $INPUT_DIR" >&2; exit 1; }
[[ -f $CONFIG ]] || { echo "Error: configuration not found: $CONFIG" >&2; exit 1; }

shopt -s nullglob
input_lists=()
for input_file in "$INPUT_DIR"/input_run*.txt; do
    filename=${input_file##*/}
    if [[ $filename =~ ^input_run[0-9]{6}\.txt$ && -f $input_file ]]; then
        input_lists+=("$input_file")
    fi
done
if ((${#input_lists[@]} == 0)); then
    echo "Error: no input_runNNNNNN.txt lists found in $INPUT_DIR" >&2
    exit 1
fi

# Rebuild only when the executable is missing or its sources changed.
make -C "$SCRIPT_DIR" run_peak_finder || exit 1
if command -v root-config >/dev/null 2>&1; then
    ROOT_LIBRARY_DIR=$(root-config --libdir) || exit 1
    export LD_LIBRARY_PATH="$ROOT_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
ANALYSIS_DIR="$OUTPUT_ROOT/$ANALYSIS_TIMESTAMP"
mkdir -p "$ANALYSIS_DIR" || exit 1

channel_arguments=()
for channel in "${CHANNELS[@]}"; do
    channel_arguments+=(--channel "$channel")
done
successful_runs=()
failed_runs=()
total=${#input_lists[@]}

for index in "${!input_lists[@]}"; do
    input_file=${input_lists[$index]}
    filename=${input_file##*/}
    run=${filename#input_run}
    run=${run%.txt}

    echo
    echo "============================================================"
    printf 'Processing run %s (%d/%d)\n' "$run" "$((index + 1))" "$total"
    echo "Input list: $input_file"
    echo "Channels: ${CHANNELS[*]:-all}"
    echo "Raw ADC saturation threshold: $SATURATION_ADC"
    echo "Output: $ANALYSIS_DIR"
    echo "============================================================"

    if "$EXECUTABLE" "$input_file" \
        --run "$run" \
        --timestamp "$ANALYSIS_TIMESTAMP" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT_ROOT" \
        --saturation-adc "$SATURATION_ADC" \
        "${channel_arguments[@]}"; then
        successful_runs+=("$run")
    else
        echo "Error: peak finder failed for run $run" >&2
        failed_runs+=("$run")
    fi
done

echo
echo "========================= Summary =========================="
echo "Successful runs (${#successful_runs[@]}): ${successful_runs[*]:-none}"
echo "Failed runs     (${#failed_runs[@]}): ${failed_runs[*]:-none}"
echo "Analysis timestamp: $ANALYSIS_TIMESTAMP"
echo "Outputs: $ANALYSIS_DIR"

if ((${#failed_runs[@]} > 0)); then
    exit 1
fi
exit 0
