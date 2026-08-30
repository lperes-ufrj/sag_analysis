#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (($# > 1)); then
    echo "Usage: $0 [ANALYSIS_TIMESTAMP]" >&2
    exit 2
fi

#analysis_timestamp=${1:-$(date +%Y%m%d_%H%M%S)}
analysis_timestamp=20260829_094702
echo "Starting full coincidence pipeline"
echo "Analysis timestamp: $analysis_timestamp"

#"$SCRIPT_DIR/run_coincidences.sh" "$analysis_timestamp"
"$SCRIPT_DIR/run_plot_wfs_all.sh" "$analysis_timestamp"

echo
echo "Full pipeline complete"
echo "Coincidence outputs: $SCRIPT_DIR/saved_coincidences/$analysis_timestamp"
echo "Plot outputs: $SCRIPT_DIR/selected_waveforms/$analysis_timestamp"
