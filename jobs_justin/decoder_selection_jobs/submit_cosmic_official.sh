#!/bin/bash

RUNS=(
"39350 cosmic"
)

for entry in "${RUNS[@]}"; do
read -r RUN FIELD <<< "${entry}"

echo "=============================================="
echo "Submitting electric-field ${RUN}"
echo "High voltage: ${FIELD}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune__raw__cosmics__39350_v1_official ordered limit 50" \
  --jobscript pdvd_decoder_gallery.jobscript \
  --description "ProtoDUNE-VD decoder plus Gallery waveform extraction: run ${RUN}, ${FIELD}" \
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --env OUTPUT_TAG="${FIELD}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
  --output-pattern "*_${FIELD}_gallery.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_gallery_logs.tgz:${FNALURL}${USERF}"

done
