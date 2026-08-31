#!/bin/bash

RUNS=(
#"43380 240kV"
"43381 240kV"
"43383 205kV"
"43384 205kV"
"43386 270kV"
"43387 270kV"
"43389 154kV"
"43390 154kV"
)

for entry in "${RUNS[@]}"; do
read -r RUN FIELD <<< "${entry}"

echo "=============================================="
echo "Submitting electric-field ramp run ${RUN}"
echo "High voltage: ${FIELD}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune_${RUN} ordered limit 50" \
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



  