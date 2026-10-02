#!/bin/bash

RUNS=(
"43440 0p01ppm"
"43552 1ppm"
"43717 2ppm"
"43790 3ppm"
"43903 5ppm"
"44010 7ppm"
"44108 10ppm"
)

for entry in "${RUNS[@]}"; do
read -r RUN FIELD <<< "${entry}"

echo "=============================================="
echo "Submitting Xe run ${RUN}"
echo "Concentration: ${FIELD}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune_${RUN} ordered limit 10" \
  --jobscript pdvd_decoder_gallery.jobscript \
  --description "ProtoDUNE-VD decoder plus Gallery waveform extraction: Xe runs, run: ${RUN}, concentration: ${FIELD}" \
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --env OUTPUT_TAG="${FIELD}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
  --output-pattern "*_${FIELD}_gallery.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_gallery_logs.tgz:${FNALURL}${USERF}"

done



  