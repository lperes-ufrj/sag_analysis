#!/bin/bash

RUNS=(
"39500 155p7kV"
"39501 140kV"
"39502 120kV"
"39503 100kV"
"39504 80kV"
"39506 60kV"
"39507 40kV"
"39508 20kV"
"39510 0kV"
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
  --description "ProtoDUNE-VD decoder plus Gallery waveform extraction: ramp down, run ${RUN}, ${FIELD}" \
  --env OUTPUT_TAG="${FIELD}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
  --output-pattern "*_${FIELD}_gallery.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_gallery_logs.tgz:${FNALURL}${USERF}"

done



  