#!/bin/bash

RUNS=(
"39510 0kV"
"39511 10kV"
"39512 20kV"
"39514 30kV"
"39515 40kV"
"39516 50kV"
"39517 60kV"
"39518 70kV"
"39519 80kV"
"39521 90kV"
"39522 100kV"
"39523 110kV"
"39525 120kV"
"39526 130kV"
"39527 140kV"
"39528 150kV"
"39529 155p7kV"
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

