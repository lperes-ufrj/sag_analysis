#!/bin/bash

RUNS=(
"39468|C7_M3_M4"
"39470|M5_1"
"39471|M5_2"
"39472|M6"
"39473|M7"
"39474|M8"
)

for entry in "${RUNS[@]}"; do
IFS='|' read -r RUN OUTPUT_TAG <<< "${entry}"


echo "=============================================="
echo "Submitting LED calibration run ${RUN}"
echo "X-ARAPUCA: ${OUTPUT_TAG}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune_${RUN} ordered limit 50" \
  --jobscript pdvd_decoder_gallery.jobscript \
  --description "ProtoDUNE-VD LED calibration: run ${RUN}; X-ARAPUCA ${OUTPUT_TAG}" \
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --env OUTPUT_TAG="${OUTPUT_TAG}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
  --output-pattern "*_${OUTPUT_TAG}_gallery.root:${FNALURL}${USERF}" \
  --output-pattern "*_${OUTPUT_TAG}_decoder_gallery_logs.tgz:${FNALURL}${USERF}"


done
