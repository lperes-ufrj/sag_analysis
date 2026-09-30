#!/bin/bash

RUNS=(
"43374 cosmic"
)

for entry in "${RUNS[@]}"; do
read -r RUN FIELD <<< "${entry}"

echo "=============================================="
echo "Submitting run ${RUN}"
echo "Type: ${FIELD}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune_43374 where name ~ '^np02vd_raw_' ordered limit 10" \
  --jobscript pdvd_decoder_keepup.jobscript \
  --description "ProtoDUNE-VD keepup test: run ${RUN}, ${FIELD}" \
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --env OUTPUT_TAG="${FIELD}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
  --output-pattern "*_${FIELD}_keepup.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_keepup_logs.tgz:${FNALURL}${USERF}"

done
