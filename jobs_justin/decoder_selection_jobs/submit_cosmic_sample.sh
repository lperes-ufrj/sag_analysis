#!/bin/bash

RUNS=(
<<<<<<< HEAD
"43374 cosmic"
=======
<<<<<<<< HEAD:jobs_justin/decoder_selection_jobs/submit_all_xe_runs.sh
"43440 0p01ppm"
"43552 1ppm"
"43717 2ppm"
"43790 3ppm"
"43903 5ppm"
"44010 7ppm"
"44108 10ppm"
========
"43374 cosmic"
>>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs):jobs_justin/decoder_selection_jobs/submit_cosmic_sample.sh
>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs)
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



  
>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs)
