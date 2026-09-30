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
<<<<<<< HEAD
echo "Submitting run ${RUN}"
echo "Type: ${FIELD}"
echo "=============================================="

justin simple-workflow \
  --mql "files from vd-protodune:vd-protodune_43374 where name ~ '^np02vd_raw_' ordered limit 10" \
  --jobscript pdvd_decoder_keepup.jobscript \
  --description "ProtoDUNE-VD keepup test: run ${RUN}, ${FIELD}" \
=======
echo "Submitting Xe run ${RUN}"
echo "Concentration: ${FIELD}"
echo "=============================================="

justin simple-workflow \
<<<<<<<< HEAD:jobs_justin/decoder_selection_jobs/submit_all_xe_runs.sh
  --mql "files from vd-protodune:vd-protodune_${RUN} ordered limit 10" \
========
  --mql "files from vd-protodune:vd-protodune_43374 ordered limit 100" \
>>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs):jobs_justin/decoder_selection_jobs/submit_cosmic_sample.sh
  --jobscript pdvd_decoder_gallery.jobscript \
  --description "ProtoDUNE-VD decoder plus Gallery waveform extraction: Xe runs, run: ${RUN}, concentration: ${FIELD}" \
>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs)
  --env INPUT_TAR_DIR_LOCAL="$INPUT_TAR_DIR_LOCAL" \
  --env OUTPUT_TAG="${FIELD}" \
  --rss-mib 8000 \
  --wall 14400 \
  --scope usertests \
  --lifetime-days 7 \
<<<<<<< HEAD
  --output-pattern "*_${FIELD}_keepup.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_keepup_logs.tgz:${FNALURL}${USERF}"

done
=======
  --output-pattern "*_${FIELD}_gallery.root:${FNALURL}${USERF}" \
  --output-pattern "*_${FIELD}_decoder_gallery_logs.tgz:${FNALURL}${USERF}"

done



  
>>>>>>> 4787005 ([ADD] sh files for analyze Xe runs)
