#!/bin/bash

./run_coincidence  ../input_run039510.txt      \
  --run 039510     --config waveform_intervals.ini  \
  --channels-coincident-left 2030 2031 2040 2041  \
  --channels-coincident-right 2050 2051 2060 2061   \
  --channels-to-save 2070 2071 2080 2081  \
  --window-ticks 10

./run_coincidence  ../input_run039511.txt     \
  --run 039511     --config waveform_intervals.ini  \
  --channels-coincident-left 2030 2031 2040 2041  \
  --channels-coincident-right 2050 2051 2060 2061   \
  --channels-to-save 2070 2071 2080 2081  \
  --window-ticks 10

./run_coincidence  ../input_run039512.txt     \
  --run 039512     --config waveform_intervals.ini  \
  --channels-coincident-left 2030 2031 2040 2041  \
  --channels-coincident-right 2050 2051 2060 2061   \
  --channels-to-save 2070 2071 2080 2081  \
  --window-ticks 10
