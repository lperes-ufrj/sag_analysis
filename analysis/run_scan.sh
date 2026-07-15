#!/usr/bin/env bash

time_windows=(20 30)
coincidence_thresholds=(10000 8000 6000 5000 4000)
save_thresholds=(4000 3000 2000)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/coincidence_cuts"
source_file="${script_dir}/coincidence_cuts.cpp"

main() {
  c++ -O2 -Wall -Wextra -std=c++17 $(root-config --cflags) "${source_file}" -o "${binary}" $(root-config --libs)

  for window_ticks in "${time_windows[@]}"; do
    for coincidence_threshold in "${coincidence_thresholds[@]}"; do
      for save_threshold in "${save_thresholds[@]}"; do
        echo "Running window=${window_ticks} ticks, coincidence_adc=${coincidence_threshold}, save_adc=${save_threshold}"

        "${binary}" \
          --coincidence-window-ticks "${window_ticks}" \
          --coincidence-threshold-adc "${coincidence_threshold}" \
          --save-threshold-adc "${save_threshold}"
      done
    done
  done
}

main "$@"
