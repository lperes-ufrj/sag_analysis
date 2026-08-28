analysis_timestamp=$(date +%Y%m%d_%H%M%S)

.run_coincidences.sh "$analysis_timestamp"

.run_plot_wfs_all.sh "$analysis_timestamp"