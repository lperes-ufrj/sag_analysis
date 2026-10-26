#ifndef WAVEFORM_PLOTTING_HPP
#define WAVEFORM_PLOTTING_HPP

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace waveform_plotting {

struct WaveformRecord {
    double baseline = 0.0;
    std::vector<short> adc;
};

using RecordsByChannel = std::map<unsigned int, std::vector<WaveformRecord>>;

struct Statistics {
    std::vector<std::int64_t> samples;
    std::vector<double> mean;
    std::vector<double> mean_uncertainty;
    std::vector<double> median;
    std::vector<double> percentile_16;
    std::vector<double> percentile_84;
};

Statistics calculateStatistics(const std::vector<WaveformRecord> &records);

void plotAllWaveformsForSelection(
    const std::set<unsigned int> &channels,
    const RecordsByChannel &records_by_channel,
    const std::string &run,
    const std::string &label,
    const std::string &selection_title,
    const std::string &selection_suffix,
    const std::filesystem::path &output_dir
);

void plotAndSaveStatistics(
    const std::set<unsigned int> &channels,
    const RecordsByChannel &records_by_channel,
    const std::string &run,
    const std::string &label,
    const std::filesystem::path &output_dir,
    const std::string &selection_title = "Final Selection",
    const std::string &selection_suffix = "final_selection"
);

}  // namespace waveform_plotting

#endif
