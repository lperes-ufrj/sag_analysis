#ifndef COINCIDENCE_LIB_HPP
#define COINCIDENCE_LIB_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class TChain;

inline constexpr std::int64_t SAMPLE_PERIOD_NS = 16;

struct PeakCandidate {
    int tick = -1;
    double amplitude = 0.0;
    double prominence = 0.0;
    bool significant_additional = false;
};

struct PeakSearchResult {
    bool found = false;
    int tick = -1;
    double amplitude = 0.0;
    std::size_t peak_count = 0;
    std::size_t signal_peak_count = 0;
    bool primary_outside_signal_region = false;
    bool additional_outside_signal_region = false;
    double baseline = 0.0;
    double noise_rms = 0.0;
    double minimum_height = 0.0;
    double additional_prominence_threshold = 0.0;
    std::vector<double> smoothed;
    std::vector<PeakCandidate> candidates;
};

class WaveformAnalyzer {
public:
    explicit WaveformAnalyzer(const std::string &ini_file);

    std::pair<int, int> interval(unsigned int channel, const std::string &name) const;
    double baseline(const std::vector<short> &waveform, int c = 8, int t = 12) const;
    double noiseBaseline(const std::vector<short> &waveform, unsigned int channel) const;
    double amplitude(
        const std::vector<short> &waveform,
        unsigned int channel,
        const std::string &name = "signal"
    ) const;
    double signalAmplitude(const std::vector<short> &waveform, unsigned int channel) const;
    double baselineAdjustedSignalAmplitude(
        const std::vector<short> &waveform,
        unsigned int channel
    ) const;
    double preSignalAmplitude(const std::vector<short> &waveform, unsigned int channel) const;
    double postSignalAmplitude(const std::vector<short> &waveform, unsigned int channel) const;
    PeakSearchResult findPeak(
        const std::vector<short> &waveform,
        unsigned int channel
    ) const;
    int signalPeakTick(const std::vector<short> &waveform, unsigned int channel) const;
    double noiseRms(const std::vector<short> &waveform, unsigned int channel) const;
    int pulseStart(const std::vector<short> &waveform, unsigned int channel) const;
    std::int64_t dtNsPulses(
        const std::vector<short> &waveform_a,
        unsigned int channel_a,
        std::uint64_t timestamp_a,
        const std::vector<short> &waveform_b,
        unsigned int channel_b,
        std::uint64_t timestamp_b
    ) const;

private:
    using Intervals = std::unordered_map<std::string, std::pair<int, int>>;
    std::unordered_map<unsigned int, Intervals> intervals_;

    std::pair<int, int> checkedRegion(
        const std::vector<short> &waveform,
        unsigned int channel,
        const std::string &name
    ) const;
};

std::unique_ptr<TChain> createChain(const std::vector<std::string> &files);

#endif
