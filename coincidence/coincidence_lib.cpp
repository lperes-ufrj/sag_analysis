#include "coincidence_lib.hpp"

#include <TChain.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string &text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

}  // namespace

WaveformAnalyzer::WaveformAnalyzer(const std::string &ini_file)
{
    std::ifstream input(ini_file);
    if (!input) throw std::runtime_error("Could not open config file: " + ini_file);

    unsigned int current_channel = 0;
    bool in_channel_section = false;
    std::string line;

    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';') continue;

        if (line.front() == '[' && line.back() == ']') {
            const std::string section = line.substr(1, line.size() - 2);
            constexpr const char prefix[] = "channel_";
            in_channel_section = section.rfind(prefix, 0) == 0;
            if (in_channel_section) {
                current_channel = static_cast<unsigned int>(std::stoul(section.substr(8)));
            }
            continue;
        }

        if (!in_channel_section) continue;
        const auto equals = line.find('=');
        if (equals == std::string::npos) continue;

        const std::string name = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        const auto colon = value.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("Invalid interval in config: " + line);
        }

        const int start = std::stoi(trim(value.substr(0, colon)));
        const int stop = std::stoi(trim(value.substr(colon + 1)));
        intervals_[current_channel][name] = {start, stop};
    }
}

std::pair<int, int> WaveformAnalyzer::interval(
    unsigned int channel,
    const std::string &name
) const
{
    const auto channel_it = intervals_.find(channel);
    if (channel_it == intervals_.end()) {
        throw std::runtime_error("Missing config section for channel " + std::to_string(channel));
    }
    const auto interval_it = channel_it->second.find(name);
    if (interval_it == channel_it->second.end()) {
        throw std::runtime_error(
            "Missing interval '" + name + "' for channel " + std::to_string(channel)
        );
    }
    return interval_it->second;
}

std::pair<int, int> WaveformAnalyzer::checkedRegion(
    const std::vector<short> &waveform,
    unsigned int channel,
    const std::string &name
) const
{
    auto [start, stop] = interval(channel, name);
    start = std::max(0, std::min(start, static_cast<int>(waveform.size())));
    stop = std::max(start, std::min(stop, static_cast<int>(waveform.size())));
    if (start == stop) {
        throw std::runtime_error(
            "Empty interval '" + name + "' for channel " + std::to_string(channel)
        );
    }
    return {start, stop};
}

double WaveformAnalyzer::baseline(const std::vector<short> &waveform, int c, int t) const
{
    if (waveform.empty()) {
        throw std::runtime_error("Cannot calculate baseline from an empty waveform");
    }

    const auto [minimum_it, maximum_it] = std::minmax_element(waveform.begin(), waveform.end());
    const int minimum = *minimum_it;
    const int maximum = *maximum_it;
    std::vector<int> counts(static_cast<std::size_t>(maximum - minimum + 1), 0);
    for (const short value : waveform) ++counts[static_cast<std::size_t>(value - minimum)];

    int mode = minimum;
    int mode_count = -1;
    for (int value = minimum; value <= maximum; ++value) {
        const int count = counts[static_cast<std::size_t>(value - minimum)];
        if (count > mode_count) {
            mode = value;
            mode_count = count;
        }
    }

    const int lower = mode - c;
    const int upper = mode + c;
    const std::size_t stride = static_cast<std::size_t>(std::max(1, t));
    double sum = 0.0;
    std::size_t accepted = 0;

    for (std::size_t tick = 0; tick < waveform.size();) {
        const int value = waveform[tick];
        if (value >= lower && value <= upper) {
            sum += value;
            ++accepted;
            tick += stride;
        } else {
            ++tick;
        }
    }
    return accepted == 0 ? static_cast<double>(mode) : sum / accepted;
}

double WaveformAnalyzer::amplitude(
    const std::vector<short> &waveform,
    unsigned int channel,
    const std::string &name
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, name);
    const auto begin = waveform.begin() + start;
    const auto end = waveform.begin() + stop;
    const auto [minimum, maximum] = std::minmax_element(begin, end);
    return static_cast<double>(*maximum) - static_cast<double>(*minimum);
}

double WaveformAnalyzer::noiseBaseline(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, "noise");
    std::vector<short> samples(waveform.begin() + start, waveform.begin() + stop);
    const std::size_t middle = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
    const double upper = samples[middle];

    if (samples.size() % 2 != 0) return upper;

    const double lower = *std::max_element(samples.begin(), samples.begin() + middle);
    return (lower + upper) / 2.0;
}

double WaveformAnalyzer::signalAmplitude(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    return amplitude(waveform, channel, "signal");
}

double WaveformAnalyzer::baselineAdjustedSignalAmplitude(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, "signal");
    const auto peak = std::max_element(waveform.begin() + start, waveform.begin() + stop);
    return static_cast<double>(*peak) - noiseBaseline(waveform, channel);
}

double WaveformAnalyzer::preSignalAmplitude(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    return amplitude(waveform, channel, "pre_signal");
}

double WaveformAnalyzer::postSignalAmplitude(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    return amplitude(waveform, channel, "post_signal");
}

int WaveformAnalyzer::signalPeakTick(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, "signal");
    return start + static_cast<int>(
        std::distance(
            waveform.begin() + start,
            std::max_element(waveform.begin() + start, waveform.begin() + stop)
        )
    );
}

double WaveformAnalyzer::noiseRms(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, "noise");
    const double count = stop - start;
    const double sum = std::accumulate(
        waveform.begin() + start,
        waveform.begin() + stop,
        0.0
    );
    const double mean = sum / count;
    double squared_difference = 0.0;
    for (int index = start; index < stop; ++index) {
        const double difference = waveform[static_cast<std::size_t>(index)] - mean;
        squared_difference += difference * difference;
    }
    return std::sqrt(squared_difference / count);
}

int WaveformAnalyzer::pulseStart(
    const std::vector<short> &waveform,
    unsigned int channel
) const
{
    const auto [start, stop] = checkedRegion(waveform, channel, "signal");
    const double threshold = noiseBaseline(waveform, channel)
        + 10.0 * noiseRms(waveform, channel);
    const auto peak_it = std::max_element(waveform.begin() + start, waveform.begin() + stop);
    int tick = static_cast<int>(std::distance(waveform.begin() + start, peak_it));
    while (tick > 0 && waveform[static_cast<std::size_t>(start + tick - 1)] >= threshold) {
        --tick;
    }
    return start + tick;
}

std::int64_t WaveformAnalyzer::dtNsPulses(
    const std::vector<short> &waveform_a,
    unsigned int channel_a,
    std::uint64_t timestamp_a,
    const std::vector<short> &waveform_b,
    unsigned int channel_b,
    std::uint64_t timestamp_b
) const
{
    const auto time_a = static_cast<std::int64_t>(timestamp_a)
        + pulseStart(waveform_a, channel_a) * SAMPLE_PERIOD_NS;
    const auto time_b = static_cast<std::int64_t>(timestamp_b)
        + pulseStart(waveform_b, channel_b) * SAMPLE_PERIOD_NS;
    return time_a - time_b;
}

std::unique_ptr<TChain> createChain(const std::vector<std::string> &files)
{
    auto chain = std::make_unique<TChain>("WaveformTree");
    for (const auto &file : files) {
        if (chain->Add(file.c_str()) == 0) {
            throw std::runtime_error("Could not add ROOT file to chain: " + file);
        }
    }
    return chain;
}
