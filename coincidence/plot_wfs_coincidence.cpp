#include "coincidence_lib.hpp"

#include <TCanvas.h>
#include <TChain.h>
#include <TGraph.h>
#include <TGraphAsymmErrors.h>
#include <TH2I.h>
#include <TLegend.h>
#include <TLine.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double DEFAULT_MAX_AUXILIARY_AMPLITUDE = 500.0;
constexpr double MAX_FULL_RANGE_AMPLITUDE = 9000.0;
constexpr int MAX_WAVEFORM_DENSITY_Y_BINS = 2000;
const std::array<std::string, 3> DEFAULT_RUNS{"039510", "039511", "039512"};
const std::string DEFAULT_CSV_SUFFIX =
    "coinc_2030-2031-2040-2041_vs_2050-2051-2060-2061_"
    "save_2070-2071-2080-2081_window_10_ticks_min_amplitude_0_adc.csv";

struct Options {
    std::vector<fs::path> input_lists;
    fs::path config;
    fs::path output_dir;
    fs::path csv_file;
    std::string csv_suffix = DEFAULT_CSV_SUFFIX;
    double max_auxiliary_amplitude = DEFAULT_MAX_AUXILIARY_AMPLITUDE;
};

struct WaveformKey {
    int event = 0;
    unsigned int channel = 0;
    int waveform_index = 0;

    bool operator==(const WaveformKey &other) const
    {
        return event == other.event && channel == other.channel
            && waveform_index == other.waveform_index;
    }
};

struct WaveformKeyHash {
    std::size_t operator()(const WaveformKey &key) const
    {
        std::size_t value = std::hash<int>{}(key.event);
        value ^= std::hash<unsigned int>{}(key.channel)
            + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        value ^= std::hash<int>{}(key.waveform_index)
            + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        return value;
    }
};

struct LocatedWaveform {
    Long64_t entry = 0;
    WaveformKey key;
    std::size_t multiplicity = 1;
};

struct WaveformRecord {
    double baseline = 0.0;
    std::vector<short> adc;
};

using RecordsByChannel = std::map<unsigned int, std::vector<WaveformRecord>>;

struct CutStatistics {
    std::size_t from_coincidence = 0;
    std::size_t rejected_noise = 0;
    std::size_t rejected_post_signal = 0;
    std::size_t rejected_full_range_amplitude = 0;
    std::size_t rejected_primary_peak_outside_signal = 0;
    std::size_t rejected_additional_peak_outside_signal = 0;
    std::size_t rejected_signal_peak_count = 0;
    std::size_t rejected_missing_primary_peak = 0;
    std::size_t rejected_combined = 0;
    std::size_t final_selection = 0;
};

struct WaveformSelections {
    RecordsByChannel from_coincidence;
    RecordsByChannel final_selection;
    CutStatistics cuts;
};

struct Statistics {
    std::vector<std::int64_t> samples;
    std::vector<double> mean;
    std::vector<double> mean_uncertainty;
    std::vector<double> median;
    std::vector<double> percentile_16;
    std::vector<double> percentile_84;
};

std::string trim(const std::string &text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> splitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) fields.push_back(trim(field));
    return fields;
}

std::string normalizeRun(std::string run)
{
    run = trim(run);
    if (run.rfind("run", 0) == 0 || run.rfind("RUN", 0) == 0) run.erase(0, 3);
    if (run.empty() || !std::all_of(run.begin(), run.end(), ::isdigit)) {
        throw std::runtime_error("Invalid run number: " + run);
    }
    if (run.size() < 6) run.insert(run.begin(), 6 - run.size(), '0');
    return run;
}

void printUsage(const char *program)
{
    std::cout
        << "Usage: " << program << " [options] [INPUT_LIST.txt ...]\n\n"
        << "Process selected coincidence waveforms and create the same data products\n"
        << "as plot_wfs_coincidence.ipynb. The run number is inferred from each input\n"
        << "list filename or its ROOT filenames. With no input lists, processes the\n"
        << "default lists for runs 039510, 039511, and 039512.\n\n"
        << "Options:\n"
        << "  --config FILE                 Waveform interval INI file\n"
        << "  --output-dir DIR             Output directory\n"
        << "  --csv FILE                   Explicit timestamped selection CSV\n"
        << "  --csv-suffix SUFFIX          Selection CSV filename suffix\n"
        << "  --max-auxiliary-amplitude N  Absolute |ADC - baseline| limit in noise/"
        << "post-signal regions (default: 500)\n"
        << "  -h, --help                    Show this help\n";
}

Options parseOptions(int argc, char **argv, const fs::path &program_dir)
{
    Options options;
    options.config = program_dir / "waveform_intervals.ini";
    options.output_dir = program_dir / "selected_waveforms";

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (argument == "--config" || argument == "--output-dir"
            || argument == "--csv"
            || argument == "--csv-suffix"
            || argument == "--max-auxiliary-amplitude") {
            if (++index >= argc) throw std::runtime_error("Missing value for " + argument);
            if (argument == "--config") options.config = argv[index];
            else if (argument == "--output-dir") options.output_dir = argv[index];
            else if (argument == "--csv") options.csv_file = argv[index];
            else if (argument == "--csv-suffix") options.csv_suffix = argv[index];
            else options.max_auxiliary_amplitude = std::stod(argv[index]);
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        }
        options.input_lists.emplace_back(argument);
    }

    if (options.input_lists.empty()) {
        for (const auto &run : DEFAULT_RUNS) {
            options.input_lists.push_back(program_dir / ("input_run" + run + ".txt"));
        }
    } else {
        for (auto &input_list : options.input_lists) {
            const fs::path command_line_path = fs::absolute(input_list);
            const fs::path program_relative_path = program_dir / input_list;
            input_list = fs::exists(command_line_path)
                ? command_line_path
                : program_relative_path;
        }
    }
    options.config = fs::absolute(options.config);
    options.output_dir = fs::absolute(options.output_dir);
    if (!options.csv_file.empty()) options.csv_file = fs::absolute(options.csv_file);
    if (!options.csv_file.empty() && options.input_lists.size() != 1) {
        throw std::runtime_error("--csv requires exactly one input list");
    }
    if (options.csv_suffix.size() <= 4
        || options.csv_suffix.substr(options.csv_suffix.size() - 4) != ".csv") {
        throw std::runtime_error("CSV suffix must end in .csv");
    }
    if (!std::isfinite(options.max_auxiliary_amplitude)
        || options.max_auxiliary_amplitude < 0.0) {
        throw std::runtime_error(
            "Maximum auxiliary amplitude must be a finite, non-negative value"
        );
    }
    return options;
}

std::string inferRun(const fs::path &input_list)
{
    const std::regex run_pattern("run[_-]?([0-9]+)", std::regex::icase);
    std::smatch match;
    const std::string list_name = input_list.filename().string();
    if (std::regex_search(list_name, match, run_pattern)) {
        return normalizeRun(match[1].str());
    }

    std::ifstream input(input_list);
    if (!input) throw std::runtime_error("Could not open input list: " + input_list.string());
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        if (std::regex_search(line, match, run_pattern)) {
            return normalizeRun(match[1].str());
        }
    }
    throw std::runtime_error(
        "Could not infer a run number from input list: " + input_list.string()
    );
}

std::vector<fs::path> readRootFiles(const fs::path &input_list, const fs::path &repo_dir)
{
    std::ifstream input(input_list);
    if (!input) throw std::runtime_error("Could not open input list: " + input_list.string());

    std::vector<fs::path> files;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;

        fs::path path = line;
        if (path.is_relative()) {
            const fs::path list_relative = input_list.parent_path() / path;
            path = fs::exists(list_relative) ? list_relative : repo_dir / path;
        }
        path = fs::absolute(path);
        if (!fs::is_regular_file(path)) {
            throw std::runtime_error("ROOT file not found: " + path.string());
        }
        files.push_back(path);
    }
    if (files.empty()) throw std::runtime_error("No ROOT files found in " + input_list.string());
    return files;
}

std::unordered_map<WaveformKey, std::size_t, WaveformKeyHash> readSelectedKeys(
    const fs::path &csv_file,
    std::set<unsigned int> &channels,
    std::size_t &row_count
)
{
    std::ifstream input(csv_file);
    if (!input) throw std::runtime_error("Could not open selection CSV: " + csv_file.string());

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty CSV: " + csv_file.string());
    const auto header = splitCsv(line);

    auto column = [&](const std::string &name) {
        const auto found = std::find(header.begin(), header.end(), name);
        if (found == header.end()) throw std::runtime_error("Missing CSV column: " + name);
        return static_cast<std::size_t>(std::distance(header.begin(), found));
    };
    const auto event_column = column("event");
    const auto channel_column = column("channel");
    const auto waveform_column = column("waveform_index_save");
    const auto required_size = std::max({event_column, channel_column, waveform_column}) + 1;

    std::unordered_map<WaveformKey, std::size_t, WaveformKeyHash> keys;
    row_count = 0;
    while (std::getline(input, line)) {
        if (trim(line).empty()) continue;
        const auto fields = splitCsv(line);
        if (fields.size() < required_size) {
            throw std::runtime_error("Malformed CSV row " + std::to_string(row_count + 2));
        }
        WaveformKey key{
            std::stoi(fields[event_column]),
            static_cast<unsigned int>(std::stoul(fields[channel_column])),
            std::stoi(fields[waveform_column]),
        };
        ++keys[key];
        channels.insert(key.channel);
        ++row_count;
    }
    return keys;
}

std::vector<LocatedWaveform> locateSelectedEntries(
    TChain &chain,
    const std::unordered_map<WaveformKey, std::size_t, WaveformKeyHash> &selected
)
{
    int event = 0;
    int waveform_index = 0;
    unsigned int channel = 0;
    chain.SetBranchStatus("*", 0);
    chain.SetBranchStatus("event", 1);
    chain.SetBranchStatus("channel", 1);
    chain.SetBranchStatus("waveform_index", 1);
    chain.SetBranchAddress("event", &event);
    chain.SetBranchAddress("channel", &channel);
    chain.SetBranchAddress("waveform_index", &waveform_index);

    std::vector<LocatedWaveform> located;
    located.reserve(selected.size());
    const Long64_t entries = chain.GetEntries();
    for (Long64_t entry = 0; entry < entries; ++entry) {
        if (chain.GetEntry(entry) <= 0) {
            throw std::runtime_error("Could not read ROOT metadata entry " + std::to_string(entry));
        }
        const WaveformKey key{event, channel, waveform_index};
        const auto found = selected.find(key);
        if (found != selected.end()) located.push_back({entry, key, found->second});
    }

    if (located.size() != selected.size()) {
        throw std::runtime_error(
            std::to_string(selected.size() - located.size())
            + " selected waveform keys were not found"
        );
    }
    return located;
}

double maximumAbsoluteAmplitudeInRegion(
    const std::vector<short> &adc,
    double baseline,
    const WaveformAnalyzer &analyzer,
    unsigned int channel,
    const std::string &name
)
{
    auto [start, stop] = analyzer.interval(channel, name);
    start = std::max(0, std::min(start, static_cast<int>(adc.size())));
    stop = std::max(start, std::min(stop, static_cast<int>(adc.size())));
    if (start == stop) throw std::runtime_error("Empty " + name + " interval");
    const auto [minimum, maximum] = std::minmax_element(
        adc.begin() + start, adc.begin() + stop
    );
    // Treat positive and negative excursions from the waveform baseline equally.
    return std::max(
        std::abs(static_cast<double>(*minimum) - baseline),
        std::abs(static_cast<double>(*maximum) - baseline)
    );
}

double fullRangeAmplitude(const std::vector<short> &adc, double baseline)
{
    if (adc.empty()) {
        throw std::runtime_error("Cannot calculate amplitude from an empty waveform");
    }
    return static_cast<double>(*std::max_element(adc.begin(), adc.end())) - baseline;
}

WaveformSelections loadWaveforms(
    TChain &chain,
    const std::vector<LocatedWaveform> &located,
    const WaveformAnalyzer &analyzer,
    double max_auxiliary_amplitude
)
{
    chain.ResetBranchAddresses();
    chain.SetBranchStatus("*", 0);
    chain.SetBranchStatus("adc", 1);
    std::vector<short> *adc = nullptr;
    chain.SetBranchAddress("adc", &adc);

    WaveformSelections selections;
    std::size_t rejected_noise = 0;
    std::size_t rejected_post_signal = 0;
    std::size_t rejected_full_range_amplitude = 0;
    std::size_t rejected_primary_peak_outside_signal = 0;
    std::size_t rejected_additional_peak_outside_signal = 0;
    std::size_t rejected_signal_peak_count = 0;
    std::size_t rejected_missing_primary_peak = 0;
    std::size_t rejected_total = 0;
    std::size_t total = 0;

    for (const auto &item : located) {
        if (chain.GetEntry(item.entry) <= 0 || adc == nullptr) {
            throw std::runtime_error("Could not read ROOT ADC entry " + std::to_string(item.entry));
        }
        const double baseline = analyzer.noiseBaseline(*adc, item.key.channel);
        for (std::size_t copy = 0; copy < item.multiplicity; ++copy) {
            selections.from_coincidence[item.key.channel].push_back({baseline, *adc});
        }
        const double noise_absolute_amplitude = maximumAbsoluteAmplitudeInRegion(
            *adc, baseline, analyzer, item.key.channel, "noise"
        );
        const double post_signal_absolute_amplitude = maximumAbsoluteAmplitudeInRegion(
            *adc, baseline, analyzer, item.key.channel, "post_signal"
        );
        const bool fails_noise = noise_absolute_amplitude > max_auxiliary_amplitude;
        const bool fails_post_signal =
            post_signal_absolute_amplitude > max_auxiliary_amplitude;
        const double amplitude = fullRangeAmplitude(*adc, baseline);
        const bool fails_full_range_amplitude = amplitude > MAX_FULL_RANGE_AMPLITUDE;
        const PeakSearchResult peak = analyzer.findPeak(*adc, item.key.channel);
        const bool fails_primary_peak = peak.primary_outside_signal_region;
        const bool fails_additional_peak = peak.additional_outside_signal_region;
        const bool fails_signal_peak_count = peak.signal_peak_count != 1;
        const bool fails_missing_primary_peak = !peak.found;

        rejected_noise += item.multiplicity * static_cast<std::size_t>(fails_noise);
        rejected_post_signal += item.multiplicity * static_cast<std::size_t>(fails_post_signal);
        rejected_full_range_amplitude += item.multiplicity
            * static_cast<std::size_t>(fails_full_range_amplitude);
        rejected_primary_peak_outside_signal += item.multiplicity
            * static_cast<std::size_t>(fails_primary_peak);
        rejected_additional_peak_outside_signal += item.multiplicity
            * static_cast<std::size_t>(fails_additional_peak);
        rejected_signal_peak_count += item.multiplicity
            * static_cast<std::size_t>(fails_signal_peak_count);
        rejected_missing_primary_peak += item.multiplicity
            * static_cast<std::size_t>(fails_missing_primary_peak);
        total += item.multiplicity;
        if (fails_noise || fails_post_signal || fails_full_range_amplitude
            || fails_primary_peak || fails_additional_peak || fails_signal_peak_count
            || fails_missing_primary_peak) {
            rejected_total += item.multiplicity;
            continue;
        }
        for (std::size_t copy = 0; copy < item.multiplicity; ++copy) {
            selections.final_selection[item.key.channel].push_back({baseline, *adc});
        }
    }

    std::cout
        << "Retained " << total - rejected_total << " of " << total
        << " selected waveforms\n"
        << "Rejected " << rejected_total
        << " waveforms in total (cut categories may overlap):\n"
        << "  Noise-region |ADC - baseline| above " << max_auxiliary_amplitude
        << " ADC: " << rejected_noise << '\n'
        << "  Post-signal |ADC - baseline| above " << max_auxiliary_amplitude
        << " ADC: " << rejected_post_signal << '\n'
        << "  Full-range amplitude above " << MAX_FULL_RANGE_AMPLITUDE
        << " ADC over baseline: " << rejected_full_range_amplitude << '\n'
        << "  Primary peak outside signal region: "
        << rejected_primary_peak_outside_signal << '\n'
        << "  Primary peak inside with another peak outside signal region: "
        << rejected_additional_peak_outside_signal << '\n'
        << "  Signal region does not contain exactly one peak candidate: "
        << rejected_signal_peak_count << '\n'
        << "  No primary peak found: " << rejected_missing_primary_peak << '\n';
    selections.cuts = {
        total,
        rejected_noise,
        rejected_post_signal,
        rejected_full_range_amplitude,
        rejected_primary_peak_outside_signal,
        rejected_additional_peak_outside_signal,
        rejected_signal_peak_count,
        rejected_missing_primary_peak,
        rejected_total,
        total - rejected_total
    };
    return selections;
}

void writeCutSummary(
    const fs::path &filename,
    const std::string &run,
    const fs::path &input_list,
    const fs::path &selection_csv,
    const fs::path &config,
    const std::set<unsigned int> &channels,
    std::size_t selected_rows,
    std::size_t unique_located_waveforms,
    double max_auxiliary_amplitude,
    const WaveformAnalyzer &analyzer,
    const CutStatistics &cuts,
    const RecordsByChannel &final_selection
)
{
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("Could not create cut summary: " + filename.string());
    }

    output << "Waveform selection cut summary\n"
           << "==============================\n"
           << "Run: " << run << '\n'
           << "Input list: " << input_list.string() << '\n'
           << "Coincidence selection CSV: " << selection_csv.string() << '\n'
           << "Waveform interval configuration: " << config.string() << '\n'
           << "Rows in coincidence CSV: " << selected_rows << '\n'
           << "Unique ROOT waveforms located: " << unique_located_waveforms << '\n'
           << "Waveforms entering selection, including CSV multiplicity: "
           << cuts.from_coincidence << "\n\n"
           << "Applied cuts\n"
           << "------------\n"
           << "1. Noise-region maximum absolute |ADC - baseline| <= "
           << max_auxiliary_amplitude << " ADC\n"
           << "2. Post-signal maximum absolute |ADC - baseline| <= "
           << max_auxiliary_amplitude << " ADC\n"
           << "3. Full-waveform positive amplitude above baseline <= "
           << MAX_FULL_RANGE_AMPLITUDE << " ADC\n"
           << "4. Primary peak must be inside the configured signal region\n"
           << "5. No significant additional peak may be outside the signal region\n"
           << "6. The signal region must contain exactly one peak candidate\n"
           << "7. A primary peak must be found\n\n"
           << "Peak-finder configuration\n"
           << "-------------------------\n"
           << "Smoothing window: " << PEAK_SMOOTHING_WINDOW << " samples\n"
           << "Minimum peak height: max(" << MINIMUM_PEAK_HEIGHT_ADC
           << " ADC, " << MINIMUM_PEAK_HEIGHT_SIGMA << " x noise RMS)\n"
           << "Additional-peak minimum amplitude: "
           << 100.0 * MINIMUM_ADDITIONAL_PEAK_FRACTION
           << "% of the primary-peak amplitude\n"
           << "Additional-peak minimum prominence: max("
           << MINIMUM_ADDITIONAL_PEAK_PROMINENCE_ADC << " ADC, "
           << MINIMUM_PEAK_HEIGHT_SIGMA << " x noise RMS)\n\n"
           << "Configured intervals (half-open [first, last))\n"
           << "------------------------------------------------\n";

    for (const unsigned int channel : channels) {
        const auto noise = analyzer.interval(channel, "noise");
        const auto signal = analyzer.interval(channel, "signal");
        const auto post_signal = analyzer.interval(channel, "post_signal");
        output << "Channel " << channel
               << ": noise=[" << noise.first << ", " << noise.second << ")"
               << ", signal=[" << signal.first << ", " << signal.second << ")"
               << ", post_signal=[" << post_signal.first << ", "
               << post_signal.second << ")\n";
    }

    output << "\nFinal-selection baseline noise\n"
           << "------------------------------\n"
           << "Definition: sample standard deviation of all baseline-subtracted "
           << "ADC values pooled from ticks [0, 50) of the final-selection waveforms.\n";
    for (const unsigned int channel : channels) {
        const auto found = final_selection.find(channel);
        if (found == final_selection.end() || found->second.empty()) {
            output << "Channel " << channel << ": no final-selection waveforms\n";
            continue;
        }

        std::size_t sample_count = 0;
        double mean = 0.0;
        double sum_squared_difference = 0.0;
        for (const auto &record : found->second) {
            const std::size_t stop = std::min<std::size_t>(50, record.adc.size());
            for (std::size_t sample = 0; sample < stop; ++sample) {
                const double value = static_cast<double>(record.adc[sample])
                    - record.baseline;
                ++sample_count;
                const double difference = value - mean;
                mean += difference / static_cast<double>(sample_count);
                const double updated_difference = value - mean;
                sum_squared_difference += difference * updated_difference;
            }
        }

        output << "Channel " << channel << ": ";
        if (sample_count > 1) {
            const double standard_deviation = std::sqrt(
                sum_squared_difference / static_cast<double>(sample_count - 1)
            );
            output << standard_deviation << " ADC";
        } else {
            output << "not enough samples";
        }
        output << " (waveforms=" << found->second.size()
               << ", baseline samples=" << sample_count << ")\n";
    }

    const auto write_counts = [&](const std::string &criterion, std::size_t rejected) {
        output << std::left << std::setw(62) << criterion
               << std::right << std::setw(12) << cuts.from_coincidence - rejected
               << std::setw(12) << rejected << '\n';
    };

    output << "\nCut results\n"
           << "-----------\n"
           << std::left << std::setw(62) << "Criterion"
           << std::right << std::setw(12) << "Accepted"
           << std::setw(12) << "Rejected" << '\n'
           << std::string(86, '-') << '\n';
    write_counts("Noise-region amplitude", cuts.rejected_noise);
    write_counts("Post-signal amplitude", cuts.rejected_post_signal);
    write_counts("Full-range amplitude", cuts.rejected_full_range_amplitude);
    write_counts(
        "Primary peak outside signal region",
        cuts.rejected_primary_peak_outside_signal
    );
    write_counts(
        "Significant additional peak outside signal region",
        cuts.rejected_additional_peak_outside_signal
    );
    write_counts("Signal-region peak count != 1", cuts.rejected_signal_peak_count);
    write_counts("No primary peak found", cuts.rejected_missing_primary_peak);
    write_counts("All cuts combined", cuts.rejected_combined);

    output << "\nFinal selected waveforms: " << cuts.final_selection << '\n'
           << "Note: individual rejection categories can overlap; therefore their "
           << "rejected counts should not be summed.\n";

    if (!output) {
        throw std::runtime_error("Failed while writing cut summary: " + filename.string());
    }
    std::cout << "Saved " << filename << '\n';
}

double percentileFromSorted(const std::vector<double> &values, double quantile)
{
    const double position = quantile * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

Statistics calculateStatistics(const std::vector<WaveformRecord> &records)
{
    if (records.empty()) throw std::runtime_error("Cannot summarize an empty channel");
    const std::size_t sample_count = records.front().adc.size();
    for (const auto &record : records) {
        if (record.adc.size() != sample_count) {
            throw std::runtime_error("Channel contains inconsistent waveform lengths");
        }
    }

    Statistics result;
    result.samples.resize(sample_count);
    result.mean.resize(sample_count);
    result.mean_uncertainty.resize(sample_count);
    result.median.resize(sample_count);
    result.percentile_16.resize(sample_count);
    result.percentile_84.resize(sample_count);
    std::vector<double> values(records.size());

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        result.samples[sample] = static_cast<std::int64_t>(sample);
        double sum = 0.0;
        for (std::size_t row = 0; row < records.size(); ++row) {
            values[row] = static_cast<double>(records[row].adc[sample])
                - records[row].baseline;
            sum += values[row];
        }
        result.mean[sample] = sum / static_cast<double>(records.size());
        double squared_difference = 0.0;
        for (const double value : values) {
            const double difference = value - result.mean[sample];
            squared_difference += difference * difference;
        }
        if (records.size() > 1) {
            const double sample_standard_deviation = std::sqrt(
                squared_difference / static_cast<double>(records.size() - 1)
            );
            result.mean_uncertainty[sample] = sample_standard_deviation
                / std::sqrt(static_cast<double>(records.size()));
        } else {
            result.mean_uncertainty[sample] = 0.0;
        }
        std::sort(values.begin(), values.end());
        result.median[sample] = percentileFromSorted(values, 0.5);
        result.percentile_16[sample] = percentileFromSorted(values, 0.16);
        result.percentile_84[sample] = percentileFromSorted(values, 0.84);
    }
    return result;
}

void writeStatisticsCsv(const fs::path &filename, const Statistics &statistics)
{
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Could not create " + filename.string());

    output << "sample,mean,mean_uncertainty,median,percentile_16,percentile_84\n";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t index = 0; index < statistics.samples.size(); ++index) {
        output << statistics.samples[index] << ','
               << statistics.mean[index] << ','
               << statistics.mean_uncertainty[index] << ','
               << statistics.median[index] << ','
               << statistics.percentile_16[index] << ','
               << statistics.percentile_84[index] << '\n';
    }
}


void plotAllWaveformsWithStatistics(
    unsigned int channel,
    const std::vector<WaveformRecord> &records,
    const Statistics &statistics,
    const std::string &run,
    const std::string &label,
    const std::string &selection_title,
    const std::string &selection_suffix,
    const fs::path &output_dir
)
{
    if (records.empty() || statistics.samples.empty()) return;

    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    for (const auto &record : records) {
        for (const short adc : record.adc) {
            const double value = static_cast<double>(adc) - record.baseline;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
    }
    if (minimum == maximum) {
        minimum -= 0.5;
        maximum += 0.5;
    }

    const double y_range = maximum - minimum;
    const int y_bins = std::max(
        1,
        std::min(MAX_WAVEFORM_DENSITY_Y_BINS, static_cast<int>(std::ceil(y_range)) + 1)
    );
    const int count = static_cast<int>(statistics.samples.size());
    const std::string suffix = std::to_string(channel) + "_" + label
        + "_" + selection_suffix;
    TH2I density(
        ("all_waveforms_" + suffix).c_str(),
        ("Run " + run + " - " + selection_title + " - Channel "
            + std::to_string(channel) + " - "
            + std::to_string(records.size())
            + " waveforms;Sample;ADC - baseline;Waveform samples").c_str(),
        count,
        -0.5,
        static_cast<double>(count) - 0.5,
        y_bins,
        minimum - 0.5,
        maximum + 0.5
    );
    density.SetStats(false);
    for (const auto &record : records) {
        for (std::size_t sample = 0; sample < record.adc.size(); ++sample) {
            density.Fill(
                static_cast<double>(sample),
                static_cast<double>(record.adc[sample]) - record.baseline
            );
        }
    }

    std::vector<double> x(count), mean(count), median(count);
    for (int index = 0; index < count; ++index) {
        const auto sample = static_cast<std::size_t>(index);
        x[index] = static_cast<double>(statistics.samples[sample]);
        mean[index] = statistics.mean[sample];
        median[index] = statistics.median[sample];
    }

    TGraph mean_graph(count, x.data(), mean.data());
    TGraph median_graph(count, x.data(), median.data());

    TCanvas canvas(("all_waveforms_canvas_" + suffix).c_str(), "", 1800, 850);
    canvas.SetLeftMargin(0.09);
    canvas.SetRightMargin(0.13);
    canvas.SetBottomMargin(0.11);
    canvas.SetTopMargin(0.10);
    canvas.SetLogz();
    canvas.SetGrid();
    density.SetMinimum(1.0);
    density.Draw("COLZ");

    median_graph.SetLineColor(kBlue + 1);
    median_graph.SetLineWidth(3);
    median_graph.Draw("L SAME");
    mean_graph.SetLineColor(kBlack);
    mean_graph.SetLineStyle(2);
    mean_graph.SetLineWidth(3);
    mean_graph.Draw("L SAME");

    TLegend legend(0.68, 0.76, 0.86, 0.89);
    legend.SetFillColorAlpha(kWhite, 0.88);
    legend.AddEntry(&median_graph, "Median", "l");
    legend.AddEntry(&mean_graph, "Mean", "l");
    legend.Draw();

    const fs::path output = output_dir / ("all_waveforms_" + suffix + ".png");
    canvas.SaveAs(output.string().c_str());
    std::cout << "Saved " << output << '\n';
}

void plotAllWaveformsForSelection(
    const std::set<unsigned int> &channels,
    const RecordsByChannel &records_by_channel,
    const std::string &run,
    const std::string &label,
    const std::string &selection_title,
    const std::string &selection_suffix,
    const fs::path &output_dir
)
{
    for (const unsigned int channel : channels) {
        const auto found = records_by_channel.find(channel);
        if (found == records_by_channel.end() || found->second.empty()) continue;
        const Statistics statistics = calculateStatistics(found->second);
        plotAllWaveformsWithStatistics(
            channel,
            found->second,
            statistics,
            run,
            label,
            selection_title,
            selection_suffix,
            output_dir
        );
    }
}

void plotAndSaveStatistics(
    const std::set<unsigned int> &channels,
    const RecordsByChannel &records_by_channel,
    const std::string &run,
    const std::string &label,
    const fs::path &output_dir
)
{
    std::vector<unsigned int> plot_channels;
    for (const auto channel : channels) {
        const auto found = records_by_channel.find(channel);
        if (found != records_by_channel.end() && !found->second.empty()) {
            plot_channels.push_back(channel);
        }
    }
    if (plot_channels.empty()) {
        throw std::runtime_error("No waveforms passed the auxiliary-region quality cuts");
    }

    const int columns = 2;
    const int rows = static_cast<int>((plot_channels.size() + columns - 1) / columns);
    TCanvas canvas(
        ("summary_" + label).c_str(),
        ("Run " + run + " - Mean and median selected waveforms").c_str(),
        1900,
        550 * rows
    );
    canvas.Divide(columns, rows);

    std::vector<Statistics> all_statistics;
    std::vector<TGraphAsymmErrors> bands;
    std::vector<TGraph> medians;
    std::vector<TGraph> means;
    std::vector<TLine> zero_lines;
    std::vector<TLegend> legends;
    all_statistics.reserve(plot_channels.size());
    bands.reserve(plot_channels.size());
    medians.reserve(plot_channels.size());
    means.reserve(plot_channels.size());
    zero_lines.reserve(plot_channels.size());
    legends.reserve(plot_channels.size());

    const std::array<int, 4> colors{
        kBlue + 1,
        kOrange + 7,
        kGreen + 2,
        kRed + 1,
    };

    for (std::size_t panel = 0; panel < plot_channels.size(); ++panel) {
        const auto channel = plot_channels[panel];
        const auto &records = records_by_channel.at(channel);
        all_statistics.push_back(calculateStatistics(records));
        const auto &statistics = all_statistics.back();
        const auto count = static_cast<int>(statistics.samples.size());

        std::vector<double> x(count), mean(count), median(count);
        std::vector<double> lower(count), upper(count), zero(count);
        for (int index = 0; index < count; ++index) {
            x[index] = statistics.samples[static_cast<std::size_t>(index)];
            mean[index] = statistics.mean[static_cast<std::size_t>(index)];
            median[index] = statistics.median[static_cast<std::size_t>(index)];
            lower[index] = median[index] - statistics.percentile_16[static_cast<std::size_t>(index)];
            upper[index] = statistics.percentile_84[static_cast<std::size_t>(index)] - median[index];
        }

        canvas.cd(static_cast<int>(panel) + 1);
        gPad->SetLeftMargin(0.13);
        gPad->SetRightMargin(0.04);
        gPad->SetBottomMargin(0.12);
        gPad->SetTopMargin(0.12);
        gPad->SetGrid();
        bands.emplace_back(count, x.data(), median.data(), zero.data(), zero.data(),
                           lower.data(), upper.data());
        auto &band = bands.back();
        band.SetTitle(
            ("Run " + run + " - Channel " + std::to_string(channel) + " - "
                + std::to_string(records.size()) + " waveforms;Sample;ADC - baseline").c_str()
        );
        band.GetXaxis()->SetTitleOffset(1.05);
        band.GetYaxis()->SetTitleOffset(1.35);
        band.SetFillColorAlpha(colors[panel % colors.size()], 0.25);
        band.SetLineColor(colors[panel % colors.size()]);
        band.Draw("A3");

        medians.emplace_back(count, x.data(), median.data());
        medians.back().SetLineColor(colors[panel % colors.size()]);
        medians.back().SetLineWidth(3);
        medians.back().Draw("L SAME");
        means.emplace_back(count, x.data(), mean.data());
        means.back().SetLineColor(kBlack);
        means.back().SetLineStyle(2);
        means.back().SetLineWidth(2);
        means.back().Draw("L SAME");

        zero_lines.emplace_back(0.0, 0.0, static_cast<double>(count - 1), 0.0);
        zero_lines.back().SetLineColor(kGray + 2);
        zero_lines.back().Draw("SAME");
        legends.emplace_back(0.66, 0.72, 0.93, 0.92);
        legends.back().AddEntry(&band, "16-84%", "f");
        legends.back().AddEntry(&medians.back(), "Median", "l");
        legends.back().AddEntry(&means.back(), "Mean", "l");
        legends.back().Draw();

        const fs::path csv_file = output_dir
            / ("channel_" + std::to_string(channel) + "_" + label + ".csv");
        writeStatisticsCsv(csv_file, statistics);
        std::cout << "Saved " << csv_file << '\n';
        plotAllWaveformsWithStatistics(
            channel,
            records,
            statistics,
            run,
            label,
            "Final Selection",
            "final_selection",
            output_dir
        );
    }

    const fs::path output_stem = output_dir / ("mean_median_waveforms_" + label);
    canvas.SaveAs((output_stem.string() + ".pdf").c_str());
    canvas.SaveAs((output_stem.string() + ".png").c_str());
    std::cout << "Saved " << output_stem << ".pdf and .png\n";
}

void processInputList(
    const fs::path &input_list,
    const fs::path &program_dir,
    const fs::path &repo_dir,
    const Options &options
)
{
    if (!fs::is_regular_file(input_list)) {
        throw std::runtime_error("Input list not found: " + input_list.string());
    }
    const std::string run = inferRun(input_list);
    fs::path csv_file = options.csv_file;
    if (csv_file.empty()) {
        const std::string csv_name =
            "waveforms_run_" + run + "_" + options.csv_suffix;
        csv_file = input_list.parent_path() / csv_name;
        if (!fs::is_regular_file(csv_file)) csv_file = program_dir / csv_name;
    }

    if (!fs::is_regular_file(csv_file)) {
        throw std::runtime_error("Selection CSV not found: " + csv_file.string());
    }

    std::cout << "\nProcessing run " << run << '\n';
    std::set<unsigned int> channels;
    std::size_t selected_rows = 0;
    const auto selected = readSelectedKeys(csv_file, channels, selected_rows);
    std::cout << "Selected CSV rows: " << selected_rows << "\nSelected channels:";
    for (const auto channel : channels) std::cout << ' ' << channel;
    std::cout << '\n';

    const auto root_paths = readRootFiles(input_list, repo_dir);
    std::vector<std::string> root_files;
    root_files.reserve(root_paths.size());
    for (const auto &path : root_paths) root_files.push_back(path.string());
    auto chain = createChain(root_files);
    std::cout << "ROOT files: " << root_files.size()
              << "\nTree entries: " << chain->GetEntries() << '\n';

    const auto located = locateSelectedEntries(*chain, selected);
    std::cout << "Located ROOT entries: " << located.size() << '\n';
    const WaveformAnalyzer analyzer(options.config.string());
    const auto selections = loadWaveforms(
        *chain, located, analyzer, options.max_auxiliary_amplitude
    );
    const std::string label = options.csv_file.empty()
        ? "run_" + run + "_"
            + options.csv_suffix.substr(0, options.csv_suffix.size() - 4)
        : csv_file.stem().string();
    const fs::path cut_summary = options.output_dir
        / ("selection_cuts_" + label + ".txt");

    writeCutSummary(
        cut_summary,
        run,
        input_list,
        csv_file,
        options.config,
        channels,
        selected_rows,
        located.size(),
        options.max_auxiliary_amplitude,
        analyzer,
        selections.cuts,
        selections.final_selection
    );

    plotAllWaveformsForSelection(
        channels,
        selections.from_coincidence,
        run,
        label,
        "From Coincidence",
        "from_coincidence",
        options.output_dir
    );
    plotAndSaveStatistics(
        channels,
        selections.final_selection,
        run,
        label,
        options.output_dir
    );
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const fs::path executable = fs::weakly_canonical(fs::absolute(argv[0]));
        const fs::path executable_dir = executable.parent_path();
        const fs::path program_dir = executable_dir.filename() == "bin"
            ? executable_dir.parent_path() / "coincidence"
            : executable_dir;
        const fs::path repo_dir = program_dir.parent_path();
        const Options options = parseOptions(argc, argv, program_dir);
        if (!fs::is_regular_file(options.config)) {
            throw std::runtime_error("Configuration file not found: " + options.config.string());
        }
        fs::create_directories(options.output_dir);

        gROOT->SetBatch(kTRUE);
        gStyle->SetOptStat(0);
        for (const auto &input_list : options.input_lists) {
            processInputList(input_list, program_dir, repo_dir, options);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
