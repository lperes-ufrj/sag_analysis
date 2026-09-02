#include "coincidence_lib.hpp"

#include <TBranch.h>
#include <TChain.h>
#include <TTree.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Arguments {
    fs::path input_list;
    fs::path config;
    fs::path output_dir;
    std::vector<unsigned int> left{2030, 2031, 2040, 2041};
    std::vector<unsigned int> right{2050, 2051, 2060, 2061};
    std::vector<unsigned int> save{2070, 2071, 2080, 2081};
    int window_ticks = 10;
    double min_amplitude_adc = 0.0;
    fs::path norm_rate_adc_threshold_file;
    std::string run;
    std::string timestamp;
};

struct WaveformRecord {
    std::int64_t pulse_time;
    std::uint64_t timestamp;
    int waveform_index;
    bool coincidence_eligible;
    bool save_eligible;
};

struct CoincidentPair {
    std::int64_t minimum_time;
    std::int64_t maximum_time;
    std::uint64_t minimum_timestamp;
    std::int64_t dt_ns;
    unsigned int left_channel;
    int left_waveform_index;
    unsigned int right_channel;
    int right_waveform_index;
};

struct Candidate {
    std::int64_t dt_ns_coinc;
    std::int64_t dt_ns_save;
    std::uint64_t minimum_timestamp;
    unsigned int left_channel;
    int left_waveform_index;
    unsigned int right_channel;
    int right_waveform_index;
};

using EventChannels = std::unordered_map<unsigned int, std::vector<WaveformRecord>>;
using WaveformKey = std::tuple<int, unsigned int, int>;
struct NormRateAdcThreshold {
    double target_frequency_khz = std::numeric_limits<double>::quiet_NaN();
    double threshold_adc = std::numeric_limits<double>::quiet_NaN();
};

using NormRateAdcThresholds =
    std::unordered_map<unsigned int, NormRateAdcThreshold>;

struct RunStatistics {
    Long64_t tree_entries = 0;
    std::size_t unique_input_events = 0;
    std::size_t relevant_waveforms = 0;
    std::size_t coincidence_waveforms_analyzed = 0;
    std::size_t coincidence_waveforms_accepted = 0;
    std::size_t coincidence_waveforms_rejected = 0;
    std::size_t saved_channel_waveforms_analyzed = 0;
    std::size_t saved_channel_waveforms_accepted = 0;
    std::size_t saved_channel_waveforms_rejected = 0;
    std::size_t events_with_accepted_waveforms = 0;
    std::size_t coincident_events = 0;
    std::uint64_t coincident_pairs = 0;
    std::size_t selected_events = 0;
    std::size_t selected_waveforms = 0;
};

// Keep these half-open ranges synchronized with analysis/rate_analysis.py.
constexpr std::size_t BASELINE_FIRST_SAMPLE = 0;
constexpr std::size_t BASELINE_LAST_SAMPLE = 50;
constexpr std::size_t SIGNAL_FIRST_SAMPLE = 50;
constexpr std::size_t SIGNAL_LAST_SAMPLE = 180;

std::string trim(const std::string &text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::string joinChannels(const std::vector<unsigned int> &channels, const char *separator)
{
    std::ostringstream output;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        if (index != 0) output << separator;
        output << channels[index];
    }
    return output.str();
}

void printUsage(const char *program)
{
    std::cout
        << "Usage: " << program << " INPUT_LIST --run RUN [options]\n"
        << "  --config FILE\n"
        << "  --output-dir DIR (timestamped subdirectory is created here)\n"
        << "  --channels-coincident-left CH [CH ...]\n"
        << "  --channels-coincident-right CH [CH ...]\n"
        << "  --channels-to-save CH [CH ...]\n"
        << "  --window-ticks N\n"
        << "  --min-amplitude-adc X (default: 0)\n"
        << "  --timestamp ID (reuse one analysis identifier across multiple runs)\n"
        << "  --norm-rate-adc-threshold-file FILE"
        << " (optional per-channel pre-coincidence cut for saved channels)\n";
}

std::vector<unsigned int> parseChannels(int &index, int argc, char **argv)
{
    std::vector<unsigned int> channels;
    while (index + 1 < argc && std::string(argv[index + 1]).rfind("--", 0) != 0) {
        channels.push_back(static_cast<unsigned int>(std::stoul(argv[++index])));
    }
    if (channels.empty()) throw std::runtime_error("A channel option requires at least one value");
    return channels;
}

Arguments parseArguments(int argc, char **argv)
{
    Arguments args;
    const fs::path executable = fs::absolute(argv[0]);
    const fs::path executable_dir = executable.parent_path();
    const fs::path program_dir = executable_dir.filename() == "bin"
        ? executable_dir.parent_path() / "coincidence"
        : executable_dir;
    args.config = program_dir / "waveform_intervals.ini";
    args.output_dir = program_dir / "saved_coincidences";

    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "-h" || option == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (option == "--config") {
            if (++index >= argc) throw std::runtime_error("--config requires a value");
            args.config = argv[index];
        } else if (option == "--output-dir") {
            if (++index >= argc) throw std::runtime_error("--output-dir requires a value");
            args.output_dir = argv[index];
        } else if (option == "--channels-coincident-left") {
            args.left = parseChannels(index, argc, argv);
        } else if (option == "--channels-coincident-right") {
            args.right = parseChannels(index, argc, argv);
        } else if (option == "--channels-to-save") {
            args.save = parseChannels(index, argc, argv);
        } else if (option == "--window-ticks") {
            if (++index >= argc) throw std::runtime_error("--window-ticks requires a value");
            args.window_ticks = std::stoi(argv[index]);
        } else if (option == "--min-amplitude-adc") {
            if (++index >= argc) {
                throw std::runtime_error("--min-amplitude-adc requires a value");
            }
            args.min_amplitude_adc = std::stod(argv[index]);
        } else if (option == "--norm-rate-adc-threshold-file"
                   || option == "--norm-rate-adc-threshold") {
            if (++index >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            args.norm_rate_adc_threshold_file = argv[index];
        } else if (option == "--run") {
            if (++index >= argc) throw std::runtime_error("--run requires a value");
            args.run = argv[index];
        } else if (option == "--timestamp") {
            if (++index >= argc) throw std::runtime_error("--timestamp requires a value");
            args.timestamp = argv[index];
        } else if (option.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + option);
        } else if (args.input_list.empty()) {
            args.input_list = option;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + option);
        }
    }

    if (args.input_list.empty()) throw std::runtime_error("Missing input list");
    if (args.run.empty()) throw std::runtime_error("Missing required option --run");
    if (args.output_dir.empty()) throw std::runtime_error("--output-dir cannot be empty");
    if (args.window_ticks < 0) throw std::runtime_error("--window-ticks cannot be negative");
    if (args.min_amplitude_adc < 0.0) {
        throw std::runtime_error("--min-amplitude-adc cannot be negative");
    }
    if (!args.timestamp.empty()
        && !std::all_of(args.timestamp.begin(), args.timestamp.end(), [](unsigned char value) {
            return std::isalnum(value) || value == '-' || value == '_';
        })) {
        throw std::runtime_error(
            "--timestamp may contain only letters, digits, '-' and '_'"
        );
    }
    return args;
}

std::vector<std::string> splitFields(const std::string &line)
{
    std::istringstream input(line);
    std::vector<std::string> fields;
    std::string field;
    while (input >> field) fields.push_back(field);
    return fields;
}

NormRateAdcThresholds readNormRateAdcThresholds(const fs::path &threshold_file)
{
    std::ifstream input(threshold_file);
    if (!input) {
        throw std::runtime_error(
            "Could not open normalized-rate ADC-threshold file: "
            + threshold_file.string()
        );
    }

    NormRateAdcThresholds thresholds;
    bool found_header = false;
    std::size_t channel_column = 0;
    std::size_t target_frequency_column = 0;
    std::size_t threshold_column = 0;
    std::size_t line_number = 0;
    std::string line;

    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty()) continue;

        const bool comment = line.front() == '#';
        if (comment) line = trim(line.substr(1));
        if (line.empty()) continue;

        const auto fields = splitFields(line);
        if (!found_header) {
            const auto channel = std::find(fields.begin(), fields.end(), "channel");
            const auto target_frequency = std::find(
                fields.begin(), fields.end(), "target_frequency_khz"
            );
            const auto threshold = std::find(fields.begin(), fields.end(), "threshold_adc");
            if (channel != fields.end() && target_frequency != fields.end()
                && threshold != fields.end()) {
                channel_column = static_cast<std::size_t>(
                    std::distance(fields.begin(), channel)
                );
                target_frequency_column = static_cast<std::size_t>(
                    std::distance(fields.begin(), target_frequency)
                );
                threshold_column = static_cast<std::size_t>(
                    std::distance(fields.begin(), threshold)
                );
                found_header = true;
                continue;
            }
            if (comment) continue;
            throw std::runtime_error(
                "Normalized-rate ADC-threshold file must contain 'channel', "
                "'target_frequency_khz', and 'threshold_adc' "
                "columns: " + threshold_file.string()
            );
        }

        if (comment) continue;
        const std::size_t required_column = std::max(
            {channel_column, target_frequency_column, threshold_column}
        );
        if (fields.size() <= required_column) {
            throw std::runtime_error(
                "Malformed normalized-rate ADC-threshold row "
                + std::to_string(line_number)
                + " in " + threshold_file.string()
            );
        }

        unsigned int channel = 0;
        NormRateAdcThreshold rate_threshold;
        try {
            const std::string &channel_text = fields[channel_column];
            if (!channel_text.empty() && channel_text.front() == '-') {
                throw std::invalid_argument("negative channel");
            }
            std::size_t consumed = 0;
            const unsigned long long parsed_channel = std::stoull(channel_text, &consumed);
            if (consumed != channel_text.size()
                || parsed_channel > std::numeric_limits<unsigned int>::max()) {
                throw std::out_of_range("invalid channel");
            }
            channel = static_cast<unsigned int>(parsed_channel);

            const std::string &target_frequency_text = fields[target_frequency_column];
            consumed = 0;
            rate_threshold.target_frequency_khz = std::stod(
                target_frequency_text, &consumed
            );
            if (consumed != target_frequency_text.size()) {
                throw std::invalid_argument("invalid target frequency");
            }

            const std::string &threshold_text = fields[threshold_column];
            consumed = 0;
            rate_threshold.threshold_adc = std::stod(threshold_text, &consumed);
            if (consumed != threshold_text.size()) {
                throw std::invalid_argument("invalid threshold");
            }
        } catch (const std::exception &) {
            throw std::runtime_error(
                "Invalid normalized-rate ADC-threshold values on row "
                + std::to_string(line_number)
                + " in " + threshold_file.string()
            );
        }

        if (!thresholds.emplace(channel, rate_threshold).second) {
            throw std::runtime_error(
                "Duplicate normalized-rate ADC-threshold channel "
                + std::to_string(channel)
                + " in " + threshold_file.string()
            );
        }
    }

    if (!found_header) {
        throw std::runtime_error(
            "Normalized-rate ADC-threshold file has no header: "
            + threshold_file.string()
        );
    }
    return thresholds;
}

std::vector<unsigned int> selectChannelsWithFiniteRateThresholds(
    Arguments &args,
    const NormRateAdcThresholds &thresholds
)
{
    std::vector<unsigned int> selected_channels;
    std::vector<unsigned int> skipped_channels;
    selected_channels.reserve(args.save.size());

    for (const unsigned int channel : args.save) {
        const auto found = thresholds.find(channel);
        if (found == thresholds.end()) {
            std::cerr << "Skipping saved channel " << channel
                      << ": no row is present in normalized-rate threshold file "
                      << args.norm_rate_adc_threshold_file << '\n';
            skipped_channels.push_back(channel);
            continue;
        }

        const auto &rate_threshold = found->second;
        if (!std::isfinite(rate_threshold.target_frequency_khz)
            || rate_threshold.target_frequency_khz < 0.0
            || !std::isfinite(rate_threshold.threshold_adc)) {
            std::cerr << "Skipping saved channel " << channel
                      << ": target frequency or ADC threshold is not finite and valid in "
                      << args.norm_rate_adc_threshold_file << '\n';
            skipped_channels.push_back(channel);
            continue;
        }
        // Negative finite thresholds are intentional results of rate_analysis.py.
        // Clamping them would change the selected rate, so preserve them exactly.
        selected_channels.push_back(channel);
    }

    args.save = std::move(selected_channels);
    return skipped_channels;
}

double baselineToPeakAmplitude(const std::vector<short> &adc)
{
    const std::size_t baseline_stop = std::min(BASELINE_LAST_SAMPLE, adc.size());
    const std::size_t signal_stop = std::min(SIGNAL_LAST_SAMPLE, adc.size());
    if (BASELINE_FIRST_SAMPLE >= baseline_stop || SIGNAL_FIRST_SAMPLE >= signal_stop) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double baseline_sum = std::accumulate(
        adc.begin() + static_cast<std::ptrdiff_t>(BASELINE_FIRST_SAMPLE),
        adc.begin() + static_cast<std::ptrdiff_t>(baseline_stop),
        0.0
    );
    const double baseline = baseline_sum
        / static_cast<double>(baseline_stop - BASELINE_FIRST_SAMPLE);
    const auto peak = std::max_element(
        adc.begin() + static_cast<std::ptrdiff_t>(SIGNAL_FIRST_SAMPLE),
        adc.begin() + static_cast<std::ptrdiff_t>(signal_stop)
    );
    return static_cast<double>(*peak) - baseline;
}

bool passesNormRateAdcThreshold(double amplitude, double threshold)
{
    return std::isfinite(amplitude) && amplitude >= threshold;
}

std::vector<std::string> readInputList(const fs::path &input_list)
{
    std::ifstream input(input_list);
    if (!input) throw std::runtime_error("Could not open input list: " + input_list.string());

    std::vector<std::string> files;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        files.push_back(fs::weakly_canonical(input_list.parent_path() / line).string());
    }
    return files;
}

std::string makeExecutionTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&time, &local_time);

    std::ostringstream timestamp;
    timestamp << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return timestamp.str();
}

std::string makeOutputFilename(const Arguments &args, const std::string &timestamp)
{
    return "coincidence_scan_run_" + args.run + "_" + timestamp + ".csv";
}

void CreateTXTfile(
    const fs::path &txt_file,
    const fs::path &csv_file,
    const std::string &execution_timestamp,
    const Arguments &args,
    const NormRateAdcThresholds &norm_rate_adc_thresholds,
    const std::vector<unsigned int> &requested_save_channels,
    const std::vector<unsigned int> &skipped_save_channels,
    const std::vector<std::string> &input_files,
    const RunStatistics &statistics
)
{
    const bool existing_summary = fs::exists(txt_file) && fs::file_size(txt_file) != 0;
    std::ofstream output(txt_file, std::ios::app);
    if (!output) {
        throw std::runtime_error("Could not open TXT summary: " + txt_file.string());
    }

    if (!existing_summary) {
        output << "Coincidence analysis summary\n"
               << "============================\n"
               << "Analysis identifier: " << execution_timestamp << "\n";
    }

    output << "\nRun " << args.run << '\n'
           << std::string(4 + args.run.size(), '-') << '\n'
           << "Analysis identifier: " << execution_timestamp << '\n'
           << "CSV output: " << csv_file.string() << '\n'
           << "Input list: " << args.input_list.string() << '\n'
           << "Waveform configuration: " << args.config.string() << '\n'
           << "Input ROOT files: " << input_files.size() << "\n\n"
           << "Selection configuration\n"
           << "-----------------------\n"
           << "Coincidence channels (left): " << joinChannels(args.left, ", ") << '\n'
           << "Coincidence channels (right): " << joinChannels(args.right, ", ") << '\n'
           << "Saved channels requested: "
           << joinChannels(requested_save_channels, ", ") << '\n'
           << "Saved channels used: "
           << (args.save.empty() ? std::string("none") : joinChannels(args.save, ", "))
           << '\n'
           << "Saved channels skipped for missing/invalid rate threshold: "
           << (skipped_save_channels.empty()
                   ? std::string("none")
                   : joinChannels(skipped_save_channels, ", "))
           << '\n'
           << "Coincidence window: " << args.window_ticks << " ticks ("
           << args.window_ticks * SAMPLE_PERIOD_NS << " ns)\n"
           << "Minimum baseline-subtracted amplitude: > "
           << args.min_amplitude_adc << " ADC\n"
           << "Normalized-rate ADC-threshold file: "
           << (args.norm_rate_adc_threshold_file.empty()
                   ? std::string("none")
                   : args.norm_rate_adc_threshold_file.string())
           << '\n';

    if (!args.norm_rate_adc_threshold_file.empty()) {
        output << "Normalized-rate ADC thresholds applied before coincidence matching:\n";
        for (const unsigned int channel : args.save) {
            const auto &rate_threshold = norm_rate_adc_thresholds.at(channel);
            output << "  Channel " << channel
                   << ": target=" << rate_threshold.target_frequency_khz
                   << " kHz, retain amplitude >= "
                   << rate_threshold.threshold_adc << " ADC\n";
        }
        output << "Threshold amplitude definition: "
               << "max(ADC[50:180]) - mean(ADC[0:50]) (half-open ranges)\n"
               << "Threshold comparison: amplitude >= threshold_adc\n";
    }

    output << "\nProcessing statistics\n"
           << "---------------------\n"
           << "Tree entries analyzed: " << statistics.tree_entries << '\n'
           << "Unique input events analyzed: " << statistics.unique_input_events << '\n'
           << "Relevant-channel waveforms analyzed: "
           << statistics.relevant_waveforms << '\n'
           << "Coincidence-channel waveforms analyzed: "
           << statistics.coincidence_waveforms_analyzed << '\n'
           << "Coincidence-channel waveforms passing minimum amplitude: "
           << statistics.coincidence_waveforms_accepted << '\n'
           << "Coincidence-channel waveforms rejected by minimum amplitude: "
           << statistics.coincidence_waveforms_rejected << '\n'
           << "Saved-channel waveforms analyzed: "
           << statistics.saved_channel_waveforms_analyzed << '\n'
           << "Saved-channel waveforms passing their pre-coincidence cut: "
           << statistics.saved_channel_waveforms_accepted << '\n'
           << "Saved-channel waveforms rejected by their pre-coincidence cut: "
           << statistics.saved_channel_waveforms_rejected << '\n'
           << "Events with accepted relevant waveforms: "
           << statistics.events_with_accepted_waveforms << '\n'
           << "Events containing coincidence pairs: "
           << statistics.coincident_events << '\n'
           << "Coincident pairs found: " << statistics.coincident_pairs << '\n'
           << "Final selected events written to CSV: "
           << statistics.selected_events << '\n'
           << "Final selected waveforms written to CSV: "
           << statistics.selected_waveforms << "\n\n"
           << "Input ROOT files\n"
           << "----------------\n";

    for (const auto &input_file : input_files) output << input_file << '\n';

    if (!output) {
        throw std::runtime_error("Failed while writing TXT summary: " + txt_file.string());
    }
}

bool betterCandidate(const Candidate &candidate, const Candidate &previous)
{
    return std::tuple{
        std::llabs(candidate.dt_ns_save),
        std::llabs(candidate.dt_ns_coinc),
        candidate.dt_ns_coinc,
        candidate.dt_ns_save,
        candidate.left_channel,
        candidate.left_waveform_index,
        candidate.right_channel,
        candidate.right_waveform_index
    } < std::tuple{
        std::llabs(previous.dt_ns_save),
        std::llabs(previous.dt_ns_coinc),
        previous.dt_ns_coinc,
        previous.dt_ns_save,
        previous.left_channel,
        previous.left_waveform_index,
        previous.right_channel,
        previous.right_waveform_index
    };
}

const std::vector<WaveformRecord> &channelRecords(
    const EventChannels &channels,
    unsigned int channel
)
{
    static const std::vector<WaveformRecord> empty;
    const auto found = channels.find(channel);
    return found == channels.end() ? empty : found->second;
}

RunStatistics scanCoincidence(
    TChain &chain,
    const WaveformAnalyzer &analyzer,
    const Arguments &args,
    const NormRateAdcThresholds &norm_rate_adc_thresholds,
    const std::string &output_file
)
{
    Int_t event = 0;
    UInt_t channel = 0;
    ULong64_t timestamp = 0;
    Int_t waveform_index = 0;
    std::vector<short> *adc = nullptr;

    std::set<unsigned int> coincidence_channels(args.left.begin(), args.left.end());
    coincidence_channels.insert(args.right.begin(), args.right.end());
    const std::set<unsigned int> save_channels(args.save.begin(), args.save.end());
    std::set<unsigned int> valid_channels = coincidence_channels;
    valid_channels.insert(save_channels.begin(), save_channels.end());

    std::map<int, EventChannels> by_event;
    std::size_t relevant_waveforms = 0;
    std::size_t coincidence_waveforms_analyzed = 0;
    std::size_t coincidence_waveforms_accepted = 0;
    std::size_t coincidence_waveforms_rejected = 0;
    std::size_t saved_channel_waveforms_analyzed = 0;
    std::size_t saved_channel_waveforms_accepted = 0;
    std::size_t saved_channel_waveforms_rejected = 0;
    std::size_t stored_waveforms = 0;
    std::set<int> unique_input_events;
    std::cout << "Scanning input and calculating pulse times..." << std::endl;

    const Long64_t entries = chain.GetEntries();
    int current_tree_number = -1;
    TBranch *event_branch = nullptr;
    TBranch *channel_branch = nullptr;
    TBranch *timestamp_branch = nullptr;
    TBranch *waveform_index_branch = nullptr;
    TBranch *adc_branch = nullptr;
    for (Long64_t entry = 0; entry < entries; ++entry) {
        const Long64_t local_entry = chain.LoadTree(entry);
        if (local_entry < 0) {
            throw std::runtime_error("Could not load chain entry " + std::to_string(entry));
        }

        if (chain.GetTreeNumber() != current_tree_number) {
            current_tree_number = chain.GetTreeNumber();
            TTree *tree = chain.GetTree();
            event_branch = tree->GetBranch("event");
            channel_branch = tree->GetBranch("channel");
            timestamp_branch = tree->GetBranch("timestamp");
            waveform_index_branch = tree->GetBranch("waveform_index");
            adc_branch = tree->GetBranch("adc");
            if (event_branch == nullptr || channel_branch == nullptr
                || timestamp_branch == nullptr || waveform_index_branch == nullptr
                || adc_branch == nullptr) {
                throw std::runtime_error("Missing required branch in input tree");
            }
            event_branch->SetAddress(&event);
            channel_branch->SetAddress(&channel);
            timestamp_branch->SetAddress(&timestamp);
            waveform_index_branch->SetAddress(&waveform_index);
            adc_branch->SetAddress(&adc);
        }

        if (event_branch->GetEntry(local_entry) <= 0
            || channel_branch->GetEntry(local_entry) <= 0
            || timestamp_branch->GetEntry(local_entry) <= 0
            || waveform_index_branch->GetEntry(local_entry) <= 0) {
            throw std::runtime_error("Could not read metadata at chain entry " + std::to_string(entry));
        }
        unique_input_events.insert(event);

        if (valid_channels.count(channel) != 0) {
            ++relevant_waveforms;
            if (adc_branch->GetEntry(local_entry) <= 0 || adc == nullptr || adc->empty()) {
                throw std::runtime_error("Could not read ADC at chain entry " + std::to_string(entry));
            }

            bool coincidence_eligible = false;
            if (coincidence_channels.count(channel) != 0) {
                ++coincidence_waveforms_analyzed;
                const double amplitude =
                    analyzer.baselineAdjustedSignalAmplitude(*adc, channel);
                coincidence_eligible = amplitude > args.min_amplitude_adc;
                if (coincidence_eligible) {
                    ++coincidence_waveforms_accepted;
                } else {
                    ++coincidence_waveforms_rejected;
                }
            }

            bool save_eligible = false;
            if (save_channels.count(channel) != 0) {
                ++saved_channel_waveforms_analyzed;
                if (norm_rate_adc_thresholds.empty()) {
                    save_eligible = true;
                } else {
                    const double amplitude = baselineToPeakAmplitude(*adc);
                    save_eligible = passesNormRateAdcThreshold(
                        amplitude,
                        norm_rate_adc_thresholds.at(channel).threshold_adc
                    );
                }
                if (save_eligible) {
                    ++saved_channel_waveforms_accepted;
                } else {
                    ++saved_channel_waveforms_rejected;
                }
            }

            if (coincidence_eligible || save_eligible) {
                const std::int64_t pulse_time = static_cast<std::int64_t>(timestamp)
                    + analyzer.pulseStart(*adc, channel) * SAMPLE_PERIOD_NS;
                by_event[event][channel].push_back({
                    pulse_time,
                    static_cast<std::uint64_t>(timestamp),
                    waveform_index,
                    coincidence_eligible,
                    save_eligible
                });
                ++stored_waveforms;
            }
        }

        if (entry == 0 || (entry + 1) % 1000000 == 0 || entry + 1 == entries) {
            std::cout << "Input progress: " << entry + 1 << '/' << entries
                      << " entries; retained for at least one role="
                      << stored_waveforms << std::endl;
        }
    }

    std::cout << "Analyzed " << relevant_waveforms << " relevant waveforms; "
              << coincidence_waveforms_accepted << " of "
              << coincidence_waveforms_analyzed
              << " coincidence-channel waveforms passed > "
              << args.min_amplitude_adc << " ADC; "
              << saved_channel_waveforms_accepted << " of "
              << saved_channel_waveforms_analyzed
              << " saved-channel waveforms passed their pre-coincidence cut"
              << std::endl;

    for (auto &[event_number, channels] : by_event) {
        (void)event_number;
        for (auto &[channel_number, records] : channels) {
            (void)channel_number;
            std::stable_sort(
                records.begin(), records.end(),
                [](const auto &left, const auto &right) {
                    return left.pulse_time < right.pulse_time;
                }
            );
        }
    }

    const std::int64_t window_ns = args.window_ticks * SAMPLE_PERIOD_NS;
    std::map<WaveformKey, Candidate> selected;
    std::size_t coincident_events = 0;
    std::uint64_t coincident_pairs_total = 0;
    std::size_t event_position = 0;

    for (const auto &[event_number, channels] : by_event) {
        ++event_position;
        std::vector<CoincidentPair> coincident_pairs;

        for (const unsigned int left_channel : args.left) {
            const auto &left_records = channelRecords(channels, left_channel);
            for (const unsigned int right_channel : args.right) {
                const auto &right_records = channelRecords(channels, right_channel);

                for (const auto &left_record : left_records) {
                    if (!left_record.coincidence_eligible) continue;
                    const auto begin = std::lower_bound(
                        right_records.begin(), right_records.end(),
                        left_record.pulse_time - window_ns,
                        [](const WaveformRecord &record, std::int64_t time) {
                            return record.pulse_time < time;
                        }
                    );
                    const auto end = std::upper_bound(
                        right_records.begin(), right_records.end(),
                        left_record.pulse_time + window_ns,
                        [](std::int64_t time, const WaveformRecord &record) {
                            return time < record.pulse_time;
                        }
                    );

                    for (auto right = begin; right != end; ++right) {
                        if (!right->coincidence_eligible) continue;
                        coincident_pairs.push_back({
                            std::min(left_record.pulse_time, right->pulse_time),
                            std::max(left_record.pulse_time, right->pulse_time),
                            std::min(left_record.timestamp, right->timestamp),
                            right->pulse_time - left_record.pulse_time,
                            left_channel,
                            left_record.waveform_index,
                            right_channel,
                            right->waveform_index
                        });
                    }
                }
            }
        }

        if (!coincident_pairs.empty()) {
            ++coincident_events;
            coincident_pairs_total += coincident_pairs.size();
        }
        std::stable_sort(
            coincident_pairs.begin(), coincident_pairs.end(),
            [](const CoincidentPair &left, const CoincidentPair &right) {
                return left.minimum_time < right.minimum_time;
            }
        );

        for (const unsigned int save_channel : args.save) {
            const auto &save_records = channelRecords(channels, save_channel);
            for (const auto &save_record : save_records) {
                if (!save_record.save_eligible) continue;
                const auto begin = std::lower_bound(
                    coincident_pairs.begin(), coincident_pairs.end(),
                    save_record.pulse_time - 2 * window_ns,
                    [](const CoincidentPair &pair, std::int64_t time) {
                        return pair.minimum_time < time;
                    }
                );
                const auto end = std::upper_bound(
                    coincident_pairs.begin(), coincident_pairs.end(),
                    save_record.pulse_time + window_ns,
                    [](std::int64_t time, const CoincidentPair &pair) {
                        return time < pair.minimum_time;
                    }
                );

                for (auto pair = begin; pair != end; ++pair) {
                    if (pair->maximum_time < save_record.pulse_time - window_ns) continue;

                    const WaveformKey key{event_number, save_channel, save_record.waveform_index};
                    const Candidate candidate{
                        pair->dt_ns,
                        save_record.pulse_time - pair->minimum_time,
                        pair->minimum_timestamp,
                        pair->left_channel,
                        pair->left_waveform_index,
                        pair->right_channel,
                        pair->right_waveform_index
                    };
                    const auto previous = selected.find(key);
                    if (previous == selected.end() || betterCandidate(candidate, previous->second)) {
                        selected[key] = candidate;
                    }
                }
            }
        }

        if (event_position == 1 || event_position % 1000 == 0
            || event_position == by_event.size()) {
            std::cout << "Progress: " << event_position << '/' << by_event.size()
                      << " events; current event=" << event_number
                      << "; coincident events=" << coincident_events
                      << "; coincident pairs=" << coincident_pairs_total
                      << "; unique waveforms selected=" << selected.size() << std::endl;
        }
    }

    std::ofstream output(output_file, std::ios::binary);
    if (!output) throw std::runtime_error("Could not open output file: " + output_file);
    output
        << "event,channel,waveform_index_save,dt_ns_coinc,dt_ns_save,"
        << "minimum_timestamp_coinc,"
        << "channel_coincident_left,waveform_index_coincident_left,"
        << "channel_coincident_right,waveform_index_coincident_right\r\n";
    for (const auto &[key, candidate] : selected) {
        const auto &[selected_event, selected_channel, selected_waveform] = key;
        output << selected_event << ','
               << selected_channel << ','
               << selected_waveform << ','
               << candidate.dt_ns_coinc << ','
               << candidate.dt_ns_save << ','
               << candidate.minimum_timestamp << ','
               << candidate.left_channel << ','
               << candidate.left_waveform_index << ','
               << candidate.right_channel << ','
               << candidate.right_waveform_index << "\r\n";
    }
    if (!output) throw std::runtime_error("Failed while writing output file: " + output_file);
    std::cout << "Saved " << selected.size() << " unique waveforms to " << output_file
              << std::endl;

    std::set<int> selected_events;
    for (const auto &[key, candidate] : selected) {
        (void)candidate;
        selected_events.insert(std::get<0>(key));
    }

    return {
        entries,
        unique_input_events.size(),
        relevant_waveforms,
        coincidence_waveforms_analyzed,
        coincidence_waveforms_accepted,
        coincidence_waveforms_rejected,
        saved_channel_waveforms_analyzed,
        saved_channel_waveforms_accepted,
        saved_channel_waveforms_rejected,
        by_event.size(),
        coincident_events,
        coincident_pairs_total,
        selected_events.size(),
        selected.size()
    };
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        Arguments args = parseArguments(argc, argv);
        const auto files = readInputList(args.input_list);
        if (files.empty()) throw std::runtime_error("No ROOT files found in input list");

        const std::vector<unsigned int> requested_save_channels = args.save;
        std::vector<unsigned int> skipped_save_channels;
        NormRateAdcThresholds norm_rate_adc_thresholds;
        if (!args.norm_rate_adc_threshold_file.empty()) {
            norm_rate_adc_thresholds = readNormRateAdcThresholds(
                args.norm_rate_adc_threshold_file
            );
            skipped_save_channels = selectChannelsWithFiniteRateThresholds(
                args,
                norm_rate_adc_thresholds
            );
        }

        const std::string execution_timestamp = args.timestamp.empty()
            ? makeExecutionTimestamp()
            : args.timestamp;
        const fs::path analysis_output_dir =
            args.output_dir / execution_timestamp;
        fs::create_directories(analysis_output_dir);
        const fs::path output_file = analysis_output_dir
            / makeOutputFilename(args, execution_timestamp);
        const fs::path txt_file = analysis_output_dir
            / ("coincidence_scan_" + execution_timestamp + ".txt");
        const WaveformAnalyzer analyzer(args.config.string());
        auto chain = createChain(files);

        std::cout << "Output file: " << output_file << '\n'
                  << "Input files: " << files.size() << '\n'
                  << "Entries: " << chain->GetEntries() << '\n'
                  << "Minimum baseline-subtracted waveform amplitude: > "
                  << args.min_amplitude_adc << " ADC\n"
                  << "Testing left channels [" << joinChannels(args.left, ", ")
                  << "] against right channels [" << joinChannels(args.right, ", ")
                  << "], saving channels ["
                  << (args.save.empty() ? std::string("none") : joinChannels(args.save, ", "))
                  << ']'
                  << std::endl;

        if (!args.norm_rate_adc_threshold_file.empty()) {
            std::cout << "Normalized-rate ADC thresholds applied to saved channels "
                      << "before coincidence matching: "
                      << args.norm_rate_adc_threshold_file << '\n';
            for (const unsigned int channel : args.save) {
                const auto &rate_threshold = norm_rate_adc_thresholds.at(channel);
                std::cout << "  Channel " << channel
                          << ": target=" << rate_threshold.target_frequency_khz
                          << " kHz, retain amplitude >= "
                          << rate_threshold.threshold_adc << " ADC\n";
            }
            std::cout
                << "  Amplitude = max(ADC[50:180]) - mean(ADC[0:50])"
                << " (half-open ranges); comparison is inclusive"
                << std::endl;
        }

        const RunStatistics statistics = scanCoincidence(
            *chain,
            analyzer,
            args,
            norm_rate_adc_thresholds,
            output_file.string()
        );
        CreateTXTfile(
            txt_file,
            output_file,
            execution_timestamp,
            args,
            norm_rate_adc_thresholds,
            requested_save_channels,
            skipped_save_channels,
            files,
            statistics
        );
        std::cout << "Summary file: " << txt_file << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
