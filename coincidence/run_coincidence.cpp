#include "coincidence_lib.hpp"

#include <TBranch.h>
#include <TChain.h>
#include <TTree.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
    std::vector<unsigned int> left{2030, 2031, 2040, 2041};
    std::vector<unsigned int> right{2050, 2051, 2060, 2061};
    std::vector<unsigned int> save{2070, 2071, 2080, 2081};
    int window_ticks = 1;
    double min_amplitude_adc = 1000.0;
    std::string run;
};

struct WaveformRecord {
    std::int64_t pulse_time;
    int waveform_index;
};

struct CoincidentPair {
    std::int64_t minimum_time;
    std::int64_t maximum_time;
    std::int64_t dt_ns;
    unsigned int left_channel;
    int left_waveform_index;
    unsigned int right_channel;
    int right_waveform_index;
};

struct Candidate {
    std::int64_t dt_ns_coinc;
    std::int64_t dt_ns_save;
    unsigned int left_channel;
    int left_waveform_index;
    unsigned int right_channel;
    int right_waveform_index;
};

using EventChannels = std::unordered_map<unsigned int, std::vector<WaveformRecord>>;
using WaveformKey = std::tuple<int, unsigned int, int>;

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
        << "  --channels-coincident-left CH [CH ...]\n"
        << "  --channels-coincident-right CH [CH ...]\n"
        << "  --channels-to-save CH [CH ...]\n"
        << "  --window-ticks N\n"
        << "  --min-amplitude-adc X (default: 1000)\n";
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

    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "-h" || option == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (option == "--config") {
            if (++index >= argc) throw std::runtime_error("--config requires a value");
            args.config = argv[index];
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
        } else if (option == "--run") {
            if (++index >= argc) throw std::runtime_error("--run requires a value");
            args.run = argv[index];
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
    if (args.window_ticks < 0) throw std::runtime_error("--window-ticks cannot be negative");
    if (args.min_amplitude_adc < 0.0) {
        throw std::runtime_error("--min-amplitude-adc cannot be negative");
    }
    return args;
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

std::string makeOutputFilename(const Arguments &args)
{
    std::ostringstream min_amplitude;
    min_amplitude << args.min_amplitude_adc;

    return "waveforms_run_" + args.run
        + "_coinc_" + joinChannels(args.left, "-")
        + "_vs_" + joinChannels(args.right, "-")
        + "_save_" + joinChannels(args.save, "-")
        + "_window_" + std::to_string(args.window_ticks) + "_ticks"
        + "_min_amplitude_" + min_amplitude.str() + "_adc.csv";
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

void scanCoincidence(
    TChain &chain,
    const WaveformAnalyzer &analyzer,
    const Arguments &args,
    const std::string &output_file
)
{
    Int_t event = 0;
    UInt_t channel = 0;
    ULong64_t timestamp = 0;
    Int_t waveform_index = 0;
    std::vector<short> *adc = nullptr;

    std::set<unsigned int> valid_channels(args.left.begin(), args.left.end());
    valid_channels.insert(args.right.begin(), args.right.end());
    valid_channels.insert(args.save.begin(), args.save.end());

    std::map<int, EventChannels> by_event;
    std::size_t relevant_waveforms = 0;
    std::size_t accepted_waveforms = 0;
    std::size_t subthreshold_waveforms = 0;
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

        if (valid_channels.count(channel) != 0) {
            ++relevant_waveforms;
            if (adc_branch->GetEntry(local_entry) <= 0 || adc == nullptr || adc->empty()) {
                throw std::runtime_error("Could not read ADC at chain entry " + std::to_string(entry));
            }

            const double amplitude = analyzer.baselineAdjustedSignalAmplitude(*adc, channel);
            if (amplitude > args.min_amplitude_adc) {
                const std::int64_t pulse_time = static_cast<std::int64_t>(timestamp)
                    + analyzer.pulseStart(*adc, channel) * SAMPLE_PERIOD_NS;
                by_event[event][channel].push_back({pulse_time, waveform_index});
                ++accepted_waveforms;
            } else {
                ++subthreshold_waveforms;
            }
        }

        if (entry == 0 || (entry + 1) % 1000000 == 0 || entry + 1 == entries) {
            std::cout << "Input progress: " << entry + 1 << '/' << entries
                      << " entries; accepted waveforms=" << accepted_waveforms << std::endl;
        }
    }

    std::cout << "Accepted " << accepted_waveforms << " of " << relevant_waveforms
              << " relevant waveforms in " << by_event.size() << " events; rejected "
              << subthreshold_waveforms << " with baseline-subtracted amplitude <= "
              << args.min_amplitude_adc << " ADC" << std::endl;

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
                        coincident_pairs.push_back({
                            std::min(left_record.pulse_time, right->pulse_time),
                            std::max(left_record.pulse_time, right->pulse_time),
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
        << "channel_coincident_left,waveform_index_coincident_left,"
        << "channel_coincident_right,waveform_index_coincident_right\r\n";
    for (const auto &[key, candidate] : selected) {
        const auto &[selected_event, selected_channel, selected_waveform] = key;
        output << selected_event << ',' << selected_channel << ',' << selected_waveform << ','
               << candidate.dt_ns_coinc << ',' << candidate.dt_ns_save << ','
               << candidate.left_channel << ',' << candidate.left_waveform_index << ','
               << candidate.right_channel << ',' << candidate.right_waveform_index << "\r\n";
    }
    if (!output) throw std::runtime_error("Failed while writing output file: " + output_file);
    std::cout << "Saved " << selected.size() << " unique waveforms to " << output_file
              << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const Arguments args = parseArguments(argc, argv);
        const auto files = readInputList(args.input_list);
        if (files.empty()) throw std::runtime_error("No ROOT files found in input list");

        const std::string output_file = makeOutputFilename(args);
        const WaveformAnalyzer analyzer(args.config.string());
        auto chain = createChain(files);

        std::cout << "Output file: " << output_file << '\n'
                  << "Input files: " << files.size() << '\n'
                  << "Entries: " << chain->GetEntries() << '\n'
                  << "Minimum baseline-subtracted waveform amplitude: > "
                  << args.min_amplitude_adc << " ADC\n"
                  << "Testing left channels [" << joinChannels(args.left, ", ")
                  << "] against right channels [" << joinChannels(args.right, ", ")
                  << "], saving channels [" << joinChannels(args.save, ", ") << ']'
                  << std::endl;

        scanCoincidence(*chain, analyzer, args, output_file);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
