#include "coincidence_lib.hpp"
#include "waveform_plotting.hpp"

#include <TBranch.h>
#include <TChain.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
    fs::path input_list;
    fs::path config;
    fs::path output_dir;
    std::string run;
    std::string timestamp;
    std::set<unsigned int> channels;
    unsigned int saturation_adc = 16383;
};

struct ChannelCounts {
    std::size_t analyzed = 0;
    std::size_t saturated = 0;
    std::size_t no_peak = 0;
    std::size_t primary_outside = 0;
    std::size_t additional_outside = 0;
    std::size_t selected = 0;
};

struct ScanResult {
    Long64_t entries = 0;
    std::size_t analyzed = 0;
    std::size_t skipped_channels = 0;
    std::size_t selected = 0;
    std::map<unsigned int, ChannelCounts> counts;
    waveform_plotting::RecordsByChannel waveforms;
};

std::string trim(const std::string &text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

void printUsage(const char *program)
{
    std::cout
        << "Usage: " << program << " INPUT_LIST --run RUN [options]\n\n"
        << "Run the shared peak finder on nonsaturated waveforms in the requested\n"
        << "channels, without coincidence matching. Write every requested waveform\n"
        << "to CSV. Plot waveforms with a detected peak and all significant\n"
        << "peaks inside the configured signal interval.\n\n"
        << "Options:\n"
        << "  --run RUN         Required six-digit run number\n"
        << "  --config FILE     Interval INI (default: waveform_intervals.ini)\n"
        << "  --output-dir DIR  Output base (default: coincidence/peak_finder_results)\n"
        << "  --timestamp ID    Analysis subdirectory (default: current local time)\n"
        << "  --channel CH      Analyze only CH; repeat for more channels (default: all)\n"
        << "  --saturation-adc N  Reject any waveform with raw ADC >= N (default: 16383)\n"
        << "  -h, --help        Show this help\n\n"
        << "Uses PEAK_SMOOTHING_WINDOW = 10 from coincidence_lib.cpp; the\n"
        << "existing inclusive window contains 11 samples away from edges.\n";
}

unsigned int parseUnsigned(const std::string &text, const std::string &option)
{
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return std::isdigit(c);
    })) {
        throw std::runtime_error(option + " requires a non-negative integer");
    }
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error(option + " is out of range");
    }
    return static_cast<unsigned int>(value);
}

Options parseOptions(int argc, char **argv)
{
    const fs::path executable_dir = fs::weakly_canonical(fs::absolute(argv[0])).parent_path();
    const fs::path program_dir = executable_dir.filename() == "bin"
        ? executable_dir.parent_path() / "coincidence" : executable_dir;
    Options options;
    options.config = program_dir / "waveform_intervals.ini";
    options.output_dir = program_dir / "peak_finder_results";
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (argument == "--run" || argument == "--config"
            || argument == "--output-dir" || argument == "--timestamp"
            || argument == "--channel" || argument == "--saturation-adc") {
            if (++index >= argc || std::string(argv[index]).empty()) {
                throw std::runtime_error("Missing value for " + argument);
            }
            if (argument == "--run") options.run = argv[index];
            else if (argument == "--config") options.config = argv[index];
            else if (argument == "--output-dir") options.output_dir = argv[index];
            else if (argument == "--timestamp") options.timestamp = argv[index];
            else if (argument == "--channel") {
                options.channels.insert(parseUnsigned(argv[index], argument));
            } else {
                options.saturation_adc = parseUnsigned(argv[index], argument);
            }
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        } else if (options.input_list.empty()) {
            options.input_list = argument;
        } else {
            throw std::runtime_error("Unexpected positional argument: " + argument);
        }
    }
    if (options.input_list.empty()) throw std::runtime_error("Missing input list");
    if (options.saturation_adc == 0
        || options.saturation_adc > static_cast<unsigned int>(std::numeric_limits<short>::max())) {
        throw std::runtime_error("--saturation-adc must be between 1 and 32767");
    }
    if (options.run.size() != 6 || !std::all_of(
        options.run.begin(), options.run.end(), [](unsigned char c) { return std::isdigit(c); }
    )) {
        throw std::runtime_error("--run must contain exactly six digits");
    }
    if (!options.timestamp.empty() && !std::all_of(
        options.timestamp.begin(), options.timestamp.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '_';
        }
    )) {
        throw std::runtime_error("--timestamp may contain only letters, digits, '-' and '_'");
    }
    options.input_list = fs::absolute(options.input_list);
    options.config = fs::absolute(options.config);
    options.output_dir = fs::absolute(options.output_dir);
    return options;
}

std::string executionTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&time, &local_time);
    std::ostringstream output;
    output << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    return output.str();
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
        const fs::path path = fs::weakly_canonical(input_list.parent_path() / line);
        if (!fs::is_regular_file(path)) {
            throw std::runtime_error("ROOT file not found: " + path.string());
        }
        files.push_back(path.string());
    }
    if (files.empty()) throw std::runtime_error("No ROOT files found in input list");
    return files;
}

template <typename Value>
TBranch *bindBranch(TTree &tree, const char *name, Value *address)
{
    TBranch *branch = nullptr;
    if (tree.GetBranch(name) == nullptr
        || tree.SetBranchAddress(name, address, &branch) < 0 || branch == nullptr) {
        throw std::runtime_error(std::string("Missing or incompatible ROOT branch: ") + name);
    }
    return branch;
}

ScanResult scanWaveforms(
    TChain &chain,
    const WaveformAnalyzer &analyzer,
    const fs::path &csv_file,
    const Options &options
)
{
    std::ofstream output(csv_file);
    if (!output) throw std::runtime_error("Could not create " + csv_file.string());
    output << "chain_entry,event,channel,waveform_index_save,timestamp,"
           << "saturated,peak_search_performed,peak_found,"
           << "peak_tick,peak_amplitude_adc,peak_count,primary_outside_signal_region,"
           << "additional_outside_signal_region,selected\n";
    output << std::setprecision(std::numeric_limits<double>::max_digits10);

    Int_t event = 0;
    UInt_t channel = 0;
    Int_t waveform_index = 0;
    ULong64_t timestamp = 0;
    std::vector<short> samples;
    std::vector<short> *adc = &samples;
    int tree_number = -1;
    std::vector<TBranch *> branches;
    ScanResult result;
    result.entries = chain.GetEntries();
    const auto report_progress = [&](Long64_t entry) {
        if (entry == 0 || (entry + 1) % 100000 == 0 || entry + 1 == result.entries) {
            std::cout << "Input: " << entry + 1 << '/' << result.entries
                      << " waveforms; analyzed=" << result.analyzed
                      << "; skipped by channel=" << result.skipped_channels
                      << "; selected=" << result.selected << std::endl;
        }
    };
    for (Long64_t entry = 0; entry < result.entries; ++entry) {
        const Long64_t local_entry = chain.LoadTree(entry);
        if (local_entry < 0) {
            throw std::runtime_error("Could not load ROOT entry " + std::to_string(entry));
        }
        if (chain.GetTreeNumber() != tree_number) {
            tree_number = chain.GetTreeNumber();
            TTree *tree = chain.GetTree();
            if (tree == nullptr) throw std::runtime_error("Missing WaveformTree in input file");
            branches = {
                bindBranch(*tree, "event", &event),
                bindBranch(*tree, "channel", &channel),
                bindBranch(*tree, "waveform_index", &waveform_index),
                bindBranch(*tree, "timestamp", &timestamp),
                bindBranch(*tree, "adc", &adc)
            };
            branches.back()->SetAutoDelete(kFALSE);
        }
        // Read the channel first, avoiding ADC I/O and interval lookup for
        // channels the user did not request.
        if (branches[1]->GetEntry(local_entry) <= 0) {
            throw std::runtime_error("Could not read ROOT channel at entry " + std::to_string(entry));
        }
        if (!options.channels.empty() && options.channels.count(channel) == 0) {
            ++result.skipped_channels;
            report_progress(entry);
            continue;
        }
        for (TBranch *branch : branches) {
            if (branch == branches[1]) continue;
            if (branch->GetEntry(local_entry) <= 0) {
                throw std::runtime_error("Could not read ROOT entry " + std::to_string(entry));
            }
        }
        if (adc == nullptr) throw std::runtime_error("Null ADC vector at entry " + std::to_string(entry));

        const bool saturated = std::any_of(adc->begin(), adc->end(), [&](short value) {
            return static_cast<int>(value) >= static_cast<int>(options.saturation_adc);
        });
        PeakSearchResult peak;
        try {
            // The shared implementation retains PEAK_SMOOTHING_WINDOW = 10.
            // Saturated ADC samples are rejected before peak finding.
            if (!saturated) peak = analyzer.findPeak(*adc, channel);
        } catch (const std::exception &error) {
            throw std::runtime_error(
                "Peak finder at entry " + std::to_string(entry) + ", channel "
                + std::to_string(channel) + ": " + error.what()
            );
        }
        const bool selected = !saturated && peak.found && !peak.primary_outside_signal_region
            && !peak.additional_outside_signal_region;
        auto &counts = result.counts[channel];
        ++result.analyzed;
        ++counts.analyzed;
        counts.saturated += saturated;
        counts.no_peak += !saturated && !peak.found;
        counts.primary_outside += peak.primary_outside_signal_region;
        counts.additional_outside += peak.additional_outside_signal_region;
        counts.selected += selected;
        if (selected) {
            ++result.selected;
            result.waveforms[channel].push_back({analyzer.noiseBaseline(*adc, channel), *adc});
        }
        output << entry << ',' << event << ',' << channel << ',' << waveform_index << ','
               << timestamp << ',' << saturated << ',' << !saturated << ','
               << peak.found << ',' << peak.tick << ','
               << peak.amplitude << ',' << peak.peak_count << ','
               << peak.primary_outside_signal_region << ','
               << peak.additional_outside_signal_region << ',' << selected << '\n';
        report_progress(entry);
    }
    chain.ResetBranchAddresses();
    output.close();
    if (!output) throw std::runtime_error("Failed while writing " + csv_file.string());
    return result;
}

void writeSummary(
    const fs::path &filename,
    const Options &options,
    const std::vector<std::string> &files,
    const ScanResult &result
)
{
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Could not create " + filename.string());
    output << "Peak finder without coincidence\n"
           << "Run: " << options.run << '\n'
           << "Analysis timestamp: " << options.timestamp << '\n'
           << "Input list: " << options.input_list << '\n'
           << "Interval configuration: " << options.config << '\n'
           << "Raw ADC saturation threshold (inclusive): " << options.saturation_adc << '\n'
           << "Requested channels:";
    if (options.channels.empty()) output << " all";
    else for (const auto channel : options.channels) output << ' ' << channel;
    output << '\n'
           << "Peak finder: WaveformAnalyzer::findPeak, PEAK_SMOOTHING_WINDOW = 10\n"
           << "Smoothing: inclusive +/-5 samples (11 interior samples), truncated at edges\n"
           << "Peak height threshold: max(500 ADC, 10 * noise population standard deviation)\n"
           << "Baseline: median of configured noise interval\n"
           << "Selection: !saturated && peak_found && !primary_outside_signal_region"
           << " && !additional_outside_signal_region\n"
           << "Saturation: any raw ADC sample >= threshold; skip peak search if saturated.\n"
           << "No coincidence requirement or other amplitude cuts.\n"
           << "One CSV row per entry in requested channels, including saturated waveforms;"
           << " no deduplication or CSV multiplicity.\n"
           << "Plots and waveform statistics use selected rows only.\n"
           << "No fprompt calculation or SiPM-type mapping is required.\n\n"
           << "Input ROOT files:\n";
    for (const auto &file : files) output << "  " << file << '\n';
    output << "\nInput tree entries: " << result.entries
           << "\nSkipped by channel filter: " << result.skipped_channels
           << "\nAnalyzed waveforms: " << result.analyzed
           << "\nSelected waveforms: " << result.selected << "\n\n"
           << "channel,analyzed,saturated,no_peak,primary_outside,additional_outside,selected\n";
    for (const auto &[channel, counts] : result.counts) {
        output << channel << ',' << counts.analyzed << ',' << counts.saturated << ',' << counts.no_peak << ','
               << counts.primary_outside << ',' << counts.additional_outside << ','
               << counts.selected << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("Failed while writing " + filename.string());
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        Options options = parseOptions(argc, argv);
        if (!fs::is_regular_file(options.config)) {
            throw std::runtime_error("Configuration file not found: " + options.config.string());
        }
        if (options.timestamp.empty()) options.timestamp = executionTimestamp();
        const auto files = readInputList(options.input_list);
        const WaveformAnalyzer analyzer(options.config.string());
        TChain chain("WaveformTree");
        for (const auto &file : files) {
            // Read each tree header now so a missing tree cannot be silently
            // omitted from an otherwise valid multi-file chain.
            if (chain.Add(file.c_str(), 0) == 0) {
                throw std::runtime_error("Could not add WaveformTree from " + file);
            }
        }
        const fs::path output_dir = options.output_dir / options.timestamp;
        fs::create_directories(output_dir);
        const std::string label = "peak_scan_run_" + options.run + "_" + options.timestamp;
        const fs::path csv_file = output_dir / (label + ".csv");
        std::cout << "Running peak finder without coincidence; channels:";
        if (options.channels.empty()) std::cout << " all";
        else for (const auto channel : options.channels) std::cout << ' ' << channel;
        std::cout << "\nRejecting raw ADC >= " << options.saturation_adc << '\n'
                  << "Output: " << csv_file << std::endl;
        const auto result = scanWaveforms(chain, analyzer, csv_file, options);
        writeSummary(output_dir / (label + ".txt"), options, files, result);
        if (result.selected == 0) {
            std::cout << "No waveforms passed peak selection; CSV and summary saved.\n";
            return 0;
        }
        std::set<unsigned int> channels;
        for (const auto &[channel, counts] : result.counts) {
            (void)counts;
            channels.insert(channel);
        }
        gROOT->SetBatch(kTRUE);
        gStyle->SetOptStat(0);
        waveform_plotting::plotAndSaveStatistics(
            channels, result.waveforms, options.run, label, output_dir,
            "Peak Selection (No Coincidence)", "peak_selection"
        );
        std::cout << "Saved peak results, plots and statistics in " << output_dir << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
