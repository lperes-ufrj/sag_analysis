#include "coincidence_lib.hpp"
#include "waveform_plotting.hpp"

#include <TChain.h>
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

using waveform_plotting::RecordsByChannel;
using waveform_plotting::plotAllWaveformsForSelection;
using waveform_plotting::plotAndSaveStatistics;

constexpr double DEFAULT_MAX_AUXILIARY_AMPLITUDE = 500.0;
constexpr double MAX_FULL_RANGE_AMPLITUDE = 10000.0;

struct Options {
    fs::path input_list;
    fs::path config;
    fs::path output_dir;
    fs::path csv_file;
    double max_auxiliary_amplitude = DEFAULT_MAX_AUXILIARY_AMPLITUDE;
};

struct CoincidenceOutputIdentity {
    std::string run;
    std::string timestamp;
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

struct FPromptRecord {
    WaveformKey key;
    double value = 0.0;
};

using FPromptRecordsByChannel =
    std::map<unsigned int, std::vector<FPromptRecord>>;
struct CutStatistics {
    std::size_t from_coincidence = 0;
    std::size_t rejected_noise = 0;
    std::size_t rejected_post_signal = 0;
    std::size_t rejected_full_range_amplitude = 0;
    std::size_t rejected_primary_peak_outside_signal = 0;
    std::size_t rejected_additional_peak_outside_signal = 0;
    //std::size_t rejected_signal_peak_count = 0;
    //std::size_t rejected_missing_primary_peak = 0;
    std::size_t rejected_combined = 0;
    std::size_t final_selection = 0;
};

struct WaveformSelections {
    RecordsByChannel from_coincidence;
    RecordsByChannel final_selection;
    FPromptRecordsByChannel final_fprompt;
    CutStatistics cuts;
};

// Compatibility contract: these are exactly the five rejection flags used by
// coin_1p0v_code/plot_wfs_coincidence.cpp.  Plotting and reporting additions
// must consume this decision; they must not add acceptance criteria of their own.
struct Coin1p0vSelectionDecision {
    bool fails_noise = false;
    bool fails_post_signal = false;
    bool fails_full_range_amplitude = false;
    bool fails_primary_peak = false;
    bool fails_additional_peak = false;

    bool rejected() const
    {
        return fails_noise || fails_post_signal || fails_full_range_amplitude
            || fails_primary_peak || fails_additional_peak;
    }
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
        << "Usage: " << program
        << " --csv COINCIDENCE_CSV [options] INPUT_LIST.txt\n\n"
        << "Plot waveforms selected by run_coincidence. COINCIDENCE_CSV must be a\n"
        << "run_coincidence output named\n"
        << "coincidence_scan_run_RUN_TIMESTAMP.csv. INPUT_LIST supplies the ROOT\n"
        << "files containing the waveform samples referenced by that CSV.\n\n"
        << "Options:\n"
        << "  --config FILE                 Waveform interval INI file\n"
        << "  --output-dir DIR             Output directory\n"
        << "  --csv FILE                   run_coincidence output CSV (required)\n"
        << "  --max-auxiliary-amplitude N  Noise/post-signal limit in ADC (default: 500)\n"
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
            || argument == "--max-auxiliary-amplitude") {
            if (++index >= argc) throw std::runtime_error("Missing value for " + argument);
            if (argument == "--config") options.config = argv[index];
            else if (argument == "--output-dir") options.output_dir = argv[index];
            else if (argument == "--csv") options.csv_file = argv[index];
            else options.max_auxiliary_amplitude = std::stod(argv[index]);
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        }
        if (!options.input_list.empty()) {
            throw std::runtime_error("Unexpected positional argument: " + argument);
        }
        options.input_list = argument;
    }

    if (options.csv_file.empty()) {
        throw std::runtime_error("Missing required option --csv");
    }
    if (options.input_list.empty()) {
        throw std::runtime_error("Missing input list");
    }

    const fs::path command_line_path = fs::absolute(options.input_list);
    const fs::path program_relative_path = program_dir / options.input_list;
    options.input_list = fs::exists(command_line_path)
        ? command_line_path
        : program_relative_path;
    options.csv_file = fs::absolute(options.csv_file);
    options.config = fs::absolute(options.config);
    options.output_dir = fs::absolute(options.output_dir);
    if (!std::isfinite(options.max_auxiliary_amplitude)
        || options.max_auxiliary_amplitude < 0.0) {
        throw std::runtime_error(
            "Maximum auxiliary amplitude must be a finite, non-negative value"
        );
    }
    return options;
}

CoincidenceOutputIdentity identifyCoincidenceOutput(const fs::path &csv_file)
{
    // run_coincidence is the sole producer accepted here. Besides documenting
    // the data-flow contract, checking its filename prevents a CSV for one run
    // from being paired accidentally with another run's ROOT input list.
    const std::regex filename_pattern(
        R"(^coincidence_scan_run_([0-9]{6})_([A-Za-z0-9_-]+)\.csv$)"
    );
    std::smatch match;
    const std::string filename = csv_file.filename().string();
    if (!std::regex_match(filename, match, filename_pattern)) {
        throw std::runtime_error(
            "Expected a run_coincidence CSV named "
            "coincidence_scan_run_RUN_TIMESTAMP.csv, got: " + filename
        );
    }
    return {match[1].str(), match[2].str()};
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
    const WaveformAnalyzer &analyzer,
    unsigned int channel,
    const std::string &name,
    double baseline
)
{
    auto [start, stop] = analyzer.interval(channel, name);
    start = std::max(0, std::min(start, static_cast<int>(adc.size())));
    stop = std::max(start, std::min(stop, static_cast<int>(adc.size())));
    if (start == stop) throw std::runtime_error("Empty " + name + " interval");

    double amplitude = 0.0;
    for (int i = start; i < stop; ++i) {
        amplitude = std::max(
            amplitude,
            std::abs(static_cast<double>(adc[i]) - baseline)
        );
    }
    return amplitude;
}

double maximumInRegion(
    const std::vector<short> &adc,
    const WaveformAnalyzer &analyzer,
    unsigned int channel,
    const std::string &name
)
{
    auto [start, stop] = analyzer.interval(channel, name);
    start = std::max(0, std::min(start, static_cast<int>(adc.size())));
    stop = std::max(start, std::min(stop, static_cast<int>(adc.size())));
    if (start == stop) throw std::runtime_error("Empty " + name + " interval");
    return *std::max_element(adc.begin() + start, adc.begin() + stop);
}

double fullRangeAmplitude(const std::vector<short> &adc, double baseline)
{
    if (adc.empty()) {
        throw std::runtime_error("Cannot calculate amplitude from an empty waveform");
    }
    return static_cast<double>(*std::max_element(adc.begin(), adc.end())) - baseline;
}

Coin1p0vSelectionDecision evaluateCoin1p0vSelection(
    const std::vector<short> &adc,
    unsigned int channel,
    double baseline,
    const WaveformAnalyzer &analyzer,
    double max_auxiliary_amplitude
)
{
    // Keep these calculations and strict boundaries synchronized with the old
    // coin_1p0v_code selection.  In particular, these are positive maxima
    // above the noise median (not absolute excursions), and equality passes.
    const double noise_amplitude = maximumInRegion(
        adc, analyzer, channel, "noise"
    ) - baseline;
    const double post_signal_amplitude = maximumAbsoluteAmplitudeInRegion(
    adc, analyzer, channel, "post_signal", baseline
    );
    const bool fails_noise = noise_amplitude > max_auxiliary_amplitude;
    const bool fails_post_signal = post_signal_amplitude > max_auxiliary_amplitude;
    const double amplitude = fullRangeAmplitude(adc, baseline);
    const bool fails_full_range_amplitude = amplitude > MAX_FULL_RANGE_AMPLITUDE;
    const PeakSearchResult peak = analyzer.findPeak(adc, channel);
    const bool fails_primary_peak = peak.primary_outside_signal_region;
    const bool fails_additional_peak = peak.additional_outside_signal_region;

    return {
        fails_noise,
        fails_post_signal,
        fails_full_range_amplitude,
        fails_primary_peak,
        fails_additional_peak
    };
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
        const Coin1p0vSelectionDecision decision = evaluateCoin1p0vSelection(
            *adc,
            item.key.channel,
            baseline,
            analyzer,
            max_auxiliary_amplitude
        );

        rejected_noise += item.multiplicity
            * static_cast<std::size_t>(decision.fails_noise);
        rejected_post_signal += item.multiplicity
            * static_cast<std::size_t>(decision.fails_post_signal);
        rejected_full_range_amplitude += item.multiplicity
            * static_cast<std::size_t>(decision.fails_full_range_amplitude);
        rejected_primary_peak_outside_signal += item.multiplicity
            * static_cast<std::size_t>(decision.fails_primary_peak);
        rejected_additional_peak_outside_signal += item.multiplicity
            * static_cast<std::size_t>(decision.fails_additional_peak);
        total += item.multiplicity;
        if (decision.rejected()) {
            rejected_total += item.multiplicity;
            continue;
        }

        // Calculate f_prompt only after the existing five cuts have accepted
        // the waveform. This keeps the selection identical and guarantees that
        // the exported rows describe exactly the final-selection population.
        const double fprompt_value = analyzer.fprompt(*adc, item.key.channel);
        for (std::size_t copy = 0; copy < item.multiplicity; ++copy) {
            selections.final_selection[item.key.channel].push_back({baseline, *adc});
            selections.final_fprompt[item.key.channel].push_back({
                item.key,
                fprompt_value
            });
        }
    }

    std::cout
        << "Retained " << total - rejected_total << " of " << total
        << " selected waveforms\n"
        << "Rejected " << rejected_total
        << " waveforms in total (cut categories may overlap):\n"
        << "  Noise-region amplitude above " << max_auxiliary_amplitude
        << " ADC: " << rejected_noise << '\n'
        << "  Post-signal amplitude above " << max_auxiliary_amplitude
        << " ADC: " << rejected_post_signal << '\n'
        << "  Full-range amplitude above " << MAX_FULL_RANGE_AMPLITUDE
        << " ADC over baseline: " << rejected_full_range_amplitude << '\n'
        << "  Primary peak outside signal region: "
        << rejected_primary_peak_outside_signal << '\n'
        << "  Primary peak inside with another peak outside signal region: "
        << rejected_additional_peak_outside_signal << '\n';
    selections.cuts  = {
        total,
        rejected_noise,
        rejected_post_signal,
        rejected_full_range_amplitude,
        rejected_primary_peak_outside_signal,
        rejected_additional_peak_outside_signal,
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
           << "Compatibility: coin_1p0v_code selection criteria and boundaries\n"
           << "1. Noise-region maximum ADC minus noise median <= "
           << max_auxiliary_amplitude << " ADC\n"
           << "2. Post-signal maximum ADC minus noise median <= "
           << max_auxiliary_amplitude << " ADC\n"
           << "3. Full-waveform positive amplitude above baseline <= "
           << MAX_FULL_RANGE_AMPLITUDE << " ADC\n"
           << "4. Primary peak must be inside the configured signal region\n"
           << "5. No significant additional peak may be outside the signal region\n"
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
  
    write_counts("All cuts combined", cuts.rejected_combined);

    output << "\nFinal selected waveforms: " << cuts.final_selection << '\n'
           << "Note: individual rejection categories can overlap; therefore their "
           << "rejected counts should not be summed.\n";

    if (!output) {
        throw std::runtime_error("Failed while writing cut summary: " + filename.string());
    }
    std::cout << "Saved " << filename << '\n';
}


void writeFpromptCsvs(
    const FPromptRecordsByChannel &records_by_channel,
    const std::string &run,
    const std::string &label,
    const fs::path &output_dir,
    std::size_t expected_total
)
{
    std::size_t total = 0;
    for (const auto &[channel, records] : records_by_channel) {
        total += records.size();
        const fs::path filename = output_dir
            / ("fprompt_" + std::to_string(channel) + "_" + label + ".csv");
        std::ofstream output(filename);
        if (!output) {
            throw std::runtime_error("Could not create " + filename.string());
        }

        // The run/event/channel/index columns make every value traceable back
        // to the run_coincidence row while `fprompt` can be histogrammed
        // directly for this run/channel file.
        output << "run,event,channel,waveform_index_save,fprompt\n";
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto &record : records) {
            if (record.key.channel != channel) {
                throw std::runtime_error("Internal channel mismatch in fprompt records");
            }
            output << run << ','
                   << record.key.event << ','
                   << record.key.channel << ','
                   << record.key.waveform_index << ','
                   << record.value << '\n';
        }
        if (!output) {
            throw std::runtime_error("Failed while writing " + filename.string());
        }
        std::cout << "Saved " << records.size() << " fprompt values to "
                  << filename << '\n';
    }

    if (total != expected_total) {
        throw std::runtime_error(
            "Internal fprompt row-count mismatch: expected "
            + std::to_string(expected_total) + ", prepared "
            + std::to_string(total)
        );
    }
}


void processInputList(
    const fs::path &input_list,
    const fs::path &repo_dir,
    const Options &options
)
{
    if (!fs::is_regular_file(input_list)) {
        throw std::runtime_error("Input list not found: " + input_list.string());
    }
    const std::string run = inferRun(input_list);
    const fs::path &csv_file = options.csv_file;
    if (!fs::is_regular_file(csv_file)) {
        throw std::runtime_error("Coincidence CSV not found: " + csv_file.string());
    }
    const CoincidenceOutputIdentity identity = identifyCoincidenceOutput(csv_file);
    if (identity.run != run) {
        throw std::runtime_error(
            "Run mismatch: coincidence CSV is for run " + identity.run
            + " but input list resolves to run " + run
        );
    }

    std::cout << "\nProcessing run " << run
              << " from analysis " << identity.timestamp << '\n';
    std::set<unsigned int> channels;
    std::size_t selected_rows = 0;
    const auto selected = readSelectedKeys(csv_file, channels, selected_rows);
    std::cout << "Selected CSV rows: " << selected_rows << "\nSelected channels:";
    for (const auto channel : channels) std::cout << ' ' << channel;
    std::cout << '\n';
    if (selected.empty()) {
        std::cout << "No waveforms were selected by run_coincidence; "
                     "nothing to plot for run "
                  << run << '\n';
        return;
    }

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
    const std::string label = csv_file.stem().string();
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

    writeFpromptCsvs(
        selections.final_fprompt,
        run,
        label,
        options.output_dir,
        selections.cuts.final_selection
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
        processInputList(options.input_list, repo_dir, options);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
