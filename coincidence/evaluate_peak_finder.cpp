#include "coincidence_lib.hpp"

#include <TAxis.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
const std::string DEFAULT_CSV_SUFFIX =
    "coinc_2030-2031-2040-2041_vs_2050-2051-2060-2061_"
    "save_2070-2071-2080-2081_window_10_ticks_min_amplitude_0_adc.csv";

struct Options {
    std::string run;
    fs::path input_list;
    fs::path csv_file;
    fs::path config;
    fs::path output_dir;
    std::string csv_suffix = DEFAULT_CSV_SUFFIX;
    double max_auxiliary_amplitude = DEFAULT_MAX_AUXILIARY_AMPLITUDE;
    std::size_t max_plots = 0;
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

struct CutDecision {
    bool noise = false;
    bool post_signal = false;
    bool full_range = false;
    bool primary_peak = false;
    bool additional_peak = false;
    bool invalid_signal_peak_count = false;
    bool missing_primary_peak = false;

    bool retained() const
    {
        return !(noise || post_signal || full_range || primary_peak || additional_peak
            || invalid_signal_peak_count || missing_primary_peak);
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
    if (run.empty() || !std::all_of(run.begin(), run.end(), [](unsigned char character) {
            return std::isdigit(character) != 0;
        })) {
        throw std::runtime_error("Invalid run number: " + run);
    }
    if (run.size() < 6) run.insert(run.begin(), 6 - run.size(), '0');
    return run;
}

void printUsage(const char *program)
{
    std::cout
        << "Usage: " << program << " [options] RUN\n\n"
        << "Plot every CSV-selected waveform with the exact peak finder and cuts used\n"
        << "by plot_wfs_coincidence. Outputs are separated into retained/ and rejected/.\n\n"
        << "Options:\n"
        << "  --input-list FILE            ROOT input list (default: input_lists/input_runRUN.txt)\n"
        << "  --csv FILE                   Selection CSV (default: coincidence/waveforms_run_RUN_SUFFIX)\n"
        << "  --csv-suffix SUFFIX          Override the default selection CSV suffix\n"
        << "  --config FILE                Waveform interval INI file\n"
        << "  --output-dir DIR             Output directory\n"
        << "  --max-auxiliary-amplitude N  Absolute |ADC - baseline| limit in noise/"
        << "post-signal regions (default: 500)\n"
        << "  --max-plots N                Plot at most N waveforms; 0 means all (default: 0)\n"
        << "  -h, --help                    Show this help\n";
}

Options parseOptions(
    int argc,
    char **argv,
    const fs::path &program_dir,
    const fs::path &repo_dir
)
{
    Options options;
    options.config = program_dir / "waveform_intervals.ini";

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "-h" || argument == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (argument == "--input-list" || argument == "--csv"
            || argument == "--csv-suffix" || argument == "--config"
            || argument == "--output-dir" || argument == "--max-auxiliary-amplitude"
            || argument == "--max-plots") {
            if (++index >= argc) throw std::runtime_error("Missing value for " + argument);
            if (argument == "--input-list") options.input_list = argv[index];
            else if (argument == "--csv") options.csv_file = argv[index];
            else if (argument == "--csv-suffix") options.csv_suffix = argv[index];
            else if (argument == "--config") options.config = argv[index];
            else if (argument == "--output-dir") options.output_dir = argv[index];
            else if (argument == "--max-auxiliary-amplitude") {
                options.max_auxiliary_amplitude = std::stod(argv[index]);
            } else {
                options.max_plots = static_cast<std::size_t>(std::stoull(argv[index]));
            }
            continue;
        }
        if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + argument);
        }
        if (!options.run.empty()) throw std::runtime_error("Specify exactly one run");
        options.run = normalizeRun(argument);
    }

    if (options.run.empty()) throw std::runtime_error("Missing run number");
    if (options.input_list.empty()) {
        options.input_list = repo_dir / "input_lists" / ("input_run" + options.run + ".txt");
        if (!fs::is_regular_file(options.input_list)) {
            options.input_list = program_dir / ("input_run" + options.run + ".txt");
        }
    }
    if (options.csv_file.empty()) {
        options.csv_file = program_dir
            / ("waveforms_run_" + options.run + "_" + options.csv_suffix);
        if (!fs::is_regular_file(options.csv_file)) {
            const std::string prefix = "waveforms_run_" + options.run + "_";
            std::vector<fs::path> matches;
            for (const auto &entry : fs::directory_iterator(program_dir)) {
                const std::string name = entry.path().filename().string();
                if (entry.is_regular_file() && name.rfind(prefix, 0) == 0
                    && entry.path().extension() == ".csv") {
                    matches.push_back(entry.path());
                }
            }
            if (matches.size() == 1) options.csv_file = matches.front();
        }
    }
    if (options.output_dir.empty()) {
        options.output_dir = program_dir / "peak_finder_evaluation" / ("run_" + options.run);
    }
    options.input_list = fs::absolute(options.input_list);
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

std::vector<std::string> readRootFiles(const fs::path &input_list, const fs::path &repo_dir)
{
    std::ifstream input(input_list);
    if (!input) throw std::runtime_error("Could not open input list: " + input_list.string());

    std::vector<std::string> files;
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
        files.push_back(path.string());
    }
    if (files.empty()) throw std::runtime_error("No ROOT files found in " + input_list.string());
    return files;
}

std::unordered_map<WaveformKey, std::size_t, WaveformKeyHash> readSelectedKeys(
    const fs::path &csv_file,
    std::size_t &row_count
)
{
    std::ifstream input(csv_file);
    if (!input) throw std::runtime_error("Could not open selection CSV: " + csv_file.string());

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty CSV: " + csv_file.string());
    const auto header = splitCsv(line);
    const auto column = [&](const std::string &name) {
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
        const WaveformKey key{
            std::stoi(fields[event_column]),
            static_cast<unsigned int>(std::stoul(fields[channel_column])),
            std::stoi(fields[waveform_column]),
        };
        ++keys[key];
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
    for (Long64_t entry = 0; entry < chain.GetEntries(); ++entry) {
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
    // Keep this symmetric auxiliary-region cut identical to plot_wfs_coincidence.
    return std::max(
        std::abs(static_cast<double>(*minimum) - baseline),
        std::abs(static_cast<double>(*maximum) - baseline)
    );
}

CutDecision applyPlotCuts(
    const std::vector<short> &adc,
    unsigned int channel,
    const WaveformAnalyzer &analyzer,
    const PeakSearchResult &peak,
    double max_auxiliary_amplitude
)
{
    CutDecision result;
    result.noise = maximumAbsoluteAmplitudeInRegion(
        adc, peak.baseline, analyzer, channel, "noise"
    ) > max_auxiliary_amplitude;
    result.post_signal = maximumAbsoluteAmplitudeInRegion(
        adc, peak.baseline, analyzer, channel, "post_signal"
    ) > max_auxiliary_amplitude;
    result.full_range = static_cast<double>(*std::max_element(adc.begin(), adc.end()))
        - peak.baseline > MAX_FULL_RANGE_AMPLITUDE;
    result.primary_peak = peak.primary_outside_signal_region;
    result.additional_peak = peak.additional_outside_signal_region;
    result.invalid_signal_peak_count = peak.signal_peak_count != 1;
    result.missing_primary_peak = !peak.found;
    return result;
}

std::string rejectionReasons(const CutDecision &decision)
{
    std::vector<std::string> reasons;
    if (decision.noise) reasons.emplace_back("noise |ADC - baseline|");
    if (decision.post_signal) reasons.emplace_back("post-signal |ADC - baseline|");
    if (decision.full_range) reasons.emplace_back("full-range amplitude");
    if (decision.primary_peak) reasons.emplace_back("primary peak outside signal");
    if (decision.additional_peak) reasons.emplace_back("additional peak outside signal");
    if (decision.invalid_signal_peak_count) {
        reasons.emplace_back("signal region does not contain exactly one peak");
    }
    if (decision.missing_primary_peak) reasons.emplace_back("no primary peak found");
    if (reasons.empty()) return "none";
    std::string text = reasons.front();
    for (std::size_t index = 1; index < reasons.size(); ++index) text += "; " + reasons[index];
    return text;
}

std::size_t significantAdditionalCount(const PeakSearchResult &peak)
{
    return static_cast<std::size_t>(std::count_if(
        peak.candidates.begin(), peak.candidates.end(),
        [](const PeakCandidate &candidate) { return candidate.significant_additional; }
    ));
}

std::string candidateSummary(const PeakSearchResult &peak)
{
    std::ostringstream output;
    output << std::setprecision(8);
    bool first = true;
    for (const auto &candidate : peak.candidates) {
        if (!first) output << ';';
        first = false;
        const double relative = peak.amplitude == 0.0
            ? 0.0 : candidate.amplitude / peak.amplitude;
        output << "tick=" << candidate.tick
               << "|amplitude=" << candidate.amplitude
               << "|prominence=" << candidate.prominence
               << "|relative=" << relative
               << "|additional_veto=" << candidate.significant_additional;
    }
    return output.str();
}

fs::path plotWaveform(
    const std::vector<short> &adc,
    const LocatedWaveform &item,
    const PeakSearchResult &peak,
    const CutDecision &decision,
    const WaveformAnalyzer &analyzer,
    const Options &options
)
{
    const int count = static_cast<int>(adc.size());
    TGraph raw(count);
    TGraph smoothed(count);
    double y_min = std::numeric_limits<double>::max();
    double y_max = std::numeric_limits<double>::lowest();
    for (int tick = 0; tick < count; ++tick) {
        const double value = static_cast<double>(adc[static_cast<std::size_t>(tick)])
            - peak.baseline;
        raw.SetPoint(tick, tick, value);
        smoothed.SetPoint(tick, tick, peak.smoothed[static_cast<std::size_t>(tick)]);
        y_min = std::min({y_min, value, peak.smoothed[static_cast<std::size_t>(tick)]});
        y_max = std::max({y_max, value, peak.smoothed[static_cast<std::size_t>(tick)]});
    }
    y_min = std::min(y_min, 0.0);
    y_max = std::max(y_max, peak.minimum_height);
    const double padding = std::max(20.0, 0.08 * (y_max - y_min));
    y_min -= padding;
    y_max += padding;

    auto [signal_start, signal_stop] = analyzer.interval(item.key.channel, "signal");
    signal_start = std::max(0, std::min(signal_start, count));
    signal_stop = std::max(signal_start, std::min(signal_stop, count));

    TGraph inside_candidates;
    TGraph significant_outside_candidates;
    TGraph ignored_outside_candidates;
    for (const auto &candidate : peak.candidates) {
        TGraph *graph = &inside_candidates;
        if (candidate.tick < signal_start || candidate.tick >= signal_stop) {
            graph = candidate.significant_additional
                ? &significant_outside_candidates : &ignored_outside_candidates;
        }
        graph->SetPoint(graph->GetN(), candidate.tick, candidate.amplitude);
    }
    TGraph primary;
    if (peak.found) primary.SetPoint(0, peak.tick, peak.amplitude);

    const std::string status = decision.retained() ? "RETAINED" : "REJECTED";
    const std::string object_suffix = std::to_string(item.key.event) + "_"
        + std::to_string(item.key.channel) + "_" + std::to_string(item.key.waveform_index);
    TCanvas canvas(("peak_evaluation_" + object_suffix).c_str(), "", 1800, 850);
    canvas.SetLeftMargin(0.09);
    canvas.SetRightMargin(0.04);
    canvas.SetBottomMargin(0.11);
    canvas.SetTopMargin(0.10);
    canvas.SetGrid();

    raw.SetTitle(
        ("Run " + options.run + ", event " + std::to_string(item.key.event)
            + ", channel " + std::to_string(item.key.channel)
            + ", waveform " + std::to_string(item.key.waveform_index)
            + " - " + status + ";Tick;ADC - baseline").c_str()
    );
    raw.SetLineColor(kBlue + 1);
    raw.SetLineWidth(1);
    raw.GetYaxis()->SetRangeUser(y_min, y_max);
    raw.Draw("AL");

    TLine zero(0.0, 0.0, count - 1.0, 0.0);
    zero.SetLineColor(kGray + 1);
    zero.SetLineStyle(3);
    zero.Draw("SAME");
    TLine threshold(0.0, peak.minimum_height, count - 1.0, peak.minimum_height);
    threshold.SetLineColor(kGray + 2);
    threshold.SetLineStyle(2);
    threshold.SetLineWidth(2);
    threshold.Draw("SAME");
    TLine signal_begin(signal_start, y_min, signal_start, y_max);
    TLine signal_end(signal_stop, y_min, signal_stop, y_max);
    signal_begin.SetLineColor(kGreen + 2);
    signal_end.SetLineColor(kGreen + 2);
    signal_begin.SetLineStyle(2);
    signal_end.SetLineStyle(2);
    signal_begin.SetLineWidth(2);
    signal_end.SetLineWidth(2);
    signal_begin.Draw("SAME");
    signal_end.Draw("SAME");

    smoothed.SetLineColor(kOrange + 7);
    smoothed.SetLineWidth(3);
    smoothed.Draw("L SAME");
    inside_candidates.SetMarkerColor(kGreen + 2);
    inside_candidates.SetMarkerStyle(20);
    inside_candidates.SetMarkerSize(1.4);
    if (inside_candidates.GetN() > 0) inside_candidates.Draw("P SAME");
    significant_outside_candidates.SetMarkerColor(kRed + 1);
    significant_outside_candidates.SetMarkerStyle(21);
    significant_outside_candidates.SetMarkerSize(1.4);
    if (significant_outside_candidates.GetN() > 0) {
        significant_outside_candidates.Draw("P SAME");
    }
    ignored_outside_candidates.SetMarkerColor(kCyan + 2);
    ignored_outside_candidates.SetMarkerStyle(25);
    ignored_outside_candidates.SetMarkerSize(1.5);
    if (ignored_outside_candidates.GetN() > 0) ignored_outside_candidates.Draw("P SAME");
    primary.SetMarkerColor(kMagenta + 2);
    primary.SetMarkerStyle(29);
    primary.SetMarkerSize(2.2);
    if (peak.found) primary.Draw("P SAME");

    TLegend legend(0.68, 0.65, 0.95, 0.89);
    legend.SetFillColorAlpha(kWhite, 0.88);
    legend.AddEntry(&raw, "Raw waveform - baseline", "l");
    legend.AddEntry(&smoothed, "Smoothed waveform", "l");
    legend.AddEntry(&threshold, "Peak threshold", "l");
    legend.AddEntry(&signal_begin, "Signal boundaries", "l");
    if (inside_candidates.GetN() > 0) legend.AddEntry(&inside_candidates, "Candidate inside signal", "p");
    if (significant_outside_candidates.GetN() > 0) {
        legend.AddEntry(&significant_outside_candidates, "Veto candidate outside signal", "p");
    }
    if (ignored_outside_candidates.GetN() > 0) {
        legend.AddEntry(&ignored_outside_candidates, "Ignored candidate outside signal", "p");
    }
    if (peak.found) legend.AddEntry(&primary, "Primary candidate", "p");
    legend.Draw();

    TPaveText information(0.58, 0.35, 0.95, 0.63, "NDC");
    information.SetFillColorAlpha(kWhite, 0.88);
    information.SetTextAlign(12);
    information.AddText(("baseline = " + std::to_string(peak.baseline)
        + ", noise RMS = " + std::to_string(peak.noise_rms)).c_str());
    information.AddText(("threshold = " + std::to_string(peak.minimum_height)
        + " ADC, candidates = " + std::to_string(peak.peak_count)
        + ", inside signal = " + std::to_string(peak.signal_peak_count)
        + ", significant additional = "
        + std::to_string(significantAdditionalCount(peak))).c_str());
    information.AddText(("additional prominence threshold = "
        + std::to_string(peak.additional_prominence_threshold) + " ADC").c_str());
    if (peak.found) {
        information.AddText(("primary tick = " + std::to_string(peak.tick)
            + ", amplitude = " + std::to_string(peak.amplitude) + " ADC").c_str());
    } else {
        information.AddText("primary peak: none");
    }
    std::size_t described = 0;
    for (const auto &candidate : peak.candidates) {
        if (candidate.tick == peak.tick || described == 2) continue;
        std::ostringstream description;
        description << std::fixed << std::setprecision(1)
                    << "additional tick " << candidate.tick
                    << ": amplitude " << candidate.amplitude
                    << ", prominence " << candidate.prominence
                    << ", relative " << 100.0 * candidate.amplitude / peak.amplitude << "% - "
                    << (candidate.significant_additional ? "VETO" : "ignored");
        information.AddText(description.str().c_str());
        ++described;
    }
    information.AddText(("plot_wfs_coincidence: " + status).c_str());
    information.AddText(("reason: " + rejectionReasons(decision)).c_str());
    information.Draw();

    const fs::path category = options.output_dir
        / (decision.retained() ? "retained" : "rejected");
    fs::create_directories(category);
    const fs::path output = category
        / ("event_" + std::to_string(item.key.event)
            + "_channel_" + std::to_string(item.key.channel)
            + "_waveform_" + std::to_string(item.key.waveform_index) + ".png");
    canvas.SaveAs(output.string().c_str());
    return output;
}

void evaluate(const Options &options, const fs::path &repo_dir)
{
    if (!fs::is_regular_file(options.input_list)) {
        throw std::runtime_error("Input list not found: " + options.input_list.string());
    }
    if (!fs::is_regular_file(options.csv_file)) {
        throw std::runtime_error("Selection CSV not found: " + options.csv_file.string());
    }
    if (!fs::is_regular_file(options.config)) {
        throw std::runtime_error("Configuration file not found: " + options.config.string());
    }

    std::size_t selected_rows = 0;
    const auto selected = readSelectedKeys(options.csv_file, selected_rows);
    auto chain = createChain(readRootFiles(options.input_list, repo_dir));
    const auto located = locateSelectedEntries(*chain, selected);
    const WaveformAnalyzer analyzer(options.config.string());

    fs::create_directories(options.output_dir);
    const fs::path summary_path = options.output_dir / "peak_finder_summary.csv";
    std::ofstream summary(summary_path);
    if (!summary) throw std::runtime_error("Could not create " + summary_path.string());
    summary << "event,channel,waveform_index,entry,multiplicity,baseline,noise_rms,"
            << "minimum_height,peak_count,signal_peak_count,significant_additional_count,"
            << "primary_tick,primary_amplitude,"
            << "primary_found,primary_outside_signal,additional_outside_signal,invalid_signal_peak_count,"
            << "retained,rejection_reasons,"
            << "candidates,plot\n";
    summary << std::setprecision(std::numeric_limits<double>::max_digits10);

    chain->ResetBranchAddresses();
    chain->SetBranchStatus("*", 0);
    chain->SetBranchStatus("adc", 1);
    std::vector<short> *adc = nullptr;
    chain->SetBranchAddress("adc", &adc);

    std::size_t retained = 0;
    std::size_t rejected = 0;
    std::size_t plotted = 0;
    for (const auto &item : located) {
        if (chain->GetEntry(item.entry) <= 0 || adc == nullptr || adc->empty()) {
            throw std::runtime_error("Could not read ROOT ADC entry " + std::to_string(item.entry));
        }
        const PeakSearchResult peak = analyzer.findPeak(*adc, item.key.channel);
        const CutDecision decision = applyPlotCuts(
            *adc, item.key.channel, analyzer, peak, options.max_auxiliary_amplitude
        );
        retained += item.multiplicity * static_cast<std::size_t>(decision.retained());
        rejected += item.multiplicity * static_cast<std::size_t>(!decision.retained());

        fs::path plot_path;
        if (options.max_plots == 0 || plotted < options.max_plots) {
            plot_path = plotWaveform(*adc, item, peak, decision, analyzer, options);
            ++plotted;
        }
        summary << item.key.event << ',' << item.key.channel << ',' << item.key.waveform_index
                << ',' << item.entry << ',' << item.multiplicity << ',' << peak.baseline
                << ',' << peak.noise_rms
                << ',' << peak.minimum_height << ',' << peak.peak_count
                << ',' << peak.signal_peak_count
                << ',' << significantAdditionalCount(peak)
                << ',' << peak.tick
                << ',' << peak.amplitude << ',' << peak.found
                << ',' << peak.primary_outside_signal_region
                << ',' << peak.additional_outside_signal_region
                << ',' << decision.invalid_signal_peak_count << ',' << decision.retained()
                << ",\"" << rejectionReasons(decision) << "\",\""
                << candidateSummary(peak) << "\",\""
                << plot_path.string() << "\"\n";
    }

    std::cout << "Run " << options.run << ": " << selected_rows << " selected CSV rows, "
              << located.size() << " unique waveforms\n"
              << "plot_wfs_coincidence cuts: " << retained << " retained, "
              << rejected << " rejected\n"
              << "Saved " << plotted << " waveform plots under " << options.output_dir << '\n'
              << "Saved " << summary_path << '\n';
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
        const Options options = parseOptions(argc, argv, program_dir, repo_dir);
        gROOT->SetBatch(kTRUE);
        gStyle->SetOptStat(0);
        evaluate(options, repo_dir);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
