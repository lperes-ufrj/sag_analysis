#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Default selection parameters. Command-line options can override all three.
constexpr int DEFAULT_COINCIDENCE_WINDOW_TICKS = 20;
constexpr double DEFAULT_COINCIDENCE_THRESHOLD_ADC = 8000.0;
constexpr double DEFAULT_SAVE_THRESHOLD_ADC = 3000.0;
constexpr double MAX_WAVEFORM_AMPLITUDE = 16383.0;

// The first eight channels form the two sides of the coincidence test. The
// remaining four channels contain the waveforms written to the output tree.
const std::vector<int> all_channels = {
    2030, 2031, 2040, 2041, 2050, 2051,
    2060, 2061, 2070, 2071, 2080, 2081,
};

const std::vector<int> channels_for_coincidence_left = {
    all_channels.begin(), all_channels.begin() + 4
};
const std::vector<int> channels_for_coincidence_right = {
    all_channels.begin() + 4, all_channels.begin() + 8
};
const std::vector<int> channels_to_save_wf = {
    all_channels.end() - 4, all_channels.end()
};

struct Args {
    int coincidence_window_ticks = DEFAULT_COINCIDENCE_WINDOW_TICKS;
    double coincidence_threshold_adc = DEFAULT_COINCIDENCE_THRESHOLD_ADC;
    double save_threshold_adc = DEFAULT_SAVE_THRESHOLD_ADC;
    std::string output;
    std::vector<std::string> input;
};

// In-memory representation of one entry from the input WaveformTree. The
// amplitude and peak position are calculated once while loading the data.
struct WaveformRecord {
    Long64_t event = 0;
    int channel = 0;
    Long64_t waveform_index = 0;
    Long64_t timestamp = 0;
    std::vector<short> adc;
    double amplitude = 0.0;
    bool is_saturated = false;
    int peak_tick = 0;
};

// Map each (event, channel) pair to indices in the records vector. Keeping
// indices instead of waveform copies makes the coincidence searches cheaper.
using LookupKey = std::pair<Long64_t, int>;
using Lookup = std::map<LookupKey, std::vector<std::size_t>>;

std::string format_double(double value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

void print_usage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --coincidence-window-ticks, --window-ticks N\n"
        << "  --coincidence-threshold-adc, --threshold-adc-coincidence X\n"
        << "  --save-threshold-adc, --threshold-adc-save X\n"
        << "  -o, --output FILE\n"
        << "  -i, --input FILE [FILE ...]\n"
        << "  -h, --help\n";
}

std::filesystem::path repo_dir_from_program(const char *program)
{
    // Resolve the executable location so the default ROOT files can be found
    // whether this program is launched from the repository root or elsewhere.
    std::filesystem::path exe_path(program);
    if (exe_path.is_relative()) {
        exe_path = std::filesystem::current_path() / exe_path;
    }

    std::error_code ec;
    exe_path = std::filesystem::weakly_canonical(exe_path, ec);
    const auto script_dir = exe_path.parent_path();

    if (script_dir.filename() == "analysis") {
        return script_dir.parent_path();
    }
    return std::filesystem::current_path();
}

std::vector<std::string> default_inputs(const char *program)
{
    const auto repo_dir = repo_dir_from_program(program);
    const std::vector<std::string> names = {
        "np02vd_raw_run039510_0000_df-s04-d0_dw_0_20250919T123428_0kV_gallery.root",
        "np02vd_raw_run039510_0001_df-s04-d0_dw_0_20250919T123526_0kV_gallery.root",
        "np02vd_raw_run039510_0002_df-s04-d0_dw_0_20250919T123624_0kV_gallery.root",
        "np02vd_raw_run039510_0003_df-s04-d0_dw_0_20250919T123722_0kV_gallery.root",
        "np02vd_raw_run039510_0004_df-s04-d0_dw_0_20250919T123821_0kV_gallery.root",
        "np02vd_raw_run039510_0005_df-s04-d0_dw_0_20250919T123919_0kV_gallery.root",
    };

    std::vector<std::string> inputs;
    inputs.reserve(names.size());

    for (const auto &name : names) {
        inputs.push_back((repo_dir / name).string());
    }

    return inputs;
}

Args parse_args(int argc, char **argv)
{
    Args args;
    args.input = default_inputs(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];

        // Advance to the value following an option, reporting missing values
        // consistently for every command-line argument.
        auto require_value = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++i];
        };

        if (option == "-h" || option == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (option == "--coincidence-window-ticks" || option == "--window-ticks") {
            args.coincidence_window_ticks = std::stoi(require_value(option));
        } else if (option == "--coincidence-threshold-adc" ||
                   option == "--threshold-adc-coincidence") {
            args.coincidence_threshold_adc = std::stod(require_value(option));
        } else if (option == "--save-threshold-adc" || option == "--threshold-adc-save") {
            args.save_threshold_adc = std::stod(require_value(option));
        } else if (option == "-o" || option == "--output") {
            args.output = require_value(option);
        } else if (option == "-i" || option == "--input") {
            // Input accepts one or more file names, up to the next option.
            args.input.clear();
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) {
                args.input.emplace_back(argv[++i]);
            }
            if (args.input.empty()) {
                throw std::runtime_error(option + " requires at least one input file");
            }
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }

    // Reject invalid values before any potentially expensive ROOT I/O.
    if (args.coincidence_window_ticks < 0) {
        throw std::runtime_error("--coincidence-window-ticks must be >= 0");
    }
    if (args.coincidence_threshold_adc < 0.0) {
        throw std::runtime_error("--coincidence-threshold-adc must be >= 0");
    }
    if (args.save_threshold_adc < 0.0) {
        throw std::runtime_error("--save-threshold-adc must be >= 0");
    }

    if (args.output.empty()) {
        // Encode the selection settings in the default output name so scans
        // with different thresholds do not overwrite one another.
        args.output = "coincident_wfs_m7_m8_window_" +
                      std::to_string(args.coincidence_window_ticks) +
                      "_ticks_coinc_adc_" +
                      format_double(args.coincidence_threshold_adc) +
                      "_save_adc_" +
                      format_double(args.save_threshold_adc) +
                      ".root";
    }

    return args;
}

double signal_amplitude(const std::vector<short> &waveform)
{
    // Restrict the measurement to the expected signal region [30, 130). The
    // min/max bounds also make short waveforms safe to process.
    const std::size_t begin = std::min<std::size_t>(30, waveform.size());
    const std::size_t end = std::min<std::size_t>(200, waveform.size());
    if (begin >= end) {
        return 0.0;
    }

    const auto minmax = std::minmax_element(waveform.begin() + begin,
                                           waveform.begin() + end);
    return static_cast<double>(*minmax.second) - static_cast<double>(*minmax.first);
}

int signal_peak_tick(const std::vector<short> &waveform)
{
    // Use the maximum ADC sample in the signal region as the peak position.
    const std::size_t begin = std::min<std::size_t>(30, waveform.size());
    const std::size_t end = std::min<std::size_t>(200, waveform.size());
    if (begin >= end) {
        return static_cast<int>(begin);
    }

    const auto max_it = std::max_element(waveform.begin() + begin,
                                         waveform.begin() + end);
    return static_cast<int>(std::distance(waveform.begin(), max_it));
}

Lookup build_lookup(const std::vector<WaveformRecord> &records,
                    double threshold_adc)
{
    Lookup lookup;

    // Only waveforms above the requested amplitude threshold participate in
    // this lookup.
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (records[i].amplitude > threshold_adc) {
            lookup[{records[i].event, records[i].channel}].push_back(i);
        }
    }

    // peak_range() relies on every list being ordered by peak tick.
    for (auto &[key, indices] : lookup) {
        // The key is part of the structured binding for readability; sorting
        // only needs the associated record indices.
        (void)key;
        std::sort(indices.begin(), indices.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      return records[lhs].peak_tick < records[rhs].peak_tick;
                  });
    }

    return lookup;
}

std::pair<std::vector<std::size_t>::const_iterator,
          std::vector<std::size_t>::const_iterator>
peak_range(const std::vector<std::size_t> &indices,
           const std::vector<WaveformRecord> &records,
           int low_tick,
           int high_tick)
{
    // Binary searches return the half-open range of records whose peaks fall
    // within the inclusive interval [low_tick, high_tick].
    const auto lower = std::lower_bound(
        indices.begin(), indices.end(), low_tick,
        [&](std::size_t index, int tick) {
            return records[index].peak_tick < tick;
        });

    const auto upper = std::upper_bound(
        indices.begin(), indices.end(), high_tick,
        [&](int tick, std::size_t index) {
            return tick < records[index].peak_tick;
        });

    return {lower, upper};
}

void print_channels(const std::string &label, const std::vector<int> &channels)
{
    std::cout << label << ": [";
    for (std::size_t i = 0; i < channels.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << channels[i];
    }
    std::cout << "]\n";
}

} // namespace

int main(int argc, char **argv)
{
    // Parse and validate all command-line configuration first.
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    // TChain exposes several input ROOT files as one WaveformTree.
    TChain chain("WaveformTree");
    for (const auto &input : args.input) {
        chain.Add(input.c_str());
    }

    const Long64_t n_entries = chain.GetEntries();
    std::cout << "Input tree entries: " << n_entries << "\n";
    if (n_entries == 0) {
        std::cerr
            << "No entries found in WaveformTree. Check that the input ROOT files exist "
            << "and contain the expected tree.\n";
        return 1;
    }

    Int_t event = 0;
    Int_t waveform_index = 0;
    UInt_t channel = 0;
    ULong64_t timestamp = 0;
    std::vector<short> *adc = nullptr;

    // Disable unused branches to reduce the amount of data read from disk.
    chain.SetBranchStatus("*", 0);
    chain.SetBranchStatus("event", 1);
    chain.SetBranchStatus("channel", 1);
    chain.SetBranchStatus("waveform_index", 1);
    chain.SetBranchStatus("timestamp", 1);
    chain.SetBranchStatus("adc", 1);
    chain.SetBranchAddress("event", &event);
    chain.SetBranchAddress("channel", &channel);
    chain.SetBranchAddress("waveform_index", &waveform_index);
    chain.SetBranchAddress("timestamp", &timestamp);
    chain.SetBranchAddress("adc", &adc);

    // Ignore every input channel that cannot contribute to a coincidence or
    // appear in the saved output.
    std::set<int> valid_channels;
    valid_channels.insert(channels_for_coincidence_left.begin(),
                          channels_for_coincidence_left.end());
    valid_channels.insert(channels_for_coincidence_right.begin(),
                          channels_for_coincidence_right.end());
    valid_channels.insert(channels_to_save_wf.begin(), channels_to_save_wf.end());

    std::vector<WaveformRecord> records;
    std::set<Long64_t> events;

    // Load relevant waveforms and cache their derived quantities. Calculating
    // amplitude and peak tick here avoids repeating the work in nested loops.
    for (Long64_t entry = 0; entry < n_entries; ++entry) {
        chain.GetEntry(entry);
        if (!adc || valid_channels.count(static_cast<int>(channel)) == 0) {
            continue;
        }

        WaveformRecord record;
        record.event = event;
        record.channel = static_cast<int>(channel);
        record.waveform_index = waveform_index;
        record.timestamp = static_cast<Long64_t>(timestamp);
        record.adc = *adc;
        record.amplitude = signal_amplitude(record.adc);
        record.is_saturated = record.amplitude >= MAX_WAVEFORM_AMPLITUDE;
        record.peak_tick = signal_peak_tick(record.adc);

        events.insert(record.event);
        records.push_back(std::move(record));
    }

    // Report the effective selection configuration for reproducibility.
    std::cout << "Total waveforms collected: " << records.size() << "\n";
    print_channels("Coincidence-left channels", channels_for_coincidence_left);
    print_channels("Coincidence-right channels", channels_for_coincidence_right);
    print_channels("Saved waveform channels", channels_to_save_wf);
    std::cout << "Coincidence window: " << args.coincidence_window_ticks << " ticks\n";
    std::cout << "Coincidence amplitude threshold: "
              << format_double(args.coincidence_threshold_adc) << " ADC\n";
    std::cout << "Saved waveform amplitude threshold: "
              << format_double(args.save_threshold_adc) << " ADC\n";

    // Coincidence channels and saved channels use independent thresholds.
    const Lookup by_event_channel_coincidence =
        build_lookup(records, args.coincidence_threshold_adc);
    const Lookup by_event_channel_save =
        build_lookup(records, args.save_threshold_adc);

    TFile out_file(args.output.c_str(), "RECREATE");
    TTree out_tree("CoincidentWaveforms",
                   "Waveforms passing coincidence selection");

    // Variables below are connected directly to branches in the output tree.
    // Their values are replaced for each saved waveform before Fill() is called.
    Long64_t pair_id = 0;
    Long64_t event_out = 0;
    Int_t coincidence_left_channel = 0;
    Int_t coincidence_right_channel = 0;
    Long64_t coincidence_left_waveform_index = 0;
    Long64_t coincidence_right_waveform_index = 0;
    Bool_t coincidence_left_is_saturated = false;
    Bool_t coincidence_right_is_saturated = false;
    Long64_t coincidence_left_timestamp = 0;
    Long64_t coincidence_right_timestamp = 0;
    Long64_t dt_timestamp_ticks = 0;
    Int_t dt_peak_ticks_coinc = 0;
    Int_t saved_channel = 0;
    Long64_t saved_waveform_index = 0;
    Long64_t saved_timestamp = 0;
    Double_t saved_amplitude = 0.0;
    Bool_t saved_is_saturated = false;
    Int_t saved_peak_tick = 0;
    Int_t dt_peak_ticks_save = 0;
    std::vector<double> adc_vec;

    out_tree.Branch("pair_id", &pair_id, "pair_id/L");
    out_tree.Branch("event", &event_out, "event/L");
    out_tree.Branch("coincidence_left_channel", &coincidence_left_channel,
                    "coincidence_left_channel/I");
    out_tree.Branch("coincidence_right_channel", &coincidence_right_channel,
                    "coincidence_right_channel/I");
    out_tree.Branch("coincidence_left_waveform_index",
                    &coincidence_left_waveform_index,
                    "coincidence_left_waveform_index/L");
    out_tree.Branch("coincidence_right_waveform_index",
                    &coincidence_right_waveform_index,
                    "coincidence_right_waveform_index/L");
    out_tree.Branch("coincidence_left_is_saturated",
                    &coincidence_left_is_saturated,
                    "coincidence_left_is_saturated/O");
    out_tree.Branch("coincidence_right_is_saturated",
                    &coincidence_right_is_saturated,
                    "coincidence_right_is_saturated/O");
    out_tree.Branch("coincidence_left_timestamp", &coincidence_left_timestamp,
                    "coincidence_left_timestamp/L");
    out_tree.Branch("coincidence_right_timestamp", &coincidence_right_timestamp,
                    "coincidence_right_timestamp/L");
    out_tree.Branch("dt_timestamp_ticks", &dt_timestamp_ticks,
                    "dt_timestamp_ticks/L");
    out_tree.Branch("dt_peak_ticks_coinc", &dt_peak_ticks_coinc,
                    "dt_peak_ticks_coinc/I");
    out_tree.Branch("saved_channel", &saved_channel, "saved_channel/I");
    out_tree.Branch("saved_waveform_index", &saved_waveform_index,
                    "saved_waveform_index/L");
    out_tree.Branch("saved_timestamp", &saved_timestamp, "saved_timestamp/L");
    out_tree.Branch("saved_amplitude", &saved_amplitude, "saved_amplitude/D");
    out_tree.Branch("saved_is_saturated", &saved_is_saturated,
                    "saved_is_saturated/O");
    out_tree.Branch("saved_peak_tick", &saved_peak_tick, "saved_peak_tick/I");
    out_tree.Branch("dt_peak_ticks_save", &dt_peak_ticks_save,
                    "dt_peak_ticks_save/I");
    out_tree.Branch("adc", &adc_vec);

    Long64_t n_pairs = 0;
    Long64_t n_saved_waveforms = 0;

    // A saved waveform may be close to multiple coincidence pairs. Track its
    // record index so it is written only once across the complete output tree.
    std::set<std::size_t> saved_once;

    // Search each event independently, pairing every above-threshold left-side
    // waveform with right-side peaks inside the coincidence window.
    for (const Long64_t evt : events) {
        for (const int ch_left : channels_for_coincidence_left) {
            const auto left_it = by_event_channel_coincidence.find({evt, ch_left});
            if (left_it == by_event_channel_coincidence.end()) {
                continue;
            }

            for (const int ch_right : channels_for_coincidence_right) {
                const auto right_it =
                    by_event_channel_coincidence.find({evt, ch_right});
                if (right_it == by_event_channel_coincidence.end()) {
                    continue;
                }

                const auto &left_sorted = left_it->second;
                const auto &right_sorted = right_it->second;

                for (const std::size_t i_left : left_sorted) {
                    const int t_left = records[i_left].peak_tick;

                    // Because right_sorted is ordered by peak tick, this binary
                    // search avoids scanning every right-side waveform.
                    const auto [right_begin, right_end] = peak_range(
                        right_sorted, records,
                        t_left - args.coincidence_window_ticks,
                        t_left + args.coincidence_window_ticks);

                    for (auto right_iter = right_begin; right_iter != right_end;
                         ++right_iter) {
                        const std::size_t i_right = *right_iter;
                        const int t_right = records[i_right].peak_tick;

                        // Anchor the save window at the earlier of the two
                        // coincidence peaks.
                        const int reference_peak_tick = std::min(t_left, t_right);
                        const int save_window_min =
                            reference_peak_tick - args.coincidence_window_ticks;
                        const int save_window_max =
                            reference_peak_tick + args.coincidence_window_ticks;

                        pair_id = n_pairs;
                        event_out = evt;
                        coincidence_left_channel = ch_left;
                        coincidence_right_channel = ch_right;
                        coincidence_left_waveform_index =
                            records[i_left].waveform_index;
                        coincidence_right_waveform_index =
                            records[i_right].waveform_index;
                        coincidence_left_is_saturated =
                            records[i_left].is_saturated;
                        coincidence_right_is_saturated =
                            records[i_right].is_saturated;
                        coincidence_left_timestamp = records[i_left].timestamp;
                        coincidence_right_timestamp = records[i_right].timestamp;
                        dt_timestamp_ticks =
                            records[i_right].timestamp - records[i_left].timestamp;
                        dt_peak_ticks_coinc = t_right - t_left;

                        // Find save-channel waveforms that pass their threshold
                        // and peak within the same time window.
                        for (const int ch_save : channels_to_save_wf) {
                            const auto save_it =
                                by_event_channel_save.find({evt, ch_save});
                            if (save_it == by_event_channel_save.end()) {
                                continue;
                            }

                            const auto [save_begin, save_end] = peak_range(
                                save_it->second, records,
                                save_window_min, save_window_max);

                            for (auto save_iter = save_begin;
                                 save_iter != save_end; ++save_iter) {
                                // Do not duplicate a waveform if another pair
                                // has already selected it.
                                if (!saved_once.insert(*save_iter).second) {
                                    continue;
                                }

                                const auto &saved = records[*save_iter];
                                saved_channel = saved.channel;
                                saved_waveform_index = saved.waveform_index;
                                saved_timestamp = saved.timestamp;
                                saved_amplitude = saved.amplitude;
                                saved_is_saturated = saved.is_saturated;
                                saved_peak_tick = saved.peak_tick;
                                dt_peak_ticks_save =
                                    saved.peak_tick - reference_peak_tick;

                                // The output branch stores ADC samples as doubles,
                                // while the input waveform uses signed shorts.
                                adc_vec.clear();
                                adc_vec.reserve(saved.adc.size());
                                for (const short value : saved.adc) {
                                    adc_vec.push_back(static_cast<double>(value));
                                }

                                out_tree.Fill();
                                ++n_saved_waveforms;
                            }
                        }

                        ++n_pairs;
                    }
                }
            }
        }
    }

    // Persist the completed tree and print a compact processing summary.
    out_file.cd();
    out_tree.Write();
    out_file.Close();

    std::cout << "Saved " << n_saved_waveforms << " waveform entries\n";
    std::cout << "Saved " << n_pairs << " coincident pairs\n";
    std::cout << "Output file: " << args.output << "\n";

    return 0;
}
