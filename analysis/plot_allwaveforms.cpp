#include <TCanvas.h>
#include <TChain.h>
#include <TColor.h>
#include <TH2I.h>
#include <TLatex.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int ADC_BINS = 100;
constexpr double ADC_MIN = 0.0;
constexpr double ADC_MAX = 16384.0;

constexpr std::array<unsigned int, 16> CATHODE_CHANNELS = {
    1010, 1011, 1020, 1021, 1030, 1031, 1040, 1041,
    1050, 1051, 1060, 1061, 1070, 1071, 1080, 1081
};
constexpr std::array<unsigned int, 16> MEMBRANE_CHANNELS = {
    2010, 2011, 2020, 2021, 2030, 2031, 2040, 2041,
    2050, 2051, 2060, 2061, 2070, 2071, 2080, 2081
};

struct InputData {
    std::string run;
    std::vector<fs::path> files;
};

struct DensityData {
    std::size_t samples = 0;
    std::unordered_map<unsigned int, std::vector<std::uint32_t>> bins;
    std::unordered_map<unsigned int, std::uint64_t> counts;
};

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

fs::path canonicalFile(const fs::path &path)
{
    std::error_code error;
    const fs::path result = fs::canonical(path, error);
    if (error || !fs::is_regular_file(result)) return {};
    return result;
}

InputData readInputList(const fs::path &argument, const fs::path &repo_dir)
{
    const fs::path input_list = canonicalFile(argument);
    if (input_list.empty()) {
        throw std::runtime_error("Input list not found: " + argument.string());
    }

    std::smatch match;
    const std::string filename = input_list.filename().string();
    if (!std::regex_match(filename, match, std::regex(R"(input_run([0-9]{6})\.txt)"))) {
        throw std::runtime_error(
            "Input-list filename must have the form input_run039510.txt"
        );
    }

    InputData input{match[1].str(), {}};
    std::ifstream stream(input_list);
    if (!stream) {
        throw std::runtime_error("Could not open input list: " + input_list.string());
    }

    std::string raw_line;
    while (std::getline(stream, raw_line)) {
        const std::string line = trim(raw_line);
        if (line.empty() || line.front() == '#') continue;

        const fs::path listed(line);
        std::vector<fs::path> candidates;
        if (!listed.is_absolute()) candidates.push_back(input_list.parent_path() / listed);
        candidates.push_back(listed);
        candidates.push_back(repo_dir / listed.filename());

        fs::path found;
        for (const auto &candidate : candidates) {
            found = canonicalFile(candidate);
            if (!found.empty()) break;
        }
        if (found.empty()) {
            throw std::runtime_error(
                "ROOT file listed in " + input_list.string() + " not found: " + line
            );
        }
        input.files.push_back(std::move(found));
    }

    if (input.files.empty()) {
        throw std::runtime_error("Input list contains no ROOT files: " + input_list.string());
    }
    return input;
}

std::vector<unsigned int> allChannels()
{
    std::vector<unsigned int> channels;
    channels.reserve(CATHODE_CHANNELS.size() + MEMBRANE_CHANNELS.size());
    channels.insert(channels.end(), CATHODE_CHANNELS.begin(), CATHODE_CHANNELS.end());
    channels.insert(channels.end(), MEMBRANE_CHANNELS.begin(), MEMBRANE_CHANNELS.end());
    return channels;
}

DensityData collectHistograms(TChain &chain)
{
    const auto channels = allChannels();
    std::unordered_map<unsigned int, std::size_t> channel_indices;
    for (std::size_t index = 0; index < channels.size(); ++index) {
        channel_indices.emplace(channels[index], index);
    }

    chain.SetBranchStatus("*", 0);
    chain.SetBranchStatus("channel", 1);
    chain.SetBranchStatus("adc", 1);
    unsigned int channel = 0;
    std::vector<short> *adc = nullptr;
    chain.SetBranchAddress("channel", &channel);
    chain.SetBranchAddress("adc", &adc);

    const Long64_t entries = chain.GetEntries();
    std::size_t samples = 0;
    for (Long64_t entry = 0; entry < entries; ++entry) {
        if (chain.GetEntry(entry) <= 0) {
            throw std::runtime_error("Could not read ROOT entry " + std::to_string(entry));
        }
        if (channel_indices.count(channel) != 0 && adc != nullptr) {
            samples = adc->size();
            break;
        }
    }
    if (samples == 0) {
        throw std::runtime_error("No waveforms were found for the configured channels");
    }

    DensityData data;
    data.samples = samples;
    for (const auto selected_channel : channels) {
        data.bins.emplace(
            selected_channel,
            std::vector<std::uint32_t>(samples * ADC_BINS, 0)
        );
        data.counts.emplace(selected_channel, 0);
    }

    for (Long64_t entry = 0; entry < entries; ++entry) {
        if (chain.GetEntry(entry) <= 0 || adc == nullptr) {
            throw std::runtime_error("Could not read ROOT entry " + std::to_string(entry));
        }
        const auto found = data.bins.find(channel);
        if (found != data.bins.end()) {
            if (adc->size() != samples) {
                throw std::runtime_error(
                    "Channel " + std::to_string(channel) + ", entry "
                    + std::to_string(entry) + " has " + std::to_string(adc->size())
                    + " samples; expected " + std::to_string(samples)
                );
            }

            auto &bins = found->second;
            for (std::size_t sample = 0; sample < samples; ++sample) {
                const double scaled =
                    (static_cast<double>((*adc)[sample]) - ADC_MIN)
                    * ADC_BINS / (ADC_MAX - ADC_MIN);
                const int adc_bin = std::clamp(static_cast<int>(scaled), 0, ADC_BINS - 1);
                ++bins[sample * ADC_BINS + static_cast<std::size_t>(adc_bin)];
            }
            ++data.counts[channel];
        }

        if ((entry + 1) % 10000 == 0 || entry + 1 == entries) {
            std::cout << "Processed " << entry + 1 << '/' << entries
                      << " tree entries\n" << std::flush;
        }
    }
    chain.ResetBranchAddresses();
    return data;
}

std::unique_ptr<TH2I> makeHistogram(
    const DensityData &data,
    unsigned int channel,
    const std::string &name
)
{
    const auto count = data.counts.at(channel);
    const std::string title = "Channel " + std::to_string(channel) + " (n="
        + std::to_string(count) + " waveforms);Sample Index;ADC Value;Counts (log scale)";
    auto histogram = std::make_unique<TH2I>(
        name.c_str(), title.c_str(), static_cast<int>(data.samples), 0,
        static_cast<double>(data.samples), ADC_BINS, ADC_MIN, ADC_MAX
    );
    histogram->SetDirectory(nullptr);
    histogram->SetStats(false);
    histogram->SetMinimum(1.0);
    histogram->GetXaxis()->SetNdivisions(510);

    const auto &bins = data.bins.at(channel);
    for (std::size_t sample = 0; sample < data.samples; ++sample) {
        for (int adc_bin = 0; adc_bin < ADC_BINS; ++adc_bin) {
            const auto value = bins[sample * ADC_BINS + static_cast<std::size_t>(adc_bin)];
            if (value != 0) {
                histogram->SetBinContent(
                    static_cast<int>(sample) + 1, adc_bin + 1, value
                );
            }
        }
    }
    return histogram;
}

template <std::size_t N>
void plotDetectorOverview(
    const std::string &side,
    const std::array<unsigned int, N> &channels,
    const DensityData &data,
    const std::string &run,
    const fs::path &output_dir
)
{
    static_assert(N == 16, "The detector overview is an 8x2 grid");
    TCanvas canvas((side + "_overview_canvas").c_str(), "", 1800, 2100);
    canvas.Divide(2, 8, 0.001, 0.001);
    std::vector<std::unique_ptr<TH2I>> histograms;
    histograms.reserve(channels.size());

    for (std::size_t index = 0; index < channels.size(); ++index) {
        const auto channel = channels[index];
        auto *pad = canvas.cd(static_cast<int>(index) + 1);
        pad->SetRightMargin(0.15);
        if (data.counts.at(channel) == 0) {
            TLatex label;
            label.SetNDC();
            label.SetTextAlign(22);
            label.SetTextSize(0.08);
            label.DrawLatex(
                0.5, 0.5, ("Channel " + std::to_string(channel) + " (no waveforms)").c_str()
            );
            continue;
        }
        pad->SetLogz();
        histograms.push_back(makeHistogram(
            data, channel, side + "_overview_density_" + std::to_string(channel)
        ));
        histograms.back()->Draw("COLZ");
    }
    canvas.Update();
    const fs::path output = output_dir /
        ("allwaveforms_allarapucas_" + side + "_run_" + run + ".png");
    canvas.SaveAs(output.string().c_str());
    std::cout << "Saved " << output << '\n';
}

void printUsage(const char *program)
{
    std::cout
        << "Usage: " << program << " INPUT_LIST.txt\n\n"
        << "Create separate 8x2 PNG waveform-density overviews for the cathode "
           "and membrane channels.\n";
}

} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            printUsage(argv[0]);
            return 0;
        }
        if (argc != 2) {
            printUsage(argv[0]);
            return 2;
        }

        gROOT->SetBatch(true);
        gStyle->SetOptStat(0);
        gStyle->SetPalette(kViridis);
        gStyle->SetNumberContours(255);

#ifdef ANALYSIS_DIR
        const fs::path analysis_dir(ANALYSIS_DIR);
#else
        const fs::path analysis_dir = fs::current_path() / "analysis";
#endif
        const fs::path repo_dir = analysis_dir.parent_path();
        const InputData input = readInputList(argv[1], repo_dir);
        const fs::path output_dir = analysis_dir / "plots_allwaveforms";
        fs::create_directories(output_dir);

        std::cout << "Run: " << input.run << "\nInput files: " << input.files.size() << '\n';
        TChain chain("WaveformTree");
        for (const auto &file : input.files) {
            std::cout << "  " << file << '\n';
            if (chain.Add(file.string().c_str()) == 0) {
                throw std::runtime_error("Could not add ROOT file to chain: " + file.string());
            }
        }
        if (chain.GetEntries() == 0) {
            throw std::runtime_error("WaveformTree contains no entries");
        }

        const DensityData data = collectHistograms(chain);
        std::uint64_t total = 0;
        for (const auto &[channel, count] : data.counts) {
            (void)channel;
            total += count;
        }
        std::cout << "Total waveforms collected: " << total << '\n';

        plotDetectorOverview(
            "cathode", CATHODE_CHANNELS, data, input.run, output_dir
        );
        plotDetectorOverview(
            "membrane", MEMBRANE_CHANNELS, data, input.run, output_dir
        );
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
