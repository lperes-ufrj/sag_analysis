#include "waveform_plotting.hpp"

#include <TCanvas.h>
#include <TGraph.h>
#include <TGraphAsymmErrors.h>
#include <TH2I.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

namespace waveform_plotting {

constexpr int MAX_WAVEFORM_DENSITY_Y_BINS = 2000;

double percentileFromSorted(const std::vector<float> &values, double quantile)
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
    result.median.resize(sample_count);
    result.mean_uncertainty.resize(sample_count);
    result.percentile_16.resize(sample_count);
    result.percentile_84.resize(sample_count);
    std::vector<float> values(records.size());

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        result.samples[sample] = static_cast<std::int64_t>(sample);
        float sum = 0.0F;
        for (std::size_t row = 0; row < records.size(); ++row) {
            values[row] = static_cast<float>(records[row].adc[sample])
                - static_cast<float>(records[row].baseline);
            sum += values[row];
        }
        result.mean[sample] = sum / static_cast<float>(records.size());
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
        result.median[sample] = static_cast<float>(percentileFromSorted(values, 0.5));
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
    const fs::path &output_dir,
    const std::string &selection_title,
    const std::string &selection_suffix
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
        throw std::runtime_error("No waveforms available for statistics");
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
            selection_title,
            selection_suffix,
            output_dir
        );
    }

    const fs::path output_stem = output_dir / ("mean_median_waveforms_" + label);
    canvas.SaveAs((output_stem.string() + ".pdf").c_str());
    canvas.SaveAs((output_stem.string() + ".png").c_str());
    std::cout << "Saved " << output_stem << ".pdf and .png\n";
}

}  // namespace waveform_plotting
