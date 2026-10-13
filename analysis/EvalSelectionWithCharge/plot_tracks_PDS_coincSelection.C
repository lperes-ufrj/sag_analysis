// Run in the DUNE Gallery/LArSoft environment; see README.md beside this macro.
R__LOAD_LIBRARY(libgallery.so)
R__LOAD_LIBRARY(libcanvas.so)
R__LOAD_LIBRARY(liblardataobj_RawData.so)
R__LOAD_LIBRARY(liblardataobj_RecoBase.so)

// Reuse the actual scanner, including amplitude cuts, pulseStart and tie-breaking.
#define COINCIDENCE_NO_MAIN
#include "../../coincidence/run_coincidence.cpp"
#undef COINCIDENCE_NO_MAIN
#include "../../coincidence/coincidence_lib.cpp"

#include "gallery/Event.h"
#include "canvas/Utilities/InputTag.h"
#include "canvas/Persistency/Common/FindManyP.h"
#include "lardataobj/RawData/OpDetWaveform.h"
#include "lardataobj/RawData/RDTimeStamp.h"
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RawData/raw.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Hit.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TMultiGraph.h"
#include "TNamed.h"
#include "TStyle.h"
#include <array>

struct PDSChargeOptions {
    // NaN deliberately means unknown: no invented charge/light calibration.
    double tpcTickNs = std::numeric_limits<double>::quiet_NaN();
    double tickZeroRelativeToTriggerNs = std::numeric_limits<double>::quiet_NaN();
    double triggerTimestampUnitNs = std::numeric_limits<double>::quiet_NaN();
    double maxDriftUs = std::numeric_limits<double>::quiet_NaN();
    double lightToChargeOffsetNs = 0.;
    // OpDetWaveform timestamps are used exactly as in gallery_pdvddaphne.C and
    // run_coincidence.cpp: integer ns. Check this convention for your decoder.
    std::string waveformConfig;
    int maxDisplays = 20; // 0 = all; selection CSV always covers the entire file.
    bool plotRawADC = true;
    int rawTickRebin = 8; // signed mean ADC per bin, not charge integral
    std::string process = "pdvdkeepupstage1";
};

namespace pds_charge {
struct Pair { int left, right; };
struct Window { std::int64_t low, high; int pairs = 1; };

bool calibrated(const PDSChargeOptions& o)
{
    return std::isfinite(o.tpcTickNs) && o.tpcTickNs > 0
        && std::isfinite(o.tickZeroRelativeToTriggerNs)
        && std::isfinite(o.triggerTimestampUnitNs) && o.triggerTimestampUnitNs > 0
        && std::isfinite(o.maxDriftUs) && o.maxDriftUs > 0;
}

void note(const std::string& text)
{
    TLatex label; label.SetNDC(); label.SetTextSize(.028);
    label.DrawLatex(.12, .94, text.c_str());
}

void save(TCanvas& c, TFile& output, const fs::path& directory, const std::string& name)
{
    output.cd(); c.Write(name.c_str());
    c.SaveAs((directory / (name + ".png")).c_str());
}

// Only this input file is scanned. Event numbers must be unique within it:
// the upstream scanner keys by event, not (run, subrun, event).
void exportLight(gallery::Event& ev, const art::InputTag& tag, const fs::path& path)
{
    TFile output(path.c_str(), "CREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create " + path.string());
    TTree tree("WaveformTree", "Optical input for the unchanged coincidence scanner");
    Int_t event = 0, index = 0; UInt_t channel = 0; ULong64_t timestamp = 0;
    std::vector<short> adc;
    tree.Branch("event", &event); tree.Branch("channel", &channel);
    tree.Branch("timestamp", &timestamp); tree.Branch("waveform_index", &index);
    tree.Branch("adc", &adc);
    std::set<int> seen;
    for (ev.toBegin(); !ev.atEnd(); ev.next()) {
        event = ev.eventAuxiliary().event();
        if (!seen.insert(event).second)
            throw std::runtime_error("Repeated event number: split input by run/subrun before scanning");
        auto waves = ev.getValidHandle<std::vector<raw::OpDetWaveform>>(tag);
        for (std::size_t i = 0; i < waves->size(); ++i) {
            const auto& w = waves->at(i);
            index = i; channel = w.ChannelNumber(); timestamp = w.TimeStamp();
            adc.assign(w.begin(), w.end()); tree.Fill();
        }
    }
    tree.Write(); // tree is destroyed before its owning TFile
}

std::map<int, std::vector<Pair>> readPairs(const fs::path& csv)
{
    std::ifstream input(csv); std::string line;
    if (!input) throw std::runtime_error("Cannot read selection CSV");
    std::getline(input, line);
    std::map<int, std::vector<Pair>> result;
    std::set<std::tuple<int,int,int>> seen;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int event, saveCh, saveIndex, leftCh, left, rightCh, right;
        long long dtPair, dtSave; unsigned long long timestamp;
        if (!(row >> event >> saveCh >> saveIndex >> dtPair >> dtSave >> timestamp
                  >> leftCh >> left >> rightCh >> right))
            throw std::runtime_error("Malformed selection CSV row");
        if (seen.emplace(event, left, right).second) result[event].push_back({left, right});
    }
    return result;
}

std::vector<Window> windows(const std::vector<Pair>& pairs,
                           const std::vector<raw::OpDetWaveform>& waves,
                           const WaveformAnalyzer& analyzer, std::int64_t halfWidth)
{
    std::vector<Window> result;
    for (const auto& p : pairs) {
        auto time = [&](int i) {
            const auto& w = waves.at(i);
            const std::vector<short> adc(w.begin(), w.end());
            return static_cast<std::int64_t>(static_cast<ULong64_t>(w.TimeStamp()))
                 + analyzer.pulseStart(adc, w.ChannelNumber()) * SAMPLE_PERIOD_NS;
        };
        const auto a = time(p.left), b = time(p.right);
        result.push_back({std::min(a,b) - halfWidth, std::max(a,b) + halfWidth, 1});
    }
    std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.low < b.low; });
    std::vector<Window> merged;
    for (auto w : result) {
        if (merged.empty() || w.low > merged.back().high) merged.push_back(w);
        else { merged.back().high = std::max(merged.back().high, w.high); merged.back().pairs += w.pairs; }
    }
    return merged;
}

void tracks(const std::vector<recob::Track>& tracks,
            const art::FindManyP<recob::Hit>& associations,
            bool gated, double low, double high, const std::string& title,
            const std::string& name, TFile& output, const fs::path& directory,
            std::ofstream& summary, int run, int subrun, int event, int window)
{
    TCanvas canvas("tracks", title.c_str(), 1600, 650); canvas.Divide(3, 1);
    std::array<std::unique_ptr<TMultiGraph>,3> graphs;
    for (auto& g : graphs) g = std::make_unique<TMultiGraph>();
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        const auto& track = tracks[i];
        const auto hits = associations.at(i);
        std::size_t inside = 0;
        for (const auto& h : hits) if (h->PeakTime() >= low && h->PeakTime() <= high) ++inside;
        const double fraction = hits.empty() ? 0. : double(inside) / hits.size();
        // Highlight is a time-compatible candidate, not a muon/PDS match.
        const bool candidate = gated && inside > 0;
        const auto a = track.Vertex(), b = track.End();
        summary << run << ',' << subrun << ',' << event << ',' << window << ',' << i << ','
                << track.ID() << ',' << track.Length() << ',' << hits.size() << ',';
        if (gated) summary << inside << ',' << fraction;
        else summary << ',';
        summary << ',' << a.X() << ',' << a.Y() << ',' << a.Z() << ','
                << b.X() << ',' << b.Y() << ',' << b.Z() << '\n';
        for (int projection = 0; projection < 3; ++projection) {
            auto g = new TGraph;
            for (std::size_t point = 0; point < track.NumberTrajectoryPoints(); ++point) {
                if (!track.HasValidPoint(point)) continue;
                const auto v = track.LocationAtPoint(point);
                g->SetPoint(g->GetN(), projection == 0 ? v.X() : v.Z(),
                            projection == 2 ? v.X() : v.Y());
            }
            if (!g->GetN()) { delete g; continue; }
            g->SetLineColor(candidate ? kRed+1 : (gated ? kGray+1 : kBlue+1));
            g->SetLineWidth(candidate ? 2 : 1);
            graphs[projection]->Add(g, "L");
        }
    }
    const char* axes[] = {";X [cm];Y [cm]", ";Z [cm];Y [cm]", ";Z [cm];X [cm]"};
    for (int p = 0; p < 3; ++p) {
        canvas.cd(p+1); graphs[p]->SetTitle((title + axes[p]).c_str());
        if (graphs[p]->GetListOfGraphs()) graphs[p]->Draw("A");
        note(gated ? "Red: has hit(s) in window; gray: other tracks" : "UNCALIBRATED: full-event reconstructed tracks");
    }
    save(canvas, output, directory, name + "_tracks");
}

void hits(const std::vector<recob::Hit>& hits, double low, double high,
          const std::string& title, const std::string& name,
          TFile& output, const fs::path& directory)
{
    using Plane = std::tuple<unsigned int,unsigned int,unsigned int>;
    std::map<Plane,std::vector<const recob::Hit*>> groups;
    for (const auto& h : hits) {
        const auto id = h.WireID(); groups[{id.Cryostat,id.TPC,id.Plane}].push_back(&h);
    }
    for (const auto& entry : groups) {
        const auto [cryo,tpc,plane] = entry.first;
        const std::string suffix = "_hits_C"+std::to_string(cryo)+"_T"+std::to_string(tpc)+"_P"+std::to_string(plane);
        unsigned int maxWire = 0;
        for (const auto h : entry.second) maxWire = std::max(maxWire, h->WireID().Wire);
        TCanvas c("hits", title.c_str(), 1400, 800); c.SetRightMargin(.16);
        TH2F map("hit_charge", (title+suffix+";Wire;TPC peak tick;Sum hit integral [ADC tick]").c_str(),
                 maxWire+1, -.5, maxWire+.5, 600, low,
                 std::nextafter(high, std::numeric_limits<double>::infinity()));
        map.SetDirectory(nullptr); map.SetStats(false);
        for (const auto h : entry.second)
            if (h->PeakTime() >= low && h->PeakTime() <= high)
                map.Fill(h->WireID().Wire, h->PeakTime(), h->Integral());
        map.Draw("COLZ"); save(c, output, directory, name+suffix);
    }
}

void rawADC(const std::vector<raw::RawDigit>& digits, double low, double high,
            int rebin, const std::string& title, const std::string& name,
            TFile& output, const fs::path& directory)
{
    if (digits.empty()) return;
    unsigned int maxChannel = 0; std::size_t maxSamples = 0;
    for (const auto& d : digits) {
        maxChannel = std::max(maxChannel, d.Channel());
        maxSamples = std::max(maxSamples, std::size_t(d.Samples()));
    }
    // Integer ticks in the inclusive requested window, clipped to available data.
    const int first = std::clamp(std::ceil(low), 0., double(maxSamples));
    const int stop = std::clamp(std::floor(high)+1, 0., double(maxSamples));
    if (stop <= first) return;
    const int bins = (stop-first+rebin-1)/rebin;
    TCanvas c("raw", title.c_str(), 1700, 900); c.SetRightMargin(.16);
    TH2F map("raw_adc", (title+";TPC channel;TPC sample tick;Mean (ADC - pedestal)").c_str(),
             maxChannel+1, -.5, maxChannel+.5, bins, first, first+bins*rebin);
    map.SetDirectory(nullptr); map.SetStats(false);
    for (const auto& d : digits) {
        std::vector<short> adc(d.Samples());
        raw::Uncompress(d.ADCs(), adc, d.Compression());
        for (int bin = 0; bin < bins; ++bin) {
            const int begin = first+bin*rebin;
            const int end = std::min({begin+rebin, stop, int(adc.size())});
            if (end <= begin) continue;
            double sum = 0.;
            for (int tick = begin; tick < end; ++tick) sum += adc[tick]-d.GetPedestal();
            map.SetBinContent(d.Channel()+1, bin+1, sum/(end-begin));
        }
    }
    map.Draw("COLZ"); save(c, output, directory, name+"_raw_adc");
}
} // namespace pds_charge

void plot_tracks_PDS_coincSelection(
    const char* inputFile = "np02vd_raw_run043374_0000_df-s04-d0_dw_0_20260320T161028_cosmic_keepup.root",
    const char* outputDirectory = "analysis/EvalSelectionWithCharge/output",
    PDSChargeOptions options = PDSChargeOptions())
{
    using namespace pds_charge;
    if (options.maxDisplays < 0 || options.rawTickRebin < 1 || !std::isfinite(options.lightToChargeOffsetNs))
        throw std::runtime_error("Invalid display limit, ADC rebinning, or timing offset");
    const bool gated = calibrated(options);
    const fs::path directory = fs::path(outputDirectory) / makeExecutionTimestamp();
    if (!fs::create_directories(directory))
        throw std::runtime_error("Output directory already exists: " + directory.string());
    Arguments args; // defaults come directly from run_coincidence.cpp
    args.config = options.waveformConfig.empty()
        ? fs::path(__FILE__).parent_path()/"../../coincidence/waveform_intervals.ini"
        : fs::path(options.waveformConfig);
    const WaveformAnalyzer analyzer(args.config.string());
    const art::InputTag opticalTag("pdvddaphne", "daq", options.process);
    const art::InputTag trackTag("pandoraTrack", "", options.process);
    const art::InputTag hitTag("gaushit", "", options.process);
    const art::InputTag rawTag("tpcrawdecoder", "daq", options.process);
    const art::InputTag triggerTag("daq", "trigger", options.process);
    gallery::Event ev(std::vector<std::string>{inputFile});
    const fs::path opticalFile = directory/"optical_input.root";
    exportLight(ev, opticalTag, opticalFile);
    ev.toBegin();
    if (!ev.atEnd()) args.run = std::to_string(ev.eventAuxiliary().run());
    auto chain = createChain({opticalFile.string()});
    const fs::path csv = directory/"selected_waveforms.csv";
    const auto stats = scanCoincidence(*chain, analyzer, args, {}, csv.string());
    CreateTXTfile(directory/"coincidence_settings.txt", csv, directory.filename().string(),
                  args, {}, args.save, {}, {inputFile}, stats);
    const auto selected = readPairs(csv);
    TFile output((directory/"charge_displays.root").c_str(), "CREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create display ROOT file");
    std::ostringstream settings;
    settings << std::setprecision(17) << "input=" << inputFile << "\ncalibrated=" << gated
             << "\ntpc_tick_ns=" << options.tpcTickNs
             << "\ntick_zero_relative_to_trigger_ns=" << options.tickZeroRelativeToTriggerNs
             << "\ntrigger_timestamp_unit_ns=" << options.triggerTimestampUnitNs
             << "\nmax_drift_us=" << options.maxDriftUs
             << "\nlight_to_charge_offset_ns=" << options.lightToChargeOffsetNs
             << "\nraw_tick_rebin=" << options.rawTickRebin << "\nmax_displays=" << options.maxDisplays;
    TNamed("timing_and_display_settings", settings.str().c_str()).Write();
    std::ofstream(directory/"timing_settings.txt") << settings.str() << '\n';
    std::ofstream windowsCSV(directory/"windows.csv"), tracksCSV(directory/"tracks.csv");
    std::ofstream clocksCSV(directory/"clock_diagnostics.csv");
    clocksCSV << "run,subrun,event,trigger_timestamp_native,min_optical_timestamp_native,max_optical_timestamp_native\n";
    windowsCSV << "run,subrun,event,window,optical_low_ns,optical_high_ns,n_pairs,calibrated,charge_low_tick,charge_high_tick,displayed\n";
    tracksCSV << "run,subrun,event,window,track_index,track_id,length_cm,n_associated_hits,n_hits_in_window,fraction_in_window,start_x_cm,start_y_cm,start_z_cm,end_x_cm,end_y_cm,end_z_cm\n";
    gStyle->SetOptStat(0);
    if (!gated) std::cerr << "Timing not calibrated: exporting FULL-EVENT views only. Set PDSChargeOptions timing fields for gated views.\n";
    int displays = 0;
    for (ev.toBegin(); !ev.atEnd(); ev.next()) {
        const int event = ev.eventAuxiliary().event();
        const auto found = selected.find(event);
        if (found == selected.end()) continue;
        const int run = ev.eventAuxiliary().run(), subrun = ev.eventAuxiliary().subRun();
        const auto waves = ev.getValidHandle<std::vector<raw::OpDetWaveform>>(opticalTag);
        const auto ws = windows(found->second, *waves, analyzer, args.window_ticks*SAMPLE_PERIOD_NS);
        const auto trigger = ev.getValidHandle<raw::RDTimeStamp>(triggerTag)->GetTimeStamp();
        double minOptical = std::numeric_limits<double>::infinity(), maxOptical = -minOptical;
        for (const auto& w : *waves) {
            minOptical = std::min(minOptical, double(w.TimeStamp()));
            maxOptical = std::max(maxOptical, double(w.TimeStamp()));
        }
        clocksCSV << run << ',' << subrun << ',' << event << ',' << trigger << ','
                  << std::setprecision(17) << minOptical << ',' << maxOptical << '\n';
        const long double triggerNs = gated ? static_cast<long double>(trigger)*options.triggerTimestampUnitNs : 0;
        bool overviewDrawn = false;
        for (std::size_t i = 0; i < ws.size(); ++i) {
            const auto w = ws[i];
            // Subtract large absolute clocks in long double BEFORE narrowing.
            const double low = gated ? double((static_cast<long double>(w.low)-triggerNs
                + options.lightToChargeOffsetNs-options.tickZeroRelativeToTriggerNs)/options.tpcTickNs) : 0.;
            const double high = gated ? double((static_cast<long double>(w.high)-triggerNs
                + options.lightToChargeOffsetNs-options.tickZeroRelativeToTriggerNs
                + options.maxDriftUs*1000.L)/options.tpcTickNs) : 0.;
            if (gated && (!std::isfinite(low) || !std::isfinite(high) || high <= low))
                throw std::runtime_error("Timing calibration produces an invalid charge window");
            const bool draw = (!options.maxDisplays || displays < options.maxDisplays) && (gated || !overviewDrawn);
            windowsCSV << run << ',' << subrun << ',' << event << ',' << i << ',' << w.low << ',' << w.high << ',' << w.pairs << ',' << gated << ',';
            if (gated) windowsCSV << std::setprecision(15) << low << ',' << high;
            else windowsCSV << ',';
            windowsCSV << ',' << draw << '\n';
            if (!draw) continue;
            auto trks = ev.getValidHandle<std::vector<recob::Track>>(trackTag);
            auto charge = ev.getValidHandle<std::vector<recob::Hit>>(hitTag);
            if (gated && std::none_of(charge->begin(), charge->end(), [&](const auto& h) {
                    return h.PeakTime() >= low && h.PeakTime() <= high;
                })) std::cerr << "No charge hits in event " << event << " window " << i
                              << "; check timing calibration and readout coverage.\n";
            art::FindManyP<recob::Hit> associated(trks, ev, trackTag);
            if (!associated.isValid()) throw std::runtime_error("Missing pandoraTrack track-hit associations");
            double plotLow = low, plotHigh = high;
            if (!gated) {
                plotLow = 0; plotHigh = 1;
                for (const auto& h : *charge) plotHigh = std::max(plotHigh, double(h.EndTick())+1);
            }
            const std::string name = "r"+std::to_string(run)+"_s"+std::to_string(subrun)+"_e"+std::to_string(event)
                                   + (gated ? "_w"+std::to_string(i) : "_FULL_EVENT_UNCALIBRATED");
            std::cout << "Drawing " << name << " ticks [" << plotLow << ", " << plotHigh << "]\n";
            tracks(*trks, associated, gated, plotLow, plotHigh, name, name, output, directory, tracksCSV, run, subrun, event, gated ? int(i) : -1);
            hits(*charge, plotLow, plotHigh, name, name, output, directory);
            if (options.plotRawADC) {
                auto digits = ev.getValidHandle<std::vector<raw::RawDigit>>(rawTag);
                if (!gated) for (const auto& d : *digits) plotHigh = std::max(plotHigh, double(d.Samples()));
                rawADC(*digits, plotLow, plotHigh, options.rawTickRebin, name, name, output, directory);
            }
            ++displays; overviewDrawn = true;
        }
    }
    if (!windowsCSV || !tracksCSV || !clocksCSV) throw std::runtime_error("Failed writing display summaries");
    std::cout << "Saved " << displays << " displays; " << stats.selected_events
              << " light-selected events. Outputs: " << directory << '\n';
    if (options.maxDisplays && displays >= options.maxDisplays)
        std::cout << "Display limit reached; use maxDisplays=0 for all windows.\n";
}
