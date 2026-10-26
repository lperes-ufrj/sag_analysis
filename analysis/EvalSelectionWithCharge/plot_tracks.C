R__LOAD_LIBRARY(libgallery.so)
R__LOAD_LIBRARY(liblardataobj_RecoBase.so)
R__LOAD_LIBRARY(liblardataobj_RawData.so)

#include "gallery/Event.h"
#include "gallery/Handle.h"
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/RecoBase/Track.h"
#include "lardataobj/RecoBase/Shower.h"
#include "lardataobj/RawData/RawDigit.h"
#include "lardataobj/RawData/raw.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"
#include "TH2F.h"
#include "TString.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// entry=0 draws the first event. Increase entry to view another event.
void plot_tracks(
    const char* inputFile = "np02vd_raw_run043374_0000_df-s04-d0_dw_0_20260320T161028_cosmic_keepup.root",
    unsigned int entry = 12,
    const char* showerLabel = "pandoraShower")
{
    gallery::Event event(std::vector<std::string>{inputFile});
    for (unsigned int i = 0; i < entry && !event.atEnd(); ++i) event.next();
    if (event.atEnd()) {
        std::cerr << "Event entry " << entry << " is outside the file.\n";
        return;
    }

    auto tracks = event.getValidHandle<std::vector<recob::Track>>(
        art::InputTag("pandoraTrack", "", "pdvdkeepupstage1"));
    gallery::Handle<std::vector<recob::Shower>> showers;
    event.getByLabel(art::InputTag(showerLabel, "", "pdvdkeepupstage1"), showers);
    if (!showers.isValid()) std::cout << "No reconstructed shower collection: " << showerLabel << '\n';
    auto canvas = new TCanvas("tracks_YX", "Reconstructed tracks and showers", 900, 900);
    canvas->SetLeftMargin(.15);
    auto plot = new TMultiGraph;
    plot->SetTitle(Form("Run %u, event %u;Y [cm];X [cm]",
                        event.eventAuxiliary().run(), event.eventAuxiliary().event()));

    TGraph* trackExample = nullptr;
    TGraph* showerExample = nullptr;
    for (const auto& track : *tracks) {
        auto line = new TGraph;
        for (std::size_t i = 0; i < track.NumberTrajectoryPoints(); ++i) {
            if (!track.HasValidPoint(i)) continue;
            const auto point = track.LocationAtPoint(i);
            line->SetPoint(line->GetN(), point.Y(), point.X());
        }
        if (!line->GetN()) { delete line; continue; }
        line->SetLineColor(kBlue + 1);
        if (!trackExample) trackExample = line;
        plot->Add(line, "L");
    }
    if (showers.isValid()) for (const auto& shower : *showers) {
        const auto& start = shower.ShowerStart();
        auto axis = new TGraph;
        axis->SetPoint(0, start.Y(), start.X());
        if (shower.has_length() && shower.Length() > 0) {
            const auto end = start + shower.Length() * shower.Direction();
            axis->SetPoint(1, end.Y(), end.X());
        }
        axis->SetLineColor(kRed + 1);
        axis->SetLineWidth(2);
        axis->SetMarkerColor(kRed + 1);
        axis->SetMarkerStyle(20);
        if (!showerExample) showerExample = axis;
        // Show the reconstructed axis, or only the start when length is absent.
        plot->Add(axis, axis->GetN() > 1 ? "L" : "P");
    }
    if (!plot->GetListOfGraphs()) {
        std::cerr << "No tracks or showers to draw in this event.\n";
        delete plot;
        delete canvas;
    } else {
        plot->Draw("A");
        auto legend = new TLegend(.65, .77, .89, .89);
        if (trackExample) legend->AddEntry(trackExample, "Tracks", "l");
        if (showerExample) legend->AddEntry(showerExample, "Showers (axis)", "lp");
        legend->Draw();
        canvas->SaveAs(Form("tracks_run%u_event%u_YX.png",
                           event.eventAuxiliary().run(), event.eventAuxiliary().event()));
    }
    std::cout << "Plotted " << tracks->size() << " tracks and "
              << (showers.isValid() ? showers->size() : 0)
              << " showers: Y horizontal, X vertical.\n";

    auto digits = event.getValidHandle<std::vector<raw::RawDigit>>(
        art::InputTag("tpcrawdecoder", "daq", "pdvdkeepupstage1"));
    if (digits->empty()) { std::cerr << "No raw TPC digits in this event.\n"; return; }
    unsigned int maxChannel = 0, maxSamples = 0;
    for (const auto& digit : *digits) {
        maxChannel = std::max(maxChannel, digit.Channel());
        maxSamples = std::max(maxSamples, unsigned(digit.Samples()));
    }
    auto rawCanvas = new TCanvas("raw_TPC", "Raw TPC samples", 1600, 900);
    rawCanvas->SetRightMargin(.17);
    auto adc = new TH2F("raw_adc", Form("Run %u, event %u;TPC channel;Sample tick;ADC - pedestal",
        event.eventAuxiliary().run(), event.eventAuxiliary().event()),
        maxChannel + 1, -.5, maxChannel + .5, maxSamples, -.5, maxSamples - .5);
    adc->SetDirectory(nullptr);
    adc->SetStats(false);
    for (const auto& digit : *digits) {
        std::vector<short> samples(digit.Samples());
        raw::Uncompress(digit.ADCs(), samples, digit.Compression());
        for (std::size_t tick = 0; tick < samples.size(); ++tick)
            adc->SetBinContent(digit.Channel() + 1, tick + 1, samples[tick] - digit.GetPedestal());
    }
    // Display range only: retain all sample values, including large pulses.
    adc->SetMinimum(-100);
    adc->SetMaximum(100);
    adc->Draw("COLZ");
    rawCanvas->SaveAs(Form("rawTPC_run%u_event%u.png",
                          event.eventAuxiliary().run(), event.eventAuxiliary().event()));
    std::cout << "Plotted " << digits->size() << " raw TPC waveforms (color range -100 to +100 ADC).\n";
}
