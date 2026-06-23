#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TTree.h"

void interactive_plot_channel(
    const char* inputFile =
        "/dune/persistent/users/lperes/sag_analysis/"
        "pdvddaphne_run39523_0000_waveforms.root"
)
{
    TFile* f = TFile::Open(inputFile, "READ");

    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: could not open:\n" << inputFile << std::endl;
        return;
    }

    TTree* tree = nullptr;
    f->GetObject("WaveformTree", tree);

    if (!tree) {
        std::cerr << "ERROR: WaveformTree was not found." << std::endl;
        f->Close();
        return;
    }

    int run = 0;
    int subrun = 0;
    int event = 0;
    int waveform_index = 0;

    unsigned int channel = 0;
    unsigned long long timestamp = 0;
    unsigned int nsamples = 0;

    std::vector<short>* adc = nullptr;

    tree->SetBranchAddress("run", &run);
    tree->SetBranchAddress("subrun", &subrun);
    tree->SetBranchAddress("event", &event);
    tree->SetBranchAddress("waveform_index", &waveform_index);
    tree->SetBranchAddress("channel", &channel);
    tree->SetBranchAddress("timestamp", &timestamp);
    tree->SetBranchAddress("nsamples", &nsamples);
    tree->SetBranchAddress("adc", &adc);

    int targetEvent = 1;

    std::cout << "\nEnter event number [default: 1]: ";
    std::string eventInput;
    std::getline(std::cin, eventInput);

    if (!eventInput.empty()) {
        try {
            targetEvent = std::stoi(eventInput);
        }
        catch (...) {
            std::cerr << "Invalid event number. Using event 1." << std::endl;
            targetEvent = 1;
        }
    }

    // First pass: collect available channels and their waveform counts.
    std::map<unsigned int, long long> channelCounts;
    Long64_t nentries = tree->GetEntries();

    int selectedRun = -1;
    int selectedSubrun = -1;

    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);

        if (event != targetEvent) continue;

        selectedRun = run;
        selectedSubrun = subrun;
        channelCounts[channel]++;
    }

    if (channelCounts.empty()) {
        std::cerr << "\nNo waveforms found for event "
                  << targetEvent << "." << std::endl;
        f->Close();
        return;
    }

    std::cout << "\n============================================\n";
    std::cout << "Run " << selectedRun
              << ", SubRun " << selectedSubrun
              << ", Event " << targetEvent << "\n";
    std::cout << "Available channels:\n";
    std::cout << "--------------------------------------------\n";
    std::cout << " Channel     Number of waveforms\n";
    std::cout << "--------------------------------------------\n";

    for (const auto& item : channelCounts) {
        std::cout << " " << item.first
                  << "          " << item.second << "\n";
    }

    std::cout << "============================================\n";

    unsigned int targetChannel = 0;

    std::cout << "\nChoose a channel to plot: ";
    if (!(std::cin >> targetChannel)) {
        std::cerr << "Invalid channel input." << std::endl;
        f->Close();
        return;
    }

    if (channelCounts.find(targetChannel) == channelCounts.end()) {
        std::cerr << "Channel " << targetChannel
                  << " is not available in event "
                  << targetEvent << "." << std::endl;
        f->Close();
        return;
    }

    std::vector<TGraph*> graphs;
    std::vector<int> waveformIndices;

    // Second pass: create one graph for every waveform in the selected channel.
    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);

        if (event != targetEvent || channel != targetChannel) continue;

        std::vector<double> x(adc->size());
        std::vector<double> y(adc->size());

        for (size_t j = 0; j < adc->size(); ++j) {
            x[j] = static_cast<double>(j);
            y[j] = static_cast<double>(adc->at(j));
        }

        TGraph* gr = new TGraph(
            static_cast<int>(adc->size()),
            x.data(),
            y.data()
        );

        gr->SetLineWidth(1);

        graphs.push_back(gr);
        waveformIndices.push_back(waveform_index);
    }

    std::cout << "\nPlotting " << graphs.size()
              << " waveforms for channel "
              << targetChannel << "." << std::endl;

    std::string baseName =
        "run" + std::to_string(selectedRun) +
        "_event" + std::to_string(targetEvent) +
        "_channel" + std::to_string(targetChannel);

    std::string overlayName = baseName + "_overlay.png";
    std::string pdfName = baseName + "_all_waveforms.pdf";

    // Overlay PNG
    TCanvas* overlayCanvas = new TCanvas(
        "overlayCanvas",
        "All waveforms overlay",
        1400,
        700
    );

    TMultiGraph* multiGraph = new TMultiGraph();

    for (TGraph* gr : graphs) {
        multiGraph->Add(gr, "L");
    }

    multiGraph->SetTitle(
        Form(
            "Run %d, SubRun %d, Event %d, Channel %u: all waveforms;"
            "Sample;ADC",
            selectedRun,
            selectedSubrun,
            targetEvent,
            targetChannel
        )
    );

    multiGraph->Draw("A");
    overlayCanvas->SaveAs(overlayName.c_str());

    // Multipage PDF: one waveform per page.
    TCanvas* individualCanvas = new TCanvas(
        "individualCanvas",
        "Individual waveforms",
        1400,
        700
    );

    individualCanvas->Print((pdfName + "[").c_str());

    for (size_t i = 0; i < graphs.size(); ++i) {
        graphs[i]->SetTitle(
            Form(
                "Run %d, SubRun %d, Event %d, Channel %u, Waveform index %d;"
                "Sample;ADC",
                selectedRun,
                selectedSubrun,
                targetEvent,
                targetChannel,
                waveformIndices[i]
            )
        );

        graphs[i]->Draw("AL");
        individualCanvas->Print(pdfName.c_str());
    }

    individualCanvas->Print((pdfName + "]").c_str());

    std::cout << "\nSaved:\n"
              << "  " << overlayName << "\n"
              << "  " << pdfName << "\n"
              << std::endl;

    f->Close();
}