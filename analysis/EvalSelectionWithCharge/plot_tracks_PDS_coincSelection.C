R__LOAD_LIBRARY(libgallery.so)
R__LOAD_LIBRARY(libcanvas.so)
R__LOAD_LIBRARY(liblardataobj_RawData.so)

#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "gallery/Event.h"
#include "canvas/Utilities/InputTag.h"
#include "lardataobj/RawData/OpDetWaveform.h"

void gallery_pdvddaphne(const char* inputFile,
                        const char* outputFile = "pdvddaphne_waveforms.root")
{
    std::vector<std::string> inputFiles;
    inputFiles.emplace_back(inputFile);

    gallery::Event ev(inputFiles);

    TFile fout(outputFile, "RECREATE");
    TTree tree("WaveformTree", "ProtoDUNE-VD DAPHNE optical waveforms");

    int run = 0;
    int subrun = 0;
    int event = 0;
    int waveform_index = 0;

    unsigned int channel = 0;
    unsigned long long timestamp = 0;
    unsigned int nsamples = 0;

    std::vector<short> adc;

    tree.Branch("run", &run, "run/I");
    tree.Branch("subrun", &subrun, "subrun/I");
    tree.Branch("event", &event, "event/I");
    tree.Branch("waveform_index", &waveform_index, "waveform_index/I");
    tree.Branch("channel", &channel, "channel/i");
    tree.Branch("timestamp", &timestamp, "timestamp/l");
    tree.Branch("nsamples", &nsamples, "nsamples/i");
    tree.Branch("adc", &adc);

    art::InputTag waveformTag{"pdvddaphne", "daq"};

    long long totalWaveforms = 0;
    long long totalEvents = 0;

    for (ev.toBegin(); !ev.atEnd(); ev.next()) {

        ++totalEvents;

        run = ev.eventAuxiliary().run();
        subrun = ev.eventAuxiliary().subRun();
        event = ev.eventAuxiliary().event();

        auto const& waveforms =
            ev.getValidHandle<std::vector<raw::OpDetWaveform>>(waveformTag);

        std::cout << "Run " << run
                  << ", subrun " << subrun
                  << ", event " << event
                  << ": " << waveforms->size()
                  << " waveforms" << std::endl;

        for (size_t i = 0; i < waveforms->size(); ++i) {

            auto const& wf = waveforms->at(i);

            waveform_index = static_cast<int>(i);
            channel = wf.ChannelNumber();
            timestamp = wf.TimeStamp();

            adc.clear();
            adc.reserve(wf.size());

            for (auto const sample : wf) {
                adc.push_back(static_cast<short>(sample));
            }

            nsamples = adc.size();

            tree.Fill();
            ++totalWaveforms;
        }
    }

    fout.cd();
    tree.Write();
    fout.Close();

    std::cout << "\nSaved " << totalWaveforms
              << " waveforms from " << totalEvents
              << " events to:\n" << outputFile << std::endl;
}