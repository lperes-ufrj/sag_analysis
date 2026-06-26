#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TChain.h"
#include <vector>

int main(){


    TChain* run_calib = new TChain("WaveformTree");

    run_calib->Add("np02vd_raw_run039357_*_gallery.root"); // read all root files in order

    // see amount of entries
    const Long64_t nEntries = run_calib->GetEntries();

    if (nEntries == 0) {
        std::cerr << "Null Entry\n";
        return 1;
    }
    std::cout << nEntries << " entries read\n";
    
    // plot waveforms for one channel
    /*
    TCanvas* c = new TCanvas("c", "c", 1200, 500);
    run_calib->Draw("adc:Iteration$", "channel==2050", "colz");
    c->SaveAs(" [PLOTS/persistence_ch2050.png");
    delete c;
    */

    std::vector<short>* adc = nullptr;   // vector<short>
    unsigned int channel  = 0;           // /i
    unsigned int nsamples = 0;  

    run_calib->SetBranchAddress("channel", &channel);
    run_calib->SetBranchAddress("adc", &adc);

    for (Long64_t i = 0; i < nEntries; ++i){
        run_calib->GetEntry(i);
        if (!adc || adc->empty()) continue;

        if (i<3) {
            std::cout << "entry" << i 
            << " channel=" << channel
            << " nsamples=" << adc->size()
            << " adc[55]=" << (*adc)[255] << "\n";
        }
    }

    delete run_calib;
    return 0;
}