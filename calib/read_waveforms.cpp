#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include <vector>
#include <map>
#include <algorithm>
#include "aux/waveform_utils.cpp" 
#include "TStyle.h"
 


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

    run_calib->SetBranchAddress("channel", &channel);
    run_calib->SetBranchAddress("adc", &adc);

    // TH2F X: 1024 Samples ; Y: ADC-baseline from -50 to 200
    TH2F* h2 = new TH2F("h", "channel 1050;sample;ADC - baseline",
        1024, 0, 1024,
        250, -50, 200);


    int n_ch = 0;
    for (Long64_t i = 0; i < nEntries; ++i){
        run_calib->GetEntry(i);

        if (!adc || adc->empty()) continue;
        if (channel != 1050) continue;
        int n = adc->size();

        // compute variables
        short mode = mode_of(*adc);
        double baseline = compute_baseline(*adc, 8, 12, mode);
        double std = noise(*adc, 240, baseline); 
        double integral = compute_integral(*adc, baseline, 255, 265);
        double pre_min, pre_max, sig_min, sig_max, post_min, post_max;
        compute_minmax(*adc, baseline, 0,   239,  pre_min,  pre_max);   // pre-signal (n=240)
        compute_minmax(*adc, baseline, 300, 1023, sig_min,  sig_max);   // signal
        compute_minmax(*adc, baseline, 350, 1023, post_min, post_max);  // post-signal

        for (int s = 0; s < n; s++)
            h2->Fill(s, (*adc)[s] - baseline);

        // confirmation iteration
        if (n_ch<3) {
            std::cout << "entry" << i 
            << " channel=" << channel
            << " nsamples=" << adc->size()
            << " adc[55]=" << (*adc)[255] << "\n";

            std::cout << " moda" << mode
            << " baseline" << baseline
            << " noise" << std << "\n";

            // Save 3 wf plots
            TH1D* h = new TH1D(Form("wf_%lld", i), Form("entry %lld;sample;ADC", i), n, 0, n);
            for (int s=0; s < n; s++)
                h->SetBinContent(s+1, (*adc)[s]);

            TCanvas* c = new TCanvas("c", "c", 1200, 400);
            h->Draw("L");
            h->SetStats(0);
            // legend
            TLegend* leg = new TLegend(0.65, 0.7, 0.88, 0.88); // x1, y1, x2, y2
            leg->AddEntry((TObject*)nullptr, Form("moda = %d", mode), "");
            leg->AddEntry((TObject*)nullptr, Form("baseline = %.1f", baseline), "");
            leg->AddEntry((TObject*)nullptr, Form("noise = %.2f", std), "");
            leg->Draw();

            c->SaveAs(Form("plots/raw_wfs/wf_%lld.png", i));
            delete c; delete h;
            n_ch++;
        }
        
    }

    TCanvas* cp = new TCanvas("cp", "cp", 1200, 700);
    cp->SetLogz();              // escala de cor log, como no plot que você quer reproduzir
    h2->Draw("colz");
    cp->SaveAs("plots/ADC-baseline_1050.png");
    delete cp; delete h2;

    delete run_calib;
    return 0;
}