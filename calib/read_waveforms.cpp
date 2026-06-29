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
#include "aux/waveform_utils.h" 
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

    // output .root
    TFile* fout = new TFile("metrics_ch1050.root", "RECREATE");
    TTree* tout = new TTree("Metrics", "per-waveform metrics, channel 1050");

    double o_baseline, o_noise, o_integral;
    double o_pre_min, o_pre_max, o_sig_min, o_sig_max, o_post_min, o_post_max;
    short  o_mode;

    tout->Branch("mode",     &o_mode);
    tout->Branch("baseline", &o_baseline);
    tout->Branch("noise",    &o_noise);
    tout->Branch("integral", &o_integral);
    tout->Branch("pre_min",  &o_pre_min);
    tout->Branch("pre_max",  &o_pre_max);
    tout->Branch("sig_min",  &o_sig_min);
    tout->Branch("sig_max",  &o_sig_max);
    tout->Branch("post_min", &o_post_min);
    tout->Branch("post_max", &o_post_max);

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

        // save in output.root
        o_mode     = mode;
        o_baseline = baseline;
        o_noise    = std;
        o_integral = integral;
        o_pre_min  = pre_min;  o_pre_max  = pre_max;
        o_sig_min  = sig_min;  o_sig_max  = sig_max;
        o_post_min = post_min; o_post_max = post_max;
        tout->Fill();

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
            leg->AddEntry((TObject*)nullptr, Form("integral = %.2f", integral), "");
            
            leg->Draw();

            c->SaveAs(Form("plots/raw_wfs/wf_%lld.png", i));
            delete c; delete h;
            n_ch++;
        }
        
    }

    TCanvas* cp = new TCanvas("cp", "cp", 1200, 700);
    cp->SetLogz();              
    h2->Draw("colz");
    cp->SaveAs("plots/adc-baseline/ADC-baseline_1050.png");
    delete cp; delete h2;

    fout->cd();      
    tout->Write();   
    fout->Close();  

    delete run_calib;
    return 0;
}