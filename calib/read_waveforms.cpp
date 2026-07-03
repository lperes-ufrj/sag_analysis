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
#include <string>

/*
    Channels of interest — SPE calibration (run 039357)

    Larsoft channel map: code Y0NX
        Y = module type (1 = C, 2 = M)
        N = module number
        X = SiPM array (0 or 1)
        channel = 2000 + N*10 + X   (M modules)

    Calibrating X-ARAPUCAs M7 and M8, both SiPM arrays:
        M7 ch0 -> 2070   (quartz filter, pTP inside)
        M7 ch1 -> 2071
        M8 ch0 -> 2080   (glass, pTP coated)
        M8 ch1 -> 2081
 */


const int CHANNELS[] = {2080, 2081};//, 2070, 2071};  
const int NCH = sizeof(CHANNELS) / sizeof(CHANNELS[0]);

int channel_index(unsigned int ch){
    for (int k = 0; k < NCH; k++)
        if (CHANNELS[k] == ch) return k;
    return -1;
}

// per-channel data: trees, persistence histograms and the branch variables
struct ChannelData {
    int channel;
    int n_saved = 0;   // example waveforms already saved for this channel

    TTree* traw;
    TTree* tsel;
    TH2F*  h_raw;
    TH2F*  h_sel;

    short  o_mode;
    double o_baseline, o_noise, o_integral;
    double o_pre_min, o_pre_max, o_sig_min, o_sig_max, o_post_min, o_post_max;
    std::vector<double> o_wf_nobaseline;   // baseline-subtracted waveform (selected only)
};

// attach the same set of metric branches to a tree
void book_metrics(TTree* t, ChannelData& d){
    t->Branch("mode",     &d.o_mode);
    t->Branch("baseline", &d.o_baseline);
    t->Branch("noise",    &d.o_noise);
    t->Branch("integral", &d.o_integral);
    t->Branch("pre_min",  &d.o_pre_min);
    t->Branch("pre_max",  &d.o_pre_max);
    t->Branch("sig_min",  &d.o_sig_min);
    t->Branch("sig_max",  &d.o_sig_max);
    t->Branch("post_min", &d.o_post_min);
    t->Branch("post_max", &d.o_post_max);
}

int main(){

    TChain* run_calib = new TChain("WaveformTree");

    // user input run number
    std::string run;
    std::cout << "Enter run number: ";
    std::cin >> run;

    std::string input_pattern = ".roots/17438/np02vd_raw_run" + run + "*_gallery.root";

    run_calib->Add(input_pattern.c_str()); // read all root files in order

    // see amount of entries
    const Long64_t nEntries = run_calib->GetEntries();

    if (nEntries == 0) {
        std::cerr << "Null Entry\n";
        return 1;
    }
    std::cout << nEntries << " entries read\n";


    std::vector<short>* adc = nullptr;   // vector<short>
    unsigned int channel  = 0;           // /i

    run_calib->SetBranchAddress("channel", &channel);
    run_calib->SetBranchAddress("adc", &adc);

    // output files: one tree per channel inside each
    std::string out_raw = ".roots/metrics_raw/run" + run + "_metrics_raw.root";
    std::string out_sel = ".roots/metrics_select/run" + run + "_metrics_select.root";

    TFile* fraw = new TFile(out_raw.c_str(), "RECREATE");
    TFile* fsel = new TFile(out_sel.c_str(), "RECREATE");

    ChannelData ch[NCH];
    for (int k = 0; k < NCH; k++){
        ch[k].channel = CHANNELS[k];

        ch[k].h_raw = new TH2F(Form("h_raw_%d",  CHANNELS[k]),
            Form("channel %d;sample;ADC - baseline (raw)",      CHANNELS[k]),
            1024, 0, 1024, 250, -50, 200);
        ch[k].h_sel = new TH2F(Form("h_sel_%d",  CHANNELS[k]),
            Form("channel %d;sample;ADC - baseline (selected)", CHANNELS[k]),
            1024, 0, 1024, 250, -50, 200);
        ch[k].h_raw->SetDirectory(nullptr);
        ch[k].h_sel->SetDirectory(nullptr);

        fraw->cd();
        ch[k].traw = new TTree(Form("Metrics_%d", CHANNELS[k]), "per-waveform metrics, raw");
        book_metrics(ch[k].traw, ch[k]);

        fsel->cd();
        ch[k].tsel = new TTree(Form("Metrics_%d", CHANNELS[k]), "per-waveform metrics, selected");
        book_metrics(ch[k].tsel, ch[k]);
        // selected tree also stores the baseline-subtracted waveform
        ch[k].tsel->Branch("wf_nobaseline", &ch[k].o_wf_nobaseline);
    }

    for (Long64_t i = 0; i < nEntries; ++i){
        run_calib->GetEntry(i);

        if (!adc || adc->empty()) continue;

        // check wanted channels
        int idx = channel_index(channel);
        if (idx < 0) continue;

        int n = adc->size();
        ChannelData& d = ch[idx];   // current channel's data

        // compute variables
        short mode = mode_of(*adc);
        double baseline = compute_baseline(*adc, 8, 12, mode);
        double std = noise(*adc, 240, baseline);
        double integral = compute_integral(*adc, baseline, 255, 265);
        double pre_min, pre_max, sig_min, sig_max, post_min, post_max;
        compute_minmax(*adc, baseline, 0,   239,  pre_min,  pre_max);   // pre-signal (n=240)
        compute_minmax(*adc, baseline, 300, 1023, sig_min,  sig_max);   // signal
        compute_minmax(*adc, baseline, 350, 1023, post_min, post_max);  // post-signal

        // fill this channel's metric variables
        d.o_mode     = mode;
        d.o_baseline = baseline;
        d.o_noise    = std;
        d.o_integral = integral;
        d.o_pre_min  = pre_min;  d.o_pre_max  = pre_max;
        d.o_sig_min  = sig_min;  d.o_sig_max  = sig_max;
        d.o_post_min = post_min; d.o_post_max = post_max;
        d.traw->Fill();

        if(std < 20 && sig_max < 150 && sig_min > -50 && pre_max < 50 && pre_min > -40 && post_max < 50 && post_min > -40){
            // baseline-subtracted waveform stored in the selected tree
            d.o_wf_nobaseline.assign(n, 0.0);
            for (int j = 0; j < n; j++)
                d.o_wf_nobaseline[j] = (*adc)[j] - baseline;
            d.tsel->Fill();

            for (int j = 0; j < n; j++)
                d.h_sel->Fill(j, (*adc)[j] - baseline);
        }

        for (int s = 0; s < n; s++)
            d.h_raw->Fill(s, (*adc)[s] - baseline);

        // confirmation: save 3 example waveforms per channel
        if (d.n_saved < 3) {
            std::cout << "entry" << i
            << " channel=" << channel
            << " nsamples=" << adc->size()
            << " adc[255]=" << (*adc)[255] << "\n";

            std::cout << " moda" << mode
            << " baseline" << baseline
            << " noise" << std << "\n";

            TH1D* h = new TH1D(Form("wf_%d_%lld", d.channel, i),
                               Form("channel %d entry %lld;sample;ADC", d.channel, i),
                               n, 0, n);
            h->SetDirectory(nullptr);
            for (int s=0; s < n; s++)
                h->SetBinContent(s+1, (*adc)[s]);

            // TCanvas* c = new TCanvas("c", "c", 1200, 400);
            // h->Draw("L");
            // h->SetStats(0);
            // TLegend* leg = new TLegend(0.65, 0.7, 0.88, 0.88); // x1, y1, x2, y2
            // leg->AddEntry((TObject*)nullptr, Form("moda = %d", mode), "");
            // leg->AddEntry((TObject*)nullptr, Form("baseline = %.1f", baseline), "");
            // leg->AddEntry((TObject*)nullptr, Form("noise = %.2f", std), "");
            // leg->AddEntry((TObject*)nullptr, Form("integral = %.2f", integral), "");
            // leg->Draw();

            // std::string plot_dir = "plots/raw_wfs/run" + run + "_" + std::to_string(d.channel) + "_wf_" + std::to_string(i) + ".png";   
            // c->SaveAs(plot_dir.c_str());
            // delete c; delete h;
            d.n_saved++;
        }


    }

    // persistence plots: raw and selected for each channel
    gStyle->SetOptStat(0); // no stats on plots
    for (int k = 0; k < NCH; k++){
        TCanvas* cp = new TCanvas("cp", "cp", 1200, 700);
        cp->SetLogz();
        ch[k].h_raw->Draw("colz");
        cp->SaveAs(Form("plots/adc-baseline/run%s_%d_ADC-baseline_raw.png", run.c_str(), CHANNELS[k]));
        delete cp; delete ch[k].h_raw;

        TCanvas* cp2 = new TCanvas("cp2", "cp2", 1200, 700);
        cp2->SetLogz();
        ch[k].h_sel->Draw("colz");
        cp2->SaveAs(Form("plots/adc-baseline/run%s_%d_ADC-baseline_selec.png", run.c_str(), CHANNELS[k]));
        delete cp2; delete ch[k].h_sel;
    }

    fraw->cd();
    for (int k = 0; k < NCH; k++) ch[k].traw->Write();
    fraw->Close();

    fsel->cd();
    for (int k = 0; k < NCH; k++) ch[k].tsel->Write();
    fsel->Close();

    delete run_calib;
    return 0;
}
