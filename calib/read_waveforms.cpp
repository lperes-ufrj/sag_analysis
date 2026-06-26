// Reads WaveformTree and computes per-waveform quantities.
// Also saves a 2D heatmap of selected waveforms (integral > 3σ) as filtered_waveforms.png.
// Compile: clang++ read_waveforms.cpp -o read_waveforms $(root-config --cflags --libs)
// Run:     ./read_waveforms input.root [output.root]

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <climits>

#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"

static const int N1_INTEGRAL    = 255,  N2_INTEGRAL    = 265;
static const int C_BASELINE     = 8,    T_BASELINE     = 12;
static const int N_NOISE        = 240;
static const int N_PRESIGNAL    = 240;
static const int N1_SIGNAL      = 300,  N2_SIGNAL      = 1023;
static const int N1_POSTSIGNAL  = 350,  N2_POSTSIGNAL  = 1023;

static const double SIGNAL_SNR  = 3.0;  // integral > SIGNAL_SNR * noise * sqrt(window)

short mode_of(const std::vector<short>& v)
{
    std::map<short, int> cnt;
    for (short x : v) cnt[x]++;
    return std::max_element(cnt.begin(), cnt.end(),
        [](auto& a, auto& b){ return a.second < b.second; })->first;
}

// Walk from tick 0: if sample is within [mode-c, mode+c] add to average and
// skip t ticks; otherwise just advance. Avoids picking up signal samples.
double compute_baseline(const std::vector<short>& adc, int c, int t)
{
    short mode = mode_of(adc);
    double sum = 0.0;
    int cnt = 0, i = 0, sz = (int)adc.size();
    while (i < sz) {
        if (adc[i] >= mode - c && adc[i] <= mode + c) {
            sum += adc[i];
            cnt++;
            i += t;
        } else {
            i++;
        }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

double compute_integral(const std::vector<short>& adc, double baseline, int n1, int n2)
{
    double s = 0.0;
    int sz = (int)adc.size();
    for (int i = n1; i <= n2 && i < sz; ++i)
        s += adc[i] - baseline;
    return s;
}

double compute_noise(const std::vector<short>& adc, double baseline, int n)
{
    int sz = std::min(n, (int)adc.size());
    if (sz < 2) return 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < sz; ++i) {
        double d = adc[i] - baseline;
        sum2 += d * d;
    }
    return std::sqrt(sum2 / sz);
}

void compute_minmax(const std::vector<short>& adc, double baseline,
                    int i0, int i1, double& vmin, double& vmax)
{
    vmin =  1e9;
    vmax = -1e9;
    int sz = (int)adc.size();
    for (int i = i0; i <= i1 && i < sz; ++i) {
        double v = adc[i] - baseline;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.root> [output.root]\n";
        return 1;
    }
    std::string outpath = (argc >= 3) ? argv[2] : "waveform_quantities.root";

    TFile* fin = TFile::Open(argv[1], "READ");
    if (!fin || fin->IsZombie()) { std::cerr << "Cannot open " << argv[1] << "\n"; return 1; }

    TTree* tin = nullptr;
    fin->GetObject("WaveformTree", tin);
    if (!tin) { std::cerr << "WaveformTree not found.\n"; return 1; }

    int run = 0, subrun = 0, event = 0, waveform_index = 0;
    unsigned int channel = 0, nsamples = 0;
    unsigned long long timestamp = 0;
    std::vector<short>* adc = nullptr;

    tin->SetBranchAddress("run",            &run);
    tin->SetBranchAddress("subrun",         &subrun);
    tin->SetBranchAddress("event",          &event);
    tin->SetBranchAddress("waveform_index", &waveform_index);
    tin->SetBranchAddress("channel",        &channel);
    tin->SetBranchAddress("timestamp",      &timestamp);
    tin->SetBranchAddress("nsamples",       &nsamples);
    tin->SetBranchAddress("adc",            &adc);

    // First pass: find ADC range and waveform length for histogram axes
    int adc_min = INT_MAX, adc_max = INT_MIN, max_samples = 0;
    Long64_t nentries = tin->GetEntries();
    std::cout << "Scanning range (" << nentries << " waveforms)...\n";
    for (Long64_t i = 0; i < nentries; ++i) {
        tin->GetEntry(i);
        if (!adc || adc->empty()) continue;
        max_samples = std::max(max_samples, (int)adc->size());
        for (short v : *adc) {
            if (v < adc_min) adc_min = v;
            if (v > adc_max) adc_max = v;
        }
    }
    std::cout << "Samples: " << max_samples
              << "  ADC range: [" << adc_min << ", " << adc_max << "]\n";

    int ybins = std::min(adc_max - adc_min + 1, 1000);
    TH2F* h = new TH2F("h", ";Sample;ADC",
                        max_samples, -0.5, max_samples - 0.5,
                        ybins, adc_min - 0.5, adc_max + 0.5);

    // Output ROOT file with quantities
    TFile* fout = TFile::Open(outpath.c_str(), "RECREATE");
    TTree* tout = new TTree("WaveformQuantities", "Per-waveform computed quantities");

    int   o_run, o_subrun, o_event, o_wf_index;
    unsigned int o_channel;
    double o_baseline, o_noise, o_integral;
    double o_presig_min, o_presig_max;
    double o_sig_min,    o_sig_max;
    double o_postsig_min, o_postsig_max;

    tout->Branch("run",            &o_run);
    tout->Branch("subrun",         &o_subrun);
    tout->Branch("event",          &o_event);
    tout->Branch("waveform_index", &o_wf_index);
    tout->Branch("channel",        &o_channel);
    tout->Branch("baseline",       &o_baseline);
    tout->Branch("noise",          &o_noise);
    tout->Branch("integral",       &o_integral);
    tout->Branch("presig_min",     &o_presig_min);
    tout->Branch("presig_max",     &o_presig_max);
    tout->Branch("sig_min",        &o_sig_min);
    tout->Branch("sig_max",        &o_sig_max);
    tout->Branch("postsig_min",    &o_postsig_min);
    tout->Branch("postsig_max",    &o_postsig_max);

    // Second pass: compute quantities and fill heatmap for selected waveforms
    std::cout << "Processing...\n";
    long long n_selected = 0;
    double window = std::sqrt(N2_INTEGRAL - N1_INTEGRAL + 1.0);

    for (Long64_t i = 0; i < nentries; ++i) {
        tin->GetEntry(i);
        if (!adc || adc->empty()) continue;

        double baseline = compute_baseline(*adc, C_BASELINE, T_BASELINE);
        double noise    = compute_noise(*adc, baseline, N_NOISE);
        double integral = compute_integral(*adc, baseline, N1_INTEGRAL, N2_INTEGRAL);

        double presig_min, presig_max;
        compute_minmax(*adc, baseline, 0, N_PRESIGNAL - 1, presig_min, presig_max);

        double sig_min, sig_max;
        compute_minmax(*adc, baseline, N1_SIGNAL, N2_SIGNAL, sig_min, sig_max);

        double postsig_min, postsig_max;
        compute_minmax(*adc, baseline, N1_POSTSIGNAL, N2_POSTSIGNAL, postsig_min, postsig_max);

        o_run         = run;
        o_subrun      = subrun;
        o_event       = event;
        o_wf_index    = waveform_index;
        o_channel     = channel;
        o_baseline    = baseline;
        o_noise       = noise;
        o_integral    = integral;
        o_presig_min  = presig_min;   o_presig_max  = presig_max;
        o_sig_min     = sig_min;      o_sig_max     = sig_max;
        o_postsig_min = postsig_min;  o_postsig_max = postsig_max;

        tout->Fill();

        if (integral > SIGNAL_SNR * noise * window) {
            for (int s = 0; s < (int)adc->size(); ++s)
                h->Fill(s, (*adc)[s]);
            n_selected++;
        }

        if (i % 10000 == 0)
            std::cout << "  " << i << " / " << nentries << "\r" << std::flush;
    }
    std::cout << "\nSelected: " << n_selected << " / " << nentries << " waveforms\n";

    fout->cd();
    tout->Write();
    fout->Close();

    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBird);
    TCanvas* c = new TCanvas("c", "Filtered Waveforms", 1600, 600);
    c->SetRightMargin(0.12);
    h->SetY
    h->Draw("COLZ");
    c->SaveAs("filtered_waveforms.png");
    std::cout << "Saved: filtered_waveforms.png\n";
    std::cout << "Done. Written: " << outpath << "\n";

    fin->Close();
    return 0;
}
