#define WaveformReader_cxx
#include "WaveformReader.h"
#include <TCanvas.h>
#include <TH2D.h>
#include <TMath.h>
#include <TStyle.h>
#include <algorithm>
#include <iostream>
#include <map>

namespace {
double ComputeBaseline(const std::vector<short> &samples, int modeHalfWidth, int skipTicks)
{
   std::map<short, int> counts;
   short mode = samples.front();
   int modeCount = 0;

   for (const short value : samples) {
      const int count = ++counts[value];
      if (count > modeCount) {
         mode = value;
         modeCount = count;
      }
   }

   const int halfWidth = std::max(0, modeHalfWidth);
   const int stride = std::max(1, skipTicks);
   const int low = static_cast<int>(mode) - halfWidth;
   const int high = static_cast<int>(mode) + halfWidth;

   double sum = 0.0;
   int nBaseline = 0;
   for (int sample = 0; sample < static_cast<int>(samples.size());) {
      const int value = samples.at(sample);
      if (value >= low && value <= high) {
         sum += value;
         ++nBaseline;
         sample += stride;
      } else {
         ++sample;
      }
   }

   return nBaseline > 0 ? sum / nBaseline : mode;
}

double mean(const std::vector<short> &vec){
   sum = 0;
   for(auto i : vec){
      sum +=i;
   }
   return mean_result / vec.size()
}

double noise(const std::vector<short> &samples, double baseline, int tick_limit)
{
   if (samples.empty()) {
      return 0.0;
   }

   std::vector<short> adc_minus_baseline;

   double sumSq = 0.0;
   for (size_t sample = 0; sample < tick_limit ; ++sample) {
      adc_minus_baseline.push_back(static_cast<double>(samples.at(sample)) - baseline);
   }
   double mean_adc = mean(adc_minus_baseline);
   double diff =0;
   for (auto value : adc_minus_baseline) {
      diff = diff + ((value - mean_adc)*(value - mean_adc));
   }

   return TMath::Sqrt(diff/static_cast<double>(samples.size()));
}

}



void WaveformReader::Loop(unsigned int targetChannel, int modeHalfWidth, int skipTicks,
                          double yMin, double yMax)
{
   if (fChain == 0) return;

   const Long64_t nentries = fChain->GetEntriesFast();
   int nWaveforms = 0;
   int maxSamples = 0;

   for (Long64_t jentry = 0; jentry < nentries; ++jentry) {
      const Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      fChain->GetEntry(jentry);

      if (channel != targetChannel) continue;
      if (!adc || adc->empty()) continue;

      ++nWaveforms;
      if (static_cast<int>(adc->size()) > maxSamples) maxSamples = adc->size();
   }

   if (nWaveforms == 0 || maxSamples == 0) {
      std::cout << "No waveform found for channel " << targetChannel << std::endl;
      return;
   }

   if (yMax <= yMin) {
      std::cout << "Invalid y-axis range: yMin = " << yMin
                << ", yMax = " << yMax << std::endl;
      return;
   }

   TH2D *hist = new TH2D(Form("h_adc_vs_sample_ch%u", targetChannel),
                         Form("ADC minus baseline, channel %u;Sample;ADC - baseline;Counts",
                              targetChannel),
                         maxSamples, 0, maxSamples,
                         100, yMin, yMax);

   for (Long64_t jentry = 0; jentry < nentries; ++jentry) {
      const Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      fChain->GetEntry(jentry);

      if (channel != targetChannel) continue;
      if (!adc || adc->empty()) continue;

      const double baseline = ComputeBaseline(*adc, modeHalfWidth, skipTicks);

      for (int sample = 0; sample < static_cast<int>(adc->size()); ++sample) {
         hist->Fill(sample, adc->at(sample) - baseline);
      }
   }

   gStyle->SetOptStat(0);
   TCanvas *canvas = new TCanvas(Form("c_adc_vs_sample_ch%u", targetChannel),
                                 Form("ADC minus baseline channel %u", targetChannel),
                                 1100, 700);
   hist->Draw("COLZ");
   canvas->Update();

   std::cout << "Filled ADC-minus-baseline histogram with " << nWaveforms
             << " waveforms for channel " << targetChannel
             << ". Baseline uses mode window +/-" << std::max(0, modeHalfWidth)
             << " ADC counts and skips " << std::max(1, skipTicks)
             << " samples. Y range is [" << yMin << ", " << yMax << "]."
             << std::endl;
}
