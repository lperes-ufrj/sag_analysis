#define WaveformReader_cxx
#include "WaveformReader.h"
#include <TCanvas.h>
#include <TH2D.h>
#include <TMath.h>
#include <TStyle.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <utility>

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

double ComputeAverageBaseline(const std::vector<short> &samples, int baselineSamples)
{
   const int nBaseline = std::min(std::max(1, baselineSamples),
                                  static_cast<int>(samples.size()));
   double sum = 0.0;
   for (int sample = 0; sample < nBaseline; ++sample) {
      sum += samples.at(sample);
   }
   return sum / nBaseline;
}

double mean(const std::vector<double> &vec)
{
   if (vec.empty()) {
      return 0.0;
   }

   double sum = 0.0;
   for (const auto value : vec) {
      sum += value;
   }
   return sum / vec.size();
}

double noise(const std::vector<short> &samples, double baseline, int tick_limit)
{
   if (samples.empty()) {
      return 0.0;
   }

   std::vector<double> adc_minus_baseline;

   const int maxSample = std::min(tick_limit, static_cast<int>(samples.size()));
   for (int sample = 0; sample < maxSample; ++sample) {
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

void WaveformReader::ScanCoincidence(const vector<unsigned int> &upperLeftChannels,
                                     const vector<unsigned int> &bottomRightChannels,
                                     bool printEvents,
                                     int sampleWindow,
                                     double threshold,
                                     int baselineSamples)
{
   if (fChain == 0) return;

   std::set<unsigned int> upperLeftSet(upperLeftChannels.begin(), upperLeftChannels.end());
   std::set<unsigned int> bottomRightSet(bottomRightChannels.begin(), bottomRightChannels.end());

   if (upperLeftSet.empty() || bottomRightSet.empty()) {
      std::cout << "Upper-left and bottom-right channel groups must both be non-empty."
                << std::endl;
      return;
   }

   struct PulseInfo {
      unsigned int channel;
      int peakSample;
      double amplitude;
   };

   struct EventPulses {
      std::vector<PulseInfo> upperLeft;
      std::vector<PulseInfo> bottomRight;
   };

   std::map<Int_t, EventPulses> pulsesByEvent;
   const Long64_t nentries = fChain->GetEntriesFast();
   const int coincidenceWindow = std::max(0, sampleWindow);

   for (Long64_t jentry = 0; jentry < nentries; ++jentry) {
      const Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      fChain->GetEntry(jentry);

      if (!adc || adc->empty()) continue;
      if (upperLeftSet.count(channel) == 0 && bottomRightSet.count(channel) == 0) continue;

      const double baseline = ComputeAverageBaseline(*adc, baselineSamples);
      int peakSample = 0;
      double amplitude = adc->at(0) - baseline;
      for (int sample = 1; sample < static_cast<int>(adc->size()); ++sample) {
         const double value = adc->at(sample) - baseline;
         if (value > amplitude) {
            amplitude = value;
            peakSample = sample;
         }
      }

      if (amplitude <= threshold) continue;

      const PulseInfo pulse = {channel, peakSample, amplitude};
      if (upperLeftSet.count(channel) > 0) {
         pulsesByEvent[event].upperLeft.push_back(pulse);
      }
      if (bottomRightSet.count(channel) > 0) {
         pulsesByEvent[event].bottomRight.push_back(pulse);
      }
   }

   int nCoincidentEvents = 0;
   int nCoincidentPairs = 0;
   for (const auto &eventInfo : pulsesByEvent) {
      const EventPulses &pulses = eventInfo.second;
      if (pulses.upperLeft.empty() || pulses.bottomRight.empty()) continue;

      int eventPairs = 0;
      PulseInfo firstUpperLeftPulse = {};
      PulseInfo firstBottomRightPulse = {};
      for (const auto &upperLeftPulse : pulses.upperLeft) {
         for (const auto &bottomRightPulse : pulses.bottomRight) {
            if (std::abs(upperLeftPulse.peakSample - bottomRightPulse.peakSample)
                <= coincidenceWindow) {
               if (eventPairs == 0) {
                  firstUpperLeftPulse = upperLeftPulse;
                  firstBottomRightPulse = bottomRightPulse;
               }
               ++eventPairs;
            }
         }
      }

      if (eventPairs == 0) continue;

      ++nCoincidentEvents;
      nCoincidentPairs += eventPairs;
      if (!printEvents) continue;

      std::cout << "Coincidence event " << eventInfo.first
                << " has " << eventPairs << " valid pairs. Example: upper-left ch "
                << firstUpperLeftPulse.channel
                << " peak " << firstUpperLeftPulse.peakSample
                << " amp " << firstUpperLeftPulse.amplitude
                << " with bottom-right ch " << firstBottomRightPulse.channel
                << " peak " << firstBottomRightPulse.peakSample
                << " amp " << firstBottomRightPulse.amplitude
                << std::endl;
   }

   std::cout << "Found " << nCoincidentEvents
             << " coincident events and " << nCoincidentPairs
             << " coincident channel pairs. Condition: same event, one upper-left channel, "
             << "one bottom-right channel, |peak sample difference| <= "
             << coincidenceWindow << ", and both amplitudes > " << threshold
             << ". Baseline uses the first " << std::max(1, baselineSamples)
             << " samples." << std::endl;
}
