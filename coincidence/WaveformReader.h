//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jun 25 11:07:04 2026 by ROOT version 6.36.04
// from TTree WaveformTree/ProtoDUNE-VD DAPHNE optical waveforms
// found on file: ../np02vd_raw_run039357_0000_df-s04-d0_dw_0_20250915T151645_gallery.root
//////////////////////////////////////////////////////////

#ifndef WaveformReader_h
#define WaveformReader_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"

class WaveformReader {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           run;
   Int_t           subrun;
   Int_t           event;
   Int_t           waveform_index;
   UInt_t          channel;
   ULong64_t       timestamp;
   UInt_t          nsamples;
   vector<short>   *adc;

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_subrun;   //!
   TBranch        *b_event;   //!
   TBranch        *b_waveform_index;   //!
   TBranch        *b_channel;   //!
   TBranch        *b_timestamp;   //!
   TBranch        *b_nsamples;   //!
   TBranch        *b_adc;   //!

   WaveformReader(TTree *tree=0);
   virtual ~WaveformReader();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop(unsigned int targetChannel = 1050, int modeHalfWidth = 8,
                         int skipTicks = 12,
                         double yMin = -50, double yMax = 200);
   virtual bool     Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef WaveformReader_cxx
WaveformReader::WaveformReader(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file(s)
// from the gallery folder and read the Tree. This will add all
// matching ROOT files (use wildcard) to a TChain so multiple files
// with the same tree structure are processed.
   if (tree == 0) {
      // Create a TChain to hold WaveformTree from multiple files.
      TChain *chain = new TChain("WaveformTree");
      // Add all gallery ROOT files in the parent directory that match
      // the original naming pattern. Adjust the pattern if needed.
      int n = chain->Add("../np02vd_raw_run039357_*_gallery.root");
      if (n == 0) {
         // fallback: try adding any ROOT files in parent dir
         chain->Add("../*.root");
      }
      tree = chain;
   }
   Init(tree);
}

WaveformReader::~WaveformReader()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t WaveformReader::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t WaveformReader::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void WaveformReader::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   adc = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("subrun", &subrun, &b_subrun);
   fChain->SetBranchAddress("event", &event, &b_event);
   fChain->SetBranchAddress("waveform_index", &waveform_index, &b_waveform_index);
   fChain->SetBranchAddress("channel", &channel, &b_channel);
   fChain->SetBranchAddress("timestamp", &timestamp, &b_timestamp);
   fChain->SetBranchAddress("nsamples", &nsamples, &b_nsamples);
   fChain->SetBranchAddress("adc", &adc, &b_adc);
   Notify();
}

bool WaveformReader::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return true;
}

void WaveformReader::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t WaveformReader::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef WaveformReader_cxx
