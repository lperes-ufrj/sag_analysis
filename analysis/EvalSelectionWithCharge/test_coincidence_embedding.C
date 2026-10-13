// ROOT-only regression for the exact scanner embedded by the Gallery display.
#define COINCIDENCE_NO_MAIN
#include "../../coincidence/run_coincidence.cpp"
#undef COINCIDENCE_NO_MAIN
#include "../../coincidence/coincidence_lib.cpp"
#include "TFile.h"
#include "TSystem.h"

void test_coincidence_embedding()
{
    const auto directory = fs::temp_directory_path() / ("pds_selection_test_" + std::to_string(gSystem->GetPid()));
    if (!fs::create_directory(directory)) throw std::runtime_error("Test directory already exists");
    const auto input = directory/"input.root", csv = directory/"selected.csv";
    {
        TFile file(input.c_str(), "CREATE");
        TTree tree("WaveformTree", "Boundary fixtures");
        Int_t event, index; UInt_t channel; ULong64_t timestamp;
        std::vector<short> adc;
        tree.Branch("event", &event); tree.Branch("channel", &channel);
        tree.Branch("waveform_index", &index); tree.Branch("timestamp", &timestamp);
        tree.Branch("adc", &adc);
        auto add = [&](int e, int i, unsigned ch, int relativeNs, bool pulse = true) {
            event=e; index=i; channel=ch; timestamp=1000000+relativeNs;
            adc.assign(1024,100);
            for (int n=0;n<50;++n) adc[n] = n%2 ? 101 : 99;
            if (pulse) adc[60]=200;
            tree.Fill();
        };
        // Inclusive +160 ns pair and +160 ns saved-waveform boundaries.
        add(1,0,2030,0); add(1,1,2050,160); add(1,2,2070,320);
        // One sample outside pair tolerance.
        add(2,0,2030,0); add(2,1,2050,176); add(2,2,2070,0);
        // One ns outside saved-waveform tolerance.
        add(3,0,2030,0); add(3,1,2050,160); add(3,2,2070,321);
        // Inclusive negative saved-waveform boundary.
        add(4,0,2030,0); add(4,1,2050,160); add(4,2,2070,-160);
        // Zero baseline-adjusted amplitude must be rejected.
        add(5,0,2030,0,false); add(5,1,2050,0); add(5,2,2070,0);
        // Multiple valid pairs: closest minimum pulse time wins, one saved row.
        add(6,0,2030,0); add(6,1,2030,48); add(6,2,2050,160); add(6,3,2070,160);
        // Reverse channel ordering in time and inclusive -160 ns boundary.
        add(7,0,2030,160); add(7,1,2050,0); add(7,2,2070,-160);
        tree.Write();
    }
    Arguments args;
    args.config = fs::path(__FILE__).parent_path()/"../../coincidence/waveform_intervals.ini";
    WaveformAnalyzer analyzer(args.config.string());
    auto chain = createChain({input.string()});
    const auto stats = scanCoincidence(*chain, analyzer, args, {}, csv.string());
    if (stats.selected_events != 4 || stats.selected_waveforms != 4)
        throw std::runtime_error("Unexpected number of selected events/waveforms");
    std::ifstream result(csv); std::string line;
    std::getline(result,line);
    const std::vector<std::string> expected{
        "1,2070,2,160,320,1000000,2030,0,2050,1",
        "4,2070,2,160,-160,1000000,2030,0,2050,1",
        "6,2070,3,112,112,1000048,2030,1,2050,2",
        "7,2070,2,-160,-160,1000000,2030,0,2050,1"
    };
    for (const auto& row : expected) {
        if (!std::getline(result,line) || trim(line) != row)
            throw std::runtime_error("Unexpected selection row: " + line);
    }
    if (std::getline(result,line)) throw std::runtime_error("Unexpected extra row");
    chain.reset(); result.close();
    fs::remove(input); fs::remove(csv); fs::remove(directory);
    std::cout << "PASS: embedded scanner boundaries, amplitude and best-pair selection\n";
}
