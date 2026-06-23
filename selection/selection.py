import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

root_file = "pdvddaphne_run39523_0000_waveforms.root"

target_event = 1      # change this
target_channel = 0    # change this

with uproot.open(root_file) as f:
    tree = f["WaveformTree"]

    arrays = tree.arrays(
        ["run", "subrun", "event", "waveform_index",
         "channel", "timestamp", "nsamples", "adc"],
        library="ak"
    )

mask = (
    (arrays["event"] == target_event) &
    (arrays["channel"] == target_channel)
)

matches = arrays[mask]

print(f"Matching waveforms: {len(matches)}")

if len(matches) == 0:
    print("No waveform found for this event/channel combination.")
else:
    for i in range(len(matches)):
        adc = np.asarray(matches["adc"][i])

        print(
            f"Match {i}: "
            f"run={matches['run'][i]}, "
            f"subrun={matches['subrun'][i]}, "
            f"event={matches['event'][i]}, "
            f"channel={matches['channel'][i]}, "
            f"timestamp={matches['timestamp'][i]}, "
            f"nsamples={matches['nsamples'][i]}"
        )

        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(adc)), adc)
        plt.xlabel("Sample")
        plt.ylabel("ADC")
        plt.title(
            f"Run {matches['run'][i]}, Event {matches['event'][i]}, "
            f"Channel {matches['channel'][i]}"
        )
        plt.grid()
        plt.tight_layout()
        plt.show()