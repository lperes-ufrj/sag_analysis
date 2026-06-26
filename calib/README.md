# calib

Calibration of DAPHNE waveforms from the PD-VD detector (run 039357).

## Input files

`*_gallery.root` — flat ROOT files produced by the gallery pipeline. Each file contains a `WaveformTree` with one entry per waveform:

| Branch | Type | Description |
|---|---|---|
| `run`, `subrun`, `event` | `int` | Event identifiers |
| `waveform_index` | `int` | Index of the waveform within the event |
| `channel` | `uint` | DAPHNE channel number |
| `timestamp` | `ull` | Hardware timestamp |
| `nsamples` | `uint` | Number of ADC samples |
| `adc` | `vector<short>` | Raw ADC samples |

## read_waveforms.cpp

Loops over all waveforms and computes the quantities below for each one. Results are saved to a `WaveformQuantities` TTree in the output ROOT file.

```bash
clang++ read_waveforms.cpp -o read_waveforms $(root-config --cflags --libs)
./read_waveforms input.root output.root
```

### Quantities

All amplitudes and integrals are **baseline-subtracted**.

| Branch | Parameters | Description |
|---|---|---|
| `baseline` | c=8, t=12 | Mode of the waveform, then running average over samples within [mode±c]; after each accepted sample, skip t ticks |
| `noise` | n=240 | Std-dev over the first 240 ticks |
| `integral` | [255, 265] | Sum of samples between ticks 255 and 265 |
| `presig_min/max` | [0, 239] | Min/max over first 240 ticks (pre-signal region) |
| `sig_min/max` | [300, 1023] | Min/max in the signal region |
| `postsig_min/max` | [350, 1023] | Min/max after the signal peak |
