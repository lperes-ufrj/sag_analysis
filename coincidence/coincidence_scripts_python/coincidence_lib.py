import configparser

import numpy as np
import ROOT


SAMPLE_PERIOD_NS = 16

def create_chain(files):
    chain = ROOT.TChain("WaveformTree")
    for file in files:
        chain.Add(file)
    return chain


class WaveformAnalyzer:
    def __init__(self, ini_file):
        self.config = configparser.ConfigParser()
        self.config.read(ini_file)

    def interval(self, channel, name):
        value = self.config[f"channel_{channel}"][name]
        start, stop = value.split(":")
        return int(start), int(stop)

    def region(self, waveform, channel, name):
        start, stop = self.interval(channel, name)
        return np.asarray(waveform)[start:stop]

    def baseline(self, waveform, c=8, t=12):

        """ Igual do Gabirel """
        samples = np.asarray(waveform)

        if samples.size == 0:
            raise ValueError("Cannot calculate baseline from an empty waveform")

        values, counts = np.unique(samples, return_counts=True)
        mode = values[np.argmax(counts)]

        lower = mode - c
        upper = mode + c
        stride = max(1, t)

        baseline_samples = []
        tick = 0

        while tick < samples.size:
            value = samples[tick]

            if lower <= value <= upper:
                baseline_samples.append(value)
                tick += stride
            else:
                tick += 1

        if not baseline_samples:
            return float(mode)

        return float(np.mean(baseline_samples))

    def noise_baseline(self, waveform, channel):
        noise = self.region(waveform, channel, "noise")
        if noise.size == 0:
            raise ValueError(
                f"Cannot calculate baseline from empty noise region for channel {channel}"
            )
        return float(np.median(noise))

    def amplitude(self, waveform, channel, name="signal"):
        region = self.region(waveform, channel, name)
        return np.max(region) - np.min(region)

    def signal_amplitude(self, waveform, channel):
        return self.amplitude(waveform, channel)

    def baseline_adjusted_signal_amplitude(self, waveform, channel):
        signal = self.region(waveform, channel, "signal")
        return float(np.max(signal)) - self.noise_baseline(waveform, channel)

    def pre_signal_amplitude(self, waveform, channel):
        return self.amplitude(waveform, channel, "pre_signal")

    def post_signal_amplitude(self, waveform, channel):
        return self.amplitude(waveform, channel, "post_signal")

    def signal_peak_tick(self, waveform, channel):
        start, _ = self.interval(channel, "signal")
        return start + int(np.argmax(self.region(waveform, channel, "signal")))

    def noise_rms(self, waveform, channel):
        return np.std(self.region(waveform, channel, "noise"))

    def pulse_start(self, waveform, channel):
        start, _ = self.interval(channel, "signal")
        signal = self.region(waveform, channel, "signal")
        threshold = self.noise_baseline(waveform, channel) + 10 * self.noise_rms(
            waveform, channel
        )

        tick = int(np.argmax(signal))
        while tick > 0 and signal[tick - 1] >= threshold:
            tick -= 1
        return start + tick

    def dt_ns_pulses(
        self,
        waveform_a,
        channel_a,
        timestamp_a,
        waveform_b,
        channel_b,
        timestamp_b,
    ):
        time_a = timestamp_a + self.pulse_start(waveform_a, channel_a) * SAMPLE_PERIOD_NS
        time_b = timestamp_b + self.pulse_start(waveform_b, channel_b) * SAMPLE_PERIOD_NS
        dt_ns = time_a - time_b
        return dt_ns

channels_dict = {
    "CH_C1_1": 1010, 
    "CH_C1_2": 1011, 
    "CH_C2_1": 1020, 
    "CH_C2_2": 1021, 
    "CH_C3_1": 1030, 
    "CH_C3_2": 1031, 
    "CH_C4_1": 1040, 
    "CH_C4_2": 1041,
    "CH_C5_1": 1050,
    "CH_C5_2": 1051,
    "CH_C6_1": 1060,
    "CH_C6_2": 1061,
    "CH_C7_1": 1070,
    "CH_C7_2": 1071,
    "CH_C8_1": 1080,
    "CH_C8_2": 1081,
    "CH_M1_1": 2010, 
    "CH_M1_2": 2011, 
    "CH_M2_1": 2020, 
    "CH_M2_2": 2021, 
    "CH_M3_1": 2030, 
    "CH_M3_2": 2031, 
    "CH_M4_1": 2040, 
    "CH_M4_2": 2041, 
    "CH_M5_1": 2050, 
    "CH_M5_2": 2051, 
    "CH_M6_1": 2060, 
    "CH_M6_2": 2061,
    "CH_M7_1": 2070,
    "CH_M7_2": 2071,
    "CH_M8_1": 2080,
    "CH_M8_2": 2081
}
