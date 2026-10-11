"""Deconvolution core: template alignment and the FFT itself.

Kept free of uproot and of any I/O convention so that every entry point
(deconvolve.py on the ROOT files, deconvolve_coincidence.py on the
coincidence CSVs, compare_deconv_templates.py) shares the same math.
"""
import re
from pathlib import Path

import numpy as np


def align_template(template, v):
    """Shift the template peak onto the peak of v (np.roll), to avoid the
    circular offset in the FFT deconvolution.

    Returns (template_aligned, shift, peak_v, peak_template).
    """
    pk_v = int(np.argmax(v))
    pk_t = int(np.argmax(template))
    shift = pk_v - pk_t
    return np.roll(template, shift), shift, pk_v, pk_t


def deconvolve_fft(v, template_aligned, gauss_sigma_ticks):
    """v(t) = h(t) * L(t)  ->  s[f] = V[f] * G[f] / H[f]

    G[f] = exp(-2 (pi f sigma)^2) tames the noise amplification of 1/H[f].
    The template is normalized to 1 SPE, so s(t) comes out in p.e./tick and
    sum(s) is the total number of p.e.
    """
    n = len(v)
    V = np.fft.rfft(v)
    H = np.fft.rfft(template_aligned)
    freqs = np.fft.rfftfreq(n)                      # in 1/tick
    G = np.exp(-2.0 * (np.pi * freqs * gauss_sigma_ticks) ** 2)

    s = np.fft.irfft(V * G / H, n=n)
    # peaks aligned => t0 of L(t) lands at zero lag (window edge);
    # roll it back onto the data time axis
    return np.roll(s, int(np.argmax(v)))


def deconvolution_kernel(template_aligned, gauss_sigma_ticks, n):
    """Impulse response k(t) of the deconvolution filter: rfft(k) = G[f]/H[f].

    The whole deconvolution is the circular convolution s_raw = v (*) k, which
    is what makes the linear error propagation below exact.
    """
    freqs = np.fft.rfftfreq(n)
    G = np.exp(-2.0 * (np.pi * freqs * gauss_sigma_ticks) ** 2)
    H = np.fft.rfft(template_aligned)
    return np.fft.irfft(G / H, n=n)


def deconvolve_fft_err(v, template_aligned, gauss_sigma_ticks, sigma_v):
    """Per-tick uncertainty of s(t), propagated from the uncertainty of v(t).

    The deconvolution is linear: s_raw[i] = sum_j k[i-j] v[j], so with the
    input errors taken as independent between samples

        Var(s_raw)[i] = sum_j k[i-j]^2 sigma_v[j]^2  =  (k^2 (*) sigma_v^2)[i]

    -- one more circular convolution, done with the same FFT. The result is
    rolled by the same amount as deconvolve_fft, so it lines up with s(t).

    Independence between samples is the weak assumption here: a common scale
    error on the aggregate would propagate as a pure scale error on s(t)
    instead, and would not broaden the per-tick band at all.
    """
    n = len(v)
    k = deconvolution_kernel(template_aligned, gauss_sigma_ticks, n)
    var = np.fft.irfft(np.fft.rfft(k ** 2)
                       * np.fft.rfft(np.asarray(sigma_v, dtype=float) ** 2), n=n)
    var = np.clip(var, 0.0, None)          # kill the FFT round-off negatives
    return np.sqrt(np.roll(var, int(np.argmax(v))))


# --- coincidence CSVs produced by coincidence/plot_wfs_coincidence.cpp ------
#
# One row per sample: sample,mean,median,percentile_16,percentile_84
# These are already aggregated -- the individual waveforms live in the gallery
# ROOT files listed in coincidence/input_lists/ (on /pnfs), not in the repo.

COINC_NAME_RE = re.compile(r"channel_(\d+)_run_(\d+)_")


def parse_coincidence_name(path):
    """('channel_2080_run_039500_...csv') -> (2080, '039500')."""
    m = COINC_NAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot read channel/run from filename: {Path(path).name}")
    return int(m.group(1)), m.group(2)


def read_coincidence_csv(path):
    """Returns a dict with the sample axis and every aggregate column."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    missing = [c for c in ("sample", "mean", "median") if c not in d.dtype.names]
    if missing:
        raise ValueError(f"{Path(path).name}: missing columns {missing}")
    return {name: np.asarray(d[name], dtype=float) for name in d.dtype.names}
