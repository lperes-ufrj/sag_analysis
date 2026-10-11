"""Deconvolution parameters that fit/ also needs to know.

Deliberately free of heavy dependencies (no uproot): fit/ imports from here
instead of importing the whole deconvolve.deconvolve module.
"""
from calib.aux.waveform import TICK_NS  # noqa: F401  (re-exported: ns per tick)

# sigma (in ticks) of the Gaussian filter applied in the frequency domain:
# G[f] = exp(-2 (pi f sigma)^2). Equivalent to convolving s(t) with a Gaussian
# of sigma ticks (1 tick = 16 ns).
#
# From deconvolve/sigma_scan.py, the pre-signal noise RMS of s(t) falls fast
# with sigma (11.3 -> 1.5 -> 0.65 -> 0.39 -> 0.28 -> 0.22 p.e./tick for
# sigma = 1..6) while the peak amplitude falls roughly like 1/sigma. sigma = 6
# is the smoothest end of the scan: cleanest tail, least resolution.
#
# The total number of p.e. does not depend on sigma at all (G[0] = 1); only
# the shape and the noise floor do.
GAUSS_SIGMA_TICKS = 6

# Relative uncertainty assigned to each sample of the coincidence aggregate,
# while the coincidence code does not yet write a per-sample error column.
# sigma_v[i] = MEAN_REL_ERR * |v[i]|, taken as independent between samples.
MEAN_REL_ERR = 0.10
