"""Deconvolution parameters that fit/ also needs to know.

Deliberately free of heavy dependencies (no uproot): fit/ imports from here
instead of importing the whole deconvolve.deconvolve module.
"""

# sigma (in ticks) of the Gaussian filter applied in the frequency domain:
# G[f] = exp(-2 (pi f sigma)^2). Equivalent to convolving s(t) with a Gaussian
# of sigma ticks (1 tick = 16 ns; sigma=4 ~ 64 ns, compatible with the
# sigma ~62 ns from Gabriel's fit).
GAUSS_SIGMA_TICKS = 4

TICK_NS = 16.0   # ns per tick (DAPHNE @ 62.5 MHz)
