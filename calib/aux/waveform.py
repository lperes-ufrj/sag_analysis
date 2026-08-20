"""Waveform helpers shared by calibration, deconvolution and fit.

Extracted from the SPE template builder so that deconvolve/ and fit/ (at the
repository root) can use the same robust-average and persistence recipes
without importing the template module.
"""
import numpy as np
import matplotlib.pyplot as plt

MAX_WF_PERSIST = 20000  # subsample for persistence plots (memory/time)


def robust_template(wf_sel):
    """
    "3 STEP" from the slides: a plain mean does not work (light noise).
    Per tick: robust center -> sigma = 0.5 * std about that center ->
    mean of the values within [center - sigma, center + sigma].

    Center = median (not mode): in 1 ADC bins the mode flips between discrete
    levels whenever the baseline falls between two ADC counts, injecting ~1 ADC
    of quantization noise into the template.
    """
    center = np.median(wf_sel, axis=0)                               # (n_samples,)
    sigma = 0.5 * np.sqrt(np.mean((wf_sel - center) ** 2, axis=0))   # (n_samples,)
    mask = np.abs(wf_sel - center) <= sigma
    n_in = mask.sum(axis=0)
    template = np.where(
        n_in > 0,
        (wf_sel * mask).sum(axis=0) / np.maximum(n_in, 1),
        center)
    return template.astype(np.float64)


def subsample(wf, rng):
    if wf.shape[0] > MAX_WF_PERSIST:
        idx = rng.choice(wf.shape[0], MAX_WF_PERSIST, replace=False)
        return wf[idx], True
    return wf, False


def persistence_hist(ax, wf, title, rng, yrange=(-50, 200)):
    """Draw a 2D persistence on an Axes; returns the hist2d for the colorbar."""
    wf_p, sub = subsample(wf, rng)
    n_events, n_samples = wf_p.shape
    x = np.tile(np.arange(n_samples), n_events)
    h = ax.hist2d(x, wf_p.flatten(), bins=[n_samples, 250],
                  range=[[0, n_samples], list(yrange)],
                  norm="log", cmap="viridis")
    sub_txt = f" (subsample {MAX_WF_PERSIST})" if sub else ""
    ax.set_title(f"{title}{sub_txt}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("ADC - baseline")
    return h


def persistence_plot(wf, title, path, rng, yrange=(-50, 200)):
    fig, ax = plt.subplots()
    h = persistence_hist(ax, wf, title, rng, yrange)
    fig.colorbar(h[3], ax=ax, label="Counts")
    fig.savefig(path)
    plt.close(fig)
