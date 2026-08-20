"""Deconvolution of the average physics waveform by the SPE template.

Analysis step 2: consumes the SPE template produced by calib/ and the
coincidence ROOT file, and delivers the deconvolved waveform in p.e./tick.

Usage:
    python deconvolve/deconvolve.py   (or: cd deconvolve && python deconvolve.py)

Outputs under deconvolve/output/:
    plots/filtering/      persistence, all vs filtered
    plots/average/        robust average of the physics signal
    plots/align/          template vs average overlay (shift check)
    plots/deconvolution/  deconvolved waveform (linear + log)
    data/avg_<ch>.npy               robust average
    data/template_aligned_<ch>.npy  template after np.roll
    data/deconv_<ch>.txt            <- consumed by fit/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uproot
import numpy as np
import matplotlib.pyplot as plt

from calib.aux import functions as fc
from calib.aux import paths
from calib.aux.spe_template import template_path
from calib.aux.waveform import robust_template, persistence_hist
from deconvolve.config import GAUSS_SIGMA_TICKS

DECONV_DIR = paths.DECONV_DIR

# coincidence ROOT file -- unchanged for now (calib/.roots/physics)
PHYS_FILE = (paths.ROOTS_DIR / "physics" /
             "coincident_waveforms_fast_window_20_ticks"
             "_coinc_adc_4000_save_adc_4000.root")

# dt_peak_ticks_save window per channel (M7: [-5,0], M8: [0,5])
DT_WINDOWS = {
    2070: (-5, 0),
    2071: (-5, 0),
    2080: (0, 5),
    2081: (0, 5),
}

N_BASELINE = 60      # pre-signal ticks for the median baseline (peak ~ tick 90)
PERSIST_YRANGE = (-500, 15000)

# quality cuts (applied after the dt cut):
#   peak_max: rejects saturation (plateau ~14k = 16383 - baseline)
#   pre_max:  max over the first N_BASELINE ticks (M8 has much larger
#             pre-signal light noise than M7, hence the looser threshold)
#   post_max: max after tick POST_START (the tail should already be gone)
PEAK_MAX = 10000
POST_START = 600
QUALITY_CUTS = {
    2070: {"pre_max": 30,  "post_max": 100},
    2071: {"pre_max": 30,  "post_max": 100},
    2080: {"pre_max": 150, "post_max": 100},
    2081: {"pre_max": 150, "post_max": 100},
}

# shape cut: rejects a secondary pulse riding on the tail, where an
# absolute-amplitude cut does not work. Each wf is normalized by its peak and
# compared to the channel's median shape; rejected if the residual inside
# SHAPE_WINDOW exceeds SHAPE_MAX_RESID (fraction of the main peak).
SHAPE_WINDOW = (200, 600)
SHAPE_MAX_RESID = 0.12

# calibration run providing the SPE template of each channel
TEMPLATE_RUNS = {
    2070: "039365",
    2071: "039365",
    2080: "039366",
    2081: "039366",
}

def deconv_txt_path(ch):
    """Canonical deconvolved-waveform path -- this is what fit/ reads."""
    return paths.out(DECONV_DIR, "data") / f"deconv_{ch}.txt"


def _select_channel(tree, ch, ch_arr, dt_arr):
    """Filter the wfs of one channel: baseline + dt + quality + shape.
    Returns (wf, sel, cutflow_str)."""
    dt_min, dt_max = DT_WINDOWS[ch]
    idx = np.where(ch_arr == ch)[0]
    if idx.size == 0:
        return None, None, None

    wf = np.stack(tree["adc"].array(library="np")[idx]).astype(np.float32)
    wf -= np.median(wf[:, :N_BASELINE], axis=1, keepdims=True)

    sel_dt = (dt_arr[idx] >= dt_min) & (dt_arr[idx] <= dt_max)

    qc = QUALITY_CUTS[ch]
    peak_arr = wf.max(axis=1)
    pre_arr  = wf[:, :N_BASELINE].max(axis=1)
    post_arr = wf[:, POST_START:].max(axis=1)
    sel = sel_dt & (peak_arr < PEAK_MAX) \
                 & (pre_arr < qc["pre_max"]) \
                 & (post_arr < qc["post_max"])

    # shape cut (reference = median of those that passed above)
    w_norm = wf[sel] / peak_arr[sel][:, None]
    ref = np.median(w_norm, axis=0)
    t0, t1 = SHAPE_WINDOW
    resid = (w_norm - ref)[:, t0:t1].max(axis=1)
    sel_idx = np.where(sel)[0][resid < SHAPE_MAX_RESID]
    n_dtq = int(sel.sum())
    sel = np.zeros_like(sel)
    sel[sel_idx] = True

    cutflow = (f"{wf.shape[0]} wfs | dt: {int(sel_dt.sum())} "
               f"| dt+quality: {n_dtq} | +shape: {int(sel.sum())}")
    return wf, sel, cutflow


def filter_and_average(phys_path=PHYS_FILE):
    """Select the physics wfs per channel, save the persistences
    (all vs filtered) under output/plots/filtering/ and the robust average
    of the filtered signal (same recipe as the SPE template: median ->
    sigma = 0.5*std -> mean within [median +- sigma]) under
    output/plots/average/.

    Returns {channel: avg_waveform}."""
    rng = np.random.default_rng(42)
    filt_dir = paths.out(DECONV_DIR, "plots", "filtering")
    avg_dir  = paths.out(DECONV_DIR, "plots", "average")
    data_dir = paths.out(DECONV_DIR, "data")

    f = uproot.open(phys_path)
    tree = f[fc.latest_cycle_keys(f)[0]]
    ch_arr = tree["saved_channel"].array(library="np")
    dt_arr = tree["dt_peak_ticks_save"].array(library="np")

    averages = {}   # {channel: avg_waveform}

    for ch in DT_WINDOWS:
        wf, sel, cutflow = _select_channel(tree, ch, ch_arr, dt_arr)
        if wf is None:
            print(f"Channel {ch}: no waveforms, skipping")
            continue

        dt_min, dt_max = DT_WINDOWS[ch]
        qc = QUALITY_CUTS[ch]

        # persistences: all vs filtered
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        h0 = persistence_hist(
            axes[0], wf, f"all ({wf.shape[0]} evts)",
            rng, yrange=PERSIST_YRANGE)
        h1 = persistence_hist(
            axes[1], wf[sel],
            f"dt in [{dt_min},{dt_max}], peak<{PEAK_MAX}, "
            f"pre<{qc['pre_max']}, post{POST_START}<{qc['post_max']}, "
            f"shape<{SHAPE_MAX_RESID} ({int(sel.sum())} evts)",
            rng, yrange=PERSIST_YRANGE)
        fig.colorbar(h0[3], ax=axes[0], label="Counts")
        fig.colorbar(h1[3], ax=axes[1], label="Counts")
        fig.suptitle(f"Ch {ch} - physics persistence")
        plt.tight_layout()
        fig.savefig(filt_dir / f"{ch}_persistence.png")
        plt.close(fig)

        # robust average of the filtered signal (same recipe as the template)
        avg = robust_template(wf[sel])
        averages[ch] = avg
        np.save(data_dir / f"avg_{ch}.npy", avg)

        plt.plot(avg)
        plt.xlim(0, 500)
        plt.xlabel("Sample")
        plt.ylabel("ADC - baseline")
        plt.title(f"Ch {ch} - average waveform ({int(sel.sum())} wfs)")
        plt.savefig(avg_dir / f"avg_{ch}.png")
        plt.close()

        print(f"Channel {ch}: {cutflow} | avg saved")

    return averages


def load_template(ch):
    """SPE template produced by calib/ (calib/output/data/templates/)."""
    return np.load(template_path(TEMPLATE_RUNS[ch], ch))


def align_templates(averages):
    """Find the peak of the DAQ average and of the SPE template, then shift
    the template (np.roll) onto the data peak, to avoid the circular offset
    in the FFT deconvolution. Saves a normalized overlay under
    output/plots/align/ for visual inspection.

    Returns {channel: aligned_template}."""
    outdir   = paths.out(DECONV_DIR, "plots", "align")
    data_dir = paths.out(DECONV_DIR, "data")

    aligned = {}
    for ch, avg in averages.items():
        tmpl = load_template(ch)
        pk_avg  = int(np.argmax(avg))
        pk_tmpl = int(np.argmax(tmpl))
        shift = pk_avg - pk_tmpl
        tmpl_al = np.roll(tmpl, shift)
        aligned[ch] = tmpl_al
        np.save(data_dir / f"template_aligned_{ch}.npy", tmpl_al)

        plt.plot(avg / avg.max(), label=f"avg physics (peak @ {pk_avg})")
        plt.plot(tmpl_al / tmpl_al.max(), alpha=0.8,
                 label=f"SPE template (peak @ {pk_tmpl}, shift {shift:+d})")
        plt.xlabel("Sample")
        plt.ylabel("normalized")
        plt.title(f"Ch {ch} - template alignment")
        plt.legend()
        plt.savefig(outdir / f"align_{ch}.png")
        plt.close()

        print(f"Channel {ch}: avg peak @ {pk_avg}, "
              f"template peak @ {pk_tmpl}, shift {shift:+d}")

    return aligned


def deconvolve_average(averages, aligned):
    """Deconvolution of the average physics signal by the SPE template
    (Gabriel's methodology, no fit):

        v(t) = h(t) * L(t)  ->  s[f] = V[f] * G[f] / H[f]

    with G[f] a Gaussian filter to tame the noise amplification of 1/H[f].
    s(t) comes out in p.e./tick (template = 1 SPE), so sum(s) ~ total
    number of p.e.

    Saves s(t) as txt (one value per line, %.9e) for fit/.
    Returns {channel: s(t)}."""
    outdir = paths.out(DECONV_DIR, "plots", "deconvolution")

    deconvolved = {}
    for ch, v in averages.items():
        h = aligned[ch]
        n = len(v)

        V = np.fft.rfft(v)
        H = np.fft.rfft(h)
        freqs = np.fft.rfftfreq(n)   # in 1/tick
        G = np.exp(-2.0 * (np.pi * freqs * GAUSS_SIGMA_TICKS) ** 2)

        s = np.fft.irfft(V * G / H, n=n)
        # peaks aligned => t0 of L(t) lands at zero lag (window edge);
        # roll it back onto the data time axis
        s = np.roll(s, int(np.argmax(v)))
        deconvolved[ch] = s
        paths.save_waveform_txt(deconv_txt_path(ch), s)

        npe = s.sum()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        ax1.plot(s)
        ax1.set_ylabel("p.e. / tick")
        ax1.set_title(f"Ch {ch} - deconvolved "
                      f"(sigma={GAUSS_SIGMA_TICKS} ticks, "
                      f"sum={npe:.0f} p.e.)")
        ax2.plot(s)
        ax2.set_yscale("log")
        ax2.set_ylim(bottom=1e0)
        ax2.set_ylabel("p.e. / tick (log)")
        ax2.set_xlabel("Sample")
        plt.tight_layout()
        plt.savefig(outdir / f"deconv_{ch}.png")
        plt.close(fig)

        print(f"Channel {ch}: deconvolved, sum = {npe:.0f} p.e. "
              f"-> {deconv_txt_path(ch).name}")

    return deconvolved


def main():
    averages    = filter_and_average()
    aligned     = align_templates(averages)
    deconvolved = deconvolve_average(averages, aligned)
    print(f"\nDeconvolved waveforms in {paths.out(DECONV_DIR, 'data')} "
          f"(deconv_<ch>.txt) -- next step: python fit/fit.py")
    return deconvolved


if __name__ == "__main__":
    main()
