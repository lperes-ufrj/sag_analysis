"""SPE template construction from the fingerplot peaks.

Calibration step 2: takes the per-channel multigauss fits, selects the
waveforms under the 1..N_PEAKS p.e. peaks, applies the quality cuts and
returns the template normalized to 1 SPE.
"""
import numpy as np
import matplotlib.pyplot as plt

from calib.aux import functions as fc
from calib.aux import paths
from calib.aux.waveform import (robust_template, persistence_hist,
                                persistence_plot)

CALIB_DIR = paths.CALIB_DIR


# Quality cuts grouped into cumulative stages.

CUT_STAGES = [
    ("noise", {
        "noise": ("<", 15),
    }),
    ("amplitude", {
        "sig_max": ("<", 60),      # slides: Amplitude < 60
        "sig_min": (">", -50),     # slides: AmplitudeMin > -50
    }),
    ("preamplitude", {
        "pre_max": ("<", 35),      # slides: PreAmplitude < 35
        "pre_min": (">", -40),     # slides: PreAmplitudeMin > -40
    }),
    ("postamplitude", {
        "post_max": ("<", 50),     # slides: PostAmplitude < 50
        "post_min": (">", -40),    # slides: PostAmplitudeMin > -40
    }),
    # extra anti light-noise cut: the expected maximum of pure noise
    # (sigma~4) is ~13-14 ADC over 240/674 ticks -> thresholds at ~4-5 sigma.
    # Above that it is a spurious pulse (random SPE inside the window).
    ("tight_prepost", {
        "pre_max":  ("<", 12.5),
        "post_max": ("<", 16),
    }),
]

# Run 041256: brighter LED pulses (noise/pre/post amplitudes run ~4x nominal,
# same scaling applied to the read_waveforms.cpp preselection for this run).
CUT_STAGES_041256 = [
    ("noise", {
        "noise": ("<", 60),
    }),
    ("amplitude", {
        "sig_max": ("<", 240),
        "sig_min": (">", -200),
    }),
    ("preamplitude", {
        "pre_max": ("<", 140),
        "pre_min": (">", -160),
    }),
    ("postamplitude", {
        "post_max": ("<", 200),
        "post_min": (">", -160),
    }),
    ("tight_prepost", {
        "pre_max":  ("<", 50),
        "post_max": ("<", 64),
    }),
]

RUN_CUT_STAGES = {
    "041256": CUT_STAGES_041256,
}


def get_cut_stages(run):
    return RUN_CUT_STAGES.get(str(run), CUT_STAGES)


N_PEAKS = 3          # slides: use up to the third peak
N_BASELINE = 200     # pre-signal ticks used to correct the template residual offset


def apply_cut(arr, op, thr):
    return arr < thr if op == "<" else arr > thr


def get_sigma_n(minuit_obj, n):
    """sigma of the n-th fingerplot peak; fallback: sigma1 * sqrt(n)."""
    key = f"sigma{n}"
    if key in minuit_obj.parameters:
        return minuit_obj.values[key]
    return minuit_obj.values["sigma1"] * np.sqrt(n)


# SiPM array label per channel (LArSoft Y0NX map)
ARRAY_LABEL = {2070: "M7_1", 2071: "M7_2", 2080: "M8_1", 2081: "M8_2"}

# Francesca's templates: same ADC scale, same 1024-sample window as ours
FRANCESCA_DIR = "francesca_template"
FRANCESCA_FILES = {
    2070: "template_41236_M7_1.txt",
    2071: "template_41236_M7_2.txt",
    2080: "template_42156_M8_1.txt",
    2081: "template_42156_M8_2.txt",
}


def template_path(run, channel):
    """Canonical SPE template path -- this is what deconvolve/ reads."""
    return paths.out(CALIB_DIR, "data", "templates", run) / f"template_{channel}.npy"


def francesca_template_path(channel):
    """Path of Francesca's template for a channel."""
    return (paths.out(CALIB_DIR, "data", "templates", FRANCESCA_DIR)
            / FRANCESCA_FILES[channel])


def load_spe_template(channel, source):
    """Load the SPE template of a channel.

    source: a calibration run ("039365", "media", ...) or "francesca".
    """
    if str(source).lower() == "francesca":
        return np.loadtxt(francesca_template_path(channel))
    return np.load(template_path(source, channel))


def spe_template(f, m, run):
    templates = {}   # {channel: template_array}
    rng = np.random.default_rng(42)

    cut_stages = get_cut_stages(run)
    keys = fc.latest_cycle_keys(f)

    for key in keys:
        # key like "Metrics_2070;1"
        channel = int(key.split("_")[1].split(";")[0])
        tree = f[key]

        outdir = paths.out(CALIB_DIR, "plots", "spe_template", run, channel)

        integral = tree["integral"].array(library="np")
        # float32: half the memory, precision is plenty for ADC counts
        wf_all = np.stack(
            tree["wf_nobaseline"].array(library="np")).astype(np.float32)
        n_total = len(integral)

        # Cumulative quality cuts

        quality = np.ones(n_total, dtype=bool)
        cutflow = [("all", n_total)]
        missing = []
        for stage_name, stage_cuts in cut_stages:
            for branch, (op, thr) in stage_cuts.items():
                if branch in tree:
                    quality &= apply_cut(
                        tree[branch].array(library="np"), op, thr)
                else:
                    missing.append(branch)
            cutflow.append((stage_name, int(quality.sum())))
        if missing:
            print(f"Channel {channel}: WARNING - missing branches "
                  f"(cuts not applied): {missing}")

        # Windows [q0 + n*gain +- sigma_n]:
        # per-peak persistence before and after the cuts

        q0   = m[channel].values["q0"]
        gain = m[channel].values["gain"]

        wf_norm_before_list = []   # peak window only (no cuts)
        wf_norm_list = []          # window + quality cuts
        n_per_peak = {}
        for n in range(1, N_PEAKS + 1):
            center  = q0 + n * gain
            sigma_n = get_sigma_n(m[channel], n)
            window = (integral > center - sigma_n) \
                   & (integral < center + sigma_n)
            sel = window & quality
            n_per_peak[n] = int(sel.sum())

            if window.sum() > 0:
                persistence_plot(
                    wf_all[window],
                    f"Ch {channel} - {n} p.e. before cuts "
                    f"({int(window.sum())} evts)",
                    outdir / f"persistence_{n}pe_before.png", rng)
                wf_norm_before_list.append(wf_all[window] / n)
            if n_per_peak[n] > 0:
                persistence_plot(
                    wf_all[sel],
                    f"Ch {channel} - {n} p.e. after cuts "
                    f"({n_per_peak[n]} evts)",
                    outdir / f"persistence_{n}pe_after.png", rng)
                wf_norm_list.append(wf_all[sel] / n)  # normalize by n p.e.

        if not wf_norm_list:
            print(f"Channel {channel}: no event selected, skipping")
            continue

        wf_norm_before = np.concatenate(wf_norm_before_list, axis=0)
        wf_sel = np.concatenate(wf_norm_list, axis=0)
        n_events = wf_sel.shape[0]

        # SPE-normalized persistence: without cuts vs with cuts
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        h0 = persistence_hist(
            axes[0], wf_norm_before,
            f"no quality cuts ({wf_norm_before.shape[0]} evts)",
            rng, yrange=(-50, 100))
        h1 = persistence_hist(
            axes[1], wf_sel,
            f"after quality cuts ({n_events} evts)",
            rng, yrange=(-50, 100))
        fig.colorbar(h0[3], ax=axes[0], label="Counts")
        fig.colorbar(h1[3], ax=axes[1], label="Counts")
        fig.suptitle(f"Ch {channel} - normalized by SPE (peaks 1-{N_PEAKS})")
        plt.tight_layout()
        fig.savefig(outdir / "persistence_norm_all.png")
        plt.close(fig)


        # Template: per-tick robust average

        template = robust_template(wf_sel)

        # residual baseline offset correction (bias of the mode-based
        # estimate done in C++): median of the pre-signal ticks
        offset = np.median(template[:N_BASELINE])
        template -= offset

        templates[channel] = template
        np.save(template_path(run, channel), template)

        plt.plot(template)
        plt.xlabel("Sample")
        plt.ylabel("ADC - baseline")
        plt.title(f"Ch {channel} - SPE Template ({n_events} wfs)")
        plt.savefig(outdir / "template.png")
        plt.close()

        print(f"Channel {channel}: residual offset = {offset:.3f} ADC")
        print(f"Channel {channel}: template from {n_events} wfs "
              f"(per peak: {n_per_peak}) | cutflow: "
              + " -> ".join(f"{n}:{c}" for n, c in cutflow))

    return templates
