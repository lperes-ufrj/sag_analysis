from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import time
import dunestyle.matplotlib as dunestyle

from iminuit import Minuit
from iminuit.cost import LeastSquares


repo_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.src import read_txt_error_file

# Reuse the run mapping, reference data and LY calculation from the averaged-channel script.
from fit_LY_wo_template import (
    E,
    PD_HD_result,
    PD_HD_err_low,
    PD_HD_err_high,
    relative_s1_previous,
    efield_previous,
    Birks,
    calculate_study_XA,
)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())
SAMPLE_LABELS = ['20261027_135746','20260909_125014']#['20260909_213507','20260909_125014']

CHANNELS = [2070,2071]
OUTPUT_DIR = repo_dir / "analysis" / "LY_vs_EF"

# Ratio plot: same channel, numerator label / denominator label.
RATIO_NUMERATOR_LABEL = '20261027_135746'
RATIO_DENOMINATOR_LABEL = '20260909_125014'


def mc_ratio(num_central, num_upper, num_lower, den_central, den_upper, den_lower,
             n_samples=500_000, seed=42):
    """Ratio of two independent asymmetric measurements; up/low inputs are bounds."""
    rng = np.random.default_rng(seed)

    def draw(central, upper, lower):
        z = rng.standard_normal((n_samples, central.size))
        return central + np.where(z >= 0, (upper - central) * z, (central - lower) * z)

    combined = draw(num_central, num_upper, num_lower) / draw(den_central, den_upper, den_lower)

    low, central, high = np.quantile(
        combined, [0.15865525393145707, 0.5, 0.8413447460685429], axis=0
    )
    return central, high, low


def fit_birks(efields, rel_ly, rel_ly_upper, rel_ly_lower):
    err_low = rel_ly - rel_ly_lower
    err_up = rel_ly_upper - rel_ly

    # Symmetric approximation for the LeastSquares fit.
    err = 0.5 * (err_low + err_up)

    fit_mask = (
        (efields > 0.0)
        & np.isfinite(efields)
        & np.isfinite(rel_ly)
        & np.isfinite(err)
        & (err > 0.0)
    )

    cost_birks = LeastSquares(
        efields[fit_mask],
        rel_ly[fit_mask],
        err[fit_mask],
        Birks,
    )

    m_birks = Minuit(cost_birks,
           B_1 = 1.0,
           k = 0.1,
            )

    m_birks.migrad()
    m_birks.hesse()

    ndof_birks = np.count_nonzero(fit_mask) - m_birks.nfit
    reduced_chi2_birks = m_birks.fval / ndof_birks if ndof_birks > 0 else np.nan

    return m_birks, ndof_birks, reduced_chi2_birks


def fit_text(title, m_birks, ndof_birks, reduced_chi2_birks):
    return (
        rf"$\mathbf{{{title}}}$"
        "\n"
        rf"$B_1 = {m_birks.values['B_1']:.3f} \pm {m_birks.errors['B_1']:.3f}$"
        "\n"
        rf"$k_\epsilon = {m_birks.values['k']:.3f} \pm {m_birks.errors['k']:.3f}$"
        "\n"
        rf"$\chi^2/\mathrm{{ndof}} = {m_birks.fval:.1f}/{ndof_birks}"
        rf" = {reduced_chi2_birks:.2f}$"
        )


def plot_references():
    plt.scatter(
        efield_previous,
        relative_s1_previous,
        marker="x",
        alpha=0.9,
        color='gray',
        label="M8 previous study",
    )
    # ProtoDUNE-HD reference
    plt.errorbar(
        E,
        PD_HD_result,
        yerr=[
            PD_HD_err_low,
            PD_HD_err_high,
        ],
        color="gray",
        fmt=".",
        capsize=1,
        label="PD-HD Data",
        zorder=3,
    )


def finish_plot(output_name):
    plt.xlabel("Electric field (kV/cm)")
    plt.ylabel(rf"$S/S_0$ (LY Normalized)")

    plt.ylim([0.7,1.2])
    dunestyle.Preliminary(x=0.05,y=0.9)
    dunestyle.WIP(x=0.05,y=0.83)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2, frameon=True, fontsize=8)
    plt.savefig(
        OUTPUT_DIR / output_name,
        dpi=300,
        bbox_inches="tight",
    )


def main() -> None:

    x_fit = np.linspace(0,0.9, 200)
    channel_results = []

    for sample_label in SAMPLE_LABELS:

        path_waveforms = (
        repo_dir / "coincidence/selected_waveforms" / sample_label
        )

        for ch in CHANNELS:
            y_err = read_txt_error_file(sample_label, ch)
            efields, rel_ly, rel_ly_upper, rel_ly_lower, ch_found = calculate_study_XA(sample_label, ch, y_err,path_waveforms)
            if ch_found == 0 or rel_ly.size == 0:
                print(f"No data found for channel {ch} in sample {sample_label}. Skipping.")
                continue

            m_birks, ndof_birks, reduced_chi2_birks = fit_birks(efields, rel_ly, rel_ly_upper, rel_ly_lower)
            y_fit_birks = Birks(
                x_fit,
                m_birks.values["B_1"],
                m_birks.values["k"],
            )

            label = f"Ch {ch}" if len(SAMPLE_LABELS) == 1 else f"Ch {ch} ({sample_label})"
            print(
                f"{label}: B_1 = {m_birks.values['B_1']:.4f} +- {m_birks.errors['B_1']:.4f}, "
                f"k = {m_birks.values['k']:.4f} +- {m_birks.errors['k']:.4f}, "
                f"chi2/ndof = {m_birks.fval:.2f}/{ndof_birks} = {reduced_chi2_birks:.2f}"
            )

            channel_results.append({
                "channel": ch,
                "sample_label": sample_label,
                "label": label,
                "efields": efields,
                "rel_ly": rel_ly,
                "rel_ly_upper": rel_ly_upper,
                "rel_ly_lower": rel_ly_lower,
                "y_fit": y_fit_birks,
                "fit": (m_birks, ndof_birks, reduced_chi2_birks),
            })

            # Individual plot for this channel.
            plt.figure(dpi=150)
            plt.grid()
            plt.text(
                0.64,
                0.95,
                fit_text(r"Fit\ Birks:", m_birks, ndof_birks, reduced_chi2_birks),
                transform=plt.gca().transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
            plt.plot(x_fit, y_fit_birks, color="red", ls='--',linewidth=2, label="Birks fit")
            #plot_references()
            plt.errorbar(
                efields,
                rel_ly,
                yerr=[rel_ly - rel_ly_lower, rel_ly_upper - rel_ly],
                fmt=".",
                label=label,
            )
            finish_plot(f"fit_LY_wo_template_comparison_ch_{ch}_{sample_label}_{TIMESTAMP}.png")
            plt.close()

    if not channel_results:
        raise RuntimeError("No valid data found.")

    # Same channel across different sample labels.
    if len(SAMPLE_LABELS) > 1:
        for ch in CHANNELS:
            results_ch = [result for result in channel_results if result["channel"] == ch]
            if len(results_ch) < 2:
                continue
            plot_comparison(
                results_ch,
                x_fit,
                f"fit_LY_wo_template_comparison_ch_{ch}_labels_{'_'.join(SAMPLE_LABELS)}_{TIMESTAMP}.png",
            )
            plt.close()

        plot_label_ratio(channel_results)

    # Comparison plot: every channel with its own Birks fit.
    plot_comparison(
        channel_results,
        x_fit,
        f"fit_LY_wo_template_comparison_chs_{'_'.join(map(str,CHANNELS))}_{TIMESTAMP}.png",
    )
    plt.show()


def plot_comparison(results, x_fit, output_name):
    plt.figure(dpi=150)
    plt.grid()
    plt.ylim([0.7,1.2])
    #plot_references()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        m_birks, ndof_birks, reduced_chi2_birks = result["fit"]
        plt.errorbar(
            result["efields"],
            result["rel_ly"],
            yerr=[
                result["rel_ly"] - result["rel_ly_lower"],
                result["rel_ly_upper"] - result["rel_ly"],
            ],
            fmt=".",
            color=color,
            label=result["label"],
        )
        
        plt.plot(
            x_fit,
            result["y_fit"],
            color=color,
            ls='--',
            linewidth=1.5,
            label=(
                rf"Birks {result['label']}: $B_1={m_birks.values['B_1']:.3f}$, "
                rf"$k_\epsilon={m_birks.values['k']:.3f}$, "
                rf"$\chi^2/\mathrm{{ndof}}={reduced_chi2_birks:.2f}$"
            ),
        )

    finish_plot(output_name)


def plot_label_ratio(channel_results):
    plt.figure(dpi=150)
    plt.grid()
    plt.axhline(1.0, color="gray", ls="--", linewidth=1)

    plotted = False
    for ch in CHANNELS:
        by_label = {
            result["sample_label"]: result
            for result in channel_results
            if result["channel"] == ch
        }
        if RATIO_NUMERATOR_LABEL not in by_label or RATIO_DENOMINATOR_LABEL not in by_label:
            print(f"Channel {ch} missing one of the ratio labels. Skipping ratio.")
            continue

        num = by_label[RATIO_NUMERATOR_LABEL]
        den = by_label[RATIO_DENOMINATOR_LABEL]

        # Only fields measured in both labels; zero field is 1/1 by construction.
        efields, i_num, i_den = np.intersect1d(num["efields"], den["efields"], return_indices=True)
        keep = efields > 0.0
        efields, i_num, i_den = efields[keep], i_num[keep], i_den[keep]
        if efields.size == 0:
            print(f"Channel {ch} has no common electric fields between labels. Skipping ratio.")
            continue

        ratio, ratio_upper, ratio_lower = mc_ratio(
            num["rel_ly"][i_num], num["rel_ly_upper"][i_num], num["rel_ly_lower"][i_num],
            den["rel_ly"][i_den], den["rel_ly_upper"][i_den], den["rel_ly_lower"][i_den],
            seed=42 + ch,
        )

        plt.errorbar(
            efields,
            ratio,
            yerr=[ratio - ratio_lower, ratio_upper - ratio],
            fmt=".",
            label=f"Ch {ch}",
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Electric field (kV/cm)")
    plt.ylabel(rf"$(S/S_0)_{{{RATIO_NUMERATOR_LABEL[:8]}}}\ /\ (S/S_0)_{{{RATIO_DENOMINATOR_LABEL[:8]}}}$")
    dunestyle.Preliminary(x=0.05,y=0.9)
    dunestyle.WIP(x=0.05,y=0.83)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=2, frameon=True, fontsize=8)
    plt.savefig(
        OUTPUT_DIR / f"fit_LY_wo_template_comparison_ratio_{RATIO_NUMERATOR_LABEL}_div_{RATIO_DENOMINATOR_LABEL}"
        f"_chs_{'_'.join(map(str,CHANNELS))}_{TIMESTAMP}.png",
        dpi=300,
        bbox_inches="tight",
    )

if __name__ == "__main__":
    main()
