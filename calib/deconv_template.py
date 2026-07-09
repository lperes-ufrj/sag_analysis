import uproot
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import aux.functions as fc
import os


def deconv_template(f, m, run):
    templates = []

    keys = fc.latest_cycle_keys(f)

    for i, key in enumerate(keys):
        integral = f[key]["integral"].array(library="np")

        q0 = m[i].values["q0"]
        gain = m[i].values["gain"]
        sigma1 = m[i].values["sigma1"]
        center = q0 + gain

        sel = (integral > center - sigma1) & (integral < center + sigma1)

        wf = f[key]["wf_nobaseline"].array(library="np")
        wf_sel = np.stack(wf[sel])

        os.makedirs(f"PLOTS/deconv_template/{run}/", exist_ok=True)

        # histograma 2D (persistencia) dos eventos selecionados, antes da media
        n_events, n_samples = wf_sel.shape
        x = np.tile(np.arange(n_samples), n_events)
        y = wf_sel.flatten()

        plt.hist2d(x, y, bins=[n_samples, 250], range=[[0, n_samples], [-50, 200]],
                    norm="log", cmap="viridis")
        plt.colorbar(label="Counts")
        plt.title(f"{key} - Selected waveforms (1 p.e.)")
        plt.xlabel("Sample")
        plt.ylabel("ADC - baseline")
        plt.savefig(f"PLOTS/deconv_template/{run}/{key}_persistence.png")
        plt.close()

        template = wf_sel.mean(axis=0)
        templates.append(template)

        plt.plot(template)
        plt.title(f"{key} - SPE Template")
        plt.xlabel("Sample")
        plt.ylabel("ADC - baseline")
        plt.savefig(f"PLOTS/deconv_template/{run}/{key}_template.png")
        plt.close()

    return templates