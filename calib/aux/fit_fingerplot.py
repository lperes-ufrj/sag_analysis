import uproot
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from calib.aux import functions as fc
from calib.aux import paths

CALIB_DIR = paths.CALIB_DIR

def fit_multigauss(f, run):
    results = {}   # {channel: Minuit object}

    fp_dir = paths.out(CALIB_DIR, "plots", "fingerplot", run)
    mg_dir = paths.out(CALIB_DIR, "plots", "multigauss", run)

    keys = fc.latest_cycle_keys(f)

    for key in keys:
        # key like "Metrics_2070;1"
        channel = int(key.split("_")[1].split(";")[0])

        integral = f[key]["integral"].array(library="np")

        # Raw fingerplot
        x_bin = np.linspace(-100, 400, 120)
        plt.hist(integral, bins=x_bin, histtype='step', color='black')
        plt.title(f"{key} - Fingerplot")
        plt.xlabel("Integral")
        plt.ylabel("Counts")
        plt.savefig(fp_dir / f"{key}_fingerplot.png")
        plt.close()

        # Multigauss fingerplot
        counts, edges = np.histogram(integral, bins=x_bin)
        centers = (edges[:-1] + edges[1:]) / 2
        y_err = np.sqrt(counts)
        y_err[y_err == 0] = 1

        model = LeastSquares(centers, counts, y_err, fc.multi_gauss)

        initial_guesses = {
        "A0": 140  ,
        "q0": -30 ,
        "sigma0": 20,
        "gain":  150 ,
        "sigma1": 50,
        "sigma2": 50,
        "sigma3": 50,
        "A1": 90 ,
        "A2": 60,
        "A3": 20 ,
        }

        m = Minuit(model, **initial_guesses)

        for a in ("A0", "A1", "A2", "A3"):
            m.limits[a] = (0, None)
        for s in ("sigma0", "sigma1", "sigma2", "sigma3"):
            m.limits[s] = (0, 50)

        m.migrad()
        m.hesse()

        # Multigauss plot saved
        x_curve = np.linspace(centers[0], centers[-1], 500)
        y_curve = fc.multi_gauss(x_curve, *m.values)

        spe      = m.values["gain"]
        spe_err  = m.errors["gain"]
        ndf      = len(centers) - m.nfit
        chi2_ndf = m.fval / ndf

        label_fit = (f"multigauss fit\n"
                    f"SPE = {spe:.1f} ± {spe_err:.1f}\n"
                    f"$\\chi^2$/ndf = {chi2_ndf:.2f}")

        plt.errorbar(centers, counts, yerr=y_err, fmt='.', color='black', label='data')
        plt.plot(x_curve, y_curve, '-', color='red', label=label_fit)
        plt.title(f"{key} - Multigauss")
        plt.xlabel("Integral")
        plt.ylabel("Counts")
        plt.legend()
        plt.savefig(mg_dir / f"{key}_multigauss.png")
        plt.close()

        print(f"Channel {channel}: Multigauss Fitted")
        print(f"  Nsamples = {len(integral)}")
        print(f"  SPE = {spe:.1f} +/- {spe_err:.1f}")
        print(f"  chi^2/ndf = {chi2_ndf:.2f}")
        print(f"  Fit Valid: {m.valid}")
        print()

        results[channel] = m

    return results
