import uproot
import numpy as np
import matplotlib.pyplot as plt          # faltava: você usa plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import aux.functions as fc               # módulo, não caminho/arquivo

# read selected wfs from read_waveforms
f = uproot.open(".roots/metrics_select.root")

# iterate over all channels
for i in range(len(f.keys())):
    integral = f[f.keys()[i]]["integral"].array(library="np")
    print(f.keys()[i], "->", len(integral), "values.")

    # Raw fingerplot
    x_bin = np.linspace(-200, 1200, 120)
    plt.hist(integral, bins=x_bin, histtype='step', color='black')
    plt.title(f"{f.keys()[i]} - Fingerplot")
    plt.xlabel("Integral")
    plt.ylabel("Counts")
    plt.savefig(f"PLOTS/fingerplot/{f.keys()[i]}_fingerplot.png")
    plt.close()

    # Multigauss fingerplot
    counts, edges = np.histogram(integral, bins=x_bin)   
    centers = (edges[:-1] + edges[1:]) / 2
    y_err = np.sqrt(counts)
    y_err[y_err == 0] = 1                                  

    model = LeastSquares(centers, counts, y_err, fc.multi_gauss)

    initial_guesses = {
    "A0": 200  ,
    "q0": 0 ,
    "sigma0": 20,
    "gain":  200 ,
    "sigma1": 10,
    "sigma2": 20,
    "sigma3": 30,
    "sigma4": 40,
    "sigma5": 50,
    "A1": 200 ,
    "A2": 150,
    "A3": 100 ,
    "A4": 50,
    "A5": 25 ,
    }

    m = Minuit(model, **initial_guesses)

    m.migrad()
    m.hesse()

    # Multigauss plot saved
    x_curve = np.linspace(centers[0], centers[-1], 500)
    y_curve = fc.multi_gauss(x_curve, *m.values)

    spe      = m.values["gain"]          # carga do SPE = espaçamento entre picos
    spe_err  = m.errors["gain"]          # incerteza (do HESSE)
    ndf      = len(centers) - m.nfit     # graus de liberdade = nº pontos - nº params livres
    chi2_ndf = m.fval / ndf              # m.fval é o chi2 do LeastSquares

    label_fit = (f"multigauss fit\n"
                 f"SPE = {spe:.1f} ± {spe_err:.1f}\n"
                 f"$\\chi^2$/ndf = {chi2_ndf:.2f}")

    plt.errorbar(centers, counts, yerr=y_err, fmt='.', color='black', label='data')
    plt.plot(x_curve, y_curve, '-', color='red', label=label_fit)
    plt.title(f"{f.keys()[i]} - Multigauss")
    plt.xlabel("Integral")
    plt.ylabel("Counts")
    plt.legend()
    plt.savefig(f"PLOTS/multigauss/{f.keys()[i]}_multigauss.png")
    plt.close()
